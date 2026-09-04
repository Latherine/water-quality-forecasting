"""Make a 4-hour-ahead water quality forecast.

This module applies whichever model won for each site and indicator in
`04_evaluation_comparison.ipynb`. For most cases that model is the baseline, so
the "forecast" really is just the current reading — which is the whole point of
the project.

Typical use:

    import pandas as pd
    from forecast import forecast

    recent = pd.read_pickle("processed/CAU_NGA_hourly.pkl")["values"].tail(48)
    result = forecast("CAU_NGA", "tss", recent)
    print(result)

Run `python forecast.py` to see it work on the processed data.
"""

import os

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Settings. These must match the ones used in 03_model_training.ipynb.
# ---------------------------------------------------------------------------
HORIZON_HOURS = 4
LAG_HOURS = [0, 1, 2, 3, 4, 6, 12, 24]
LOOKBACK_HOURS = 24

DECISION_FILE = "results/final_decision.csv"
MODELS_DIR = "models"
FEATURES_FILE = "processed/selected_features.json"

SITE_NAME = {"BAY_MAU": "Bay Mau", "CAU_NGA": "Cau Nga", "HO_TAY": "Ho Tay"}
NAME_TO_SITE = {name: key for key, name in SITE_NAME.items()}

# How many hours of history each kind of model needs before it can predict.
HOURS_NEEDED = {
    "persistence": 1,
    "train_mean": 1,
    "moving_average": HORIZON_HOURS,
    "seasonal_naive": 24 - HORIZON_HOURS + 1,
    "drift": HORIZON_HOURS + 1,
    "ridge": LOOKBACK_HOURS + 1,
    "gradient_boosting": LOOKBACK_HOURS + 1,
}


def chosen_model(site, indicator):
    """Return the model name that notebook 04 decided to keep for this case."""
    if not os.path.exists(DECISION_FILE):
        raise FileNotFoundError(
            DECISION_FILE + " not found. Run 04_evaluation_comparison.ipynb first.")

    decisions = pd.read_csv(DECISION_FILE)
    match = decisions[(decisions.site == SITE_NAME[site])
                      & (decisions.indicator == indicator)]

    if len(match) == 0:
        raise ValueError("No decision recorded for " + site + " / " + indicator)

    return match.iloc[0]["keep_this_model"]


def build_features(recent, features, feature_columns):
    """Rebuild the same feature row that notebook 03 trained on.

    `recent` is an hourly table ending at the moment we forecast from. Only the
    final row is returned, because that is the one we predict from.
    """
    table = pd.DataFrame(index=recent.index)

    for column in features:
        for lag in LAG_HOURS:
            table[column + "_lag" + str(lag)] = recent[column].shift(lag)
        table[column + "_mean_last_4h"] = recent[column].rolling(HORIZON_HOURS).mean()
        table[column + "_change_last_4h"] = (recent[column]
                                             - recent[column].shift(HORIZON_HOURS))

    table["hour_sin"] = np.sin(2 * np.pi * recent.index.hour / 24)
    table["hour_cos"] = np.cos(2 * np.pi * recent.index.hour / 24)
    table["day_of_week"] = recent.index.dayofweek

    # Reindex to the exact training column order. Any mismatch shows up here
    # as a missing column rather than as a silently wrong prediction.
    table = table.reindex(columns=feature_columns)

    last_row = table.iloc[[-1]]
    if last_row.isna().any(axis=1).iloc[0]:
        missing = last_row.columns[last_row.isna().iloc[0]].tolist()
        raise ValueError(
            "Cannot build features - these are missing: " + str(missing[:5])
            + (" ..." if len(missing) > 5 else "")
            + ". Provide more history, or history with fewer gaps.")

    return last_row


def forecast(site, indicator, recent, model_name=None):
    """Forecast one indicator 4 hours after the last row of `recent`.

    site       : "BAY_MAU", "CAU_NGA" or "HO_TAY"
    indicator  : "nh4", "cod" or "tss"
    recent     : hourly DataFrame with a DatetimeIndex, ending at the moment
                 you are forecasting from
    model_name : override the chosen model (mainly useful for comparison)

    Returns a dictionary with the forecast, the time it applies to, and the
    model used.
    """
    if site not in SITE_NAME:
        raise ValueError("Unknown site: " + str(site))
    if indicator not in ["nh4", "cod", "tss"]:
        raise ValueError("Unknown indicator: " + str(indicator))

    if model_name is None:
        model_name = chosen_model(site, indicator)

    needed = HOURS_NEEDED.get(model_name, LOOKBACK_HOURS + 1)
    if len(recent) < needed:
        raise ValueError(model_name + " needs at least " + str(needed)
                         + " hours of history, got " + str(len(recent)))

    forecast_from = recent.index[-1]
    valid_for = forecast_from + pd.Timedelta(hours=HORIZON_HOURS)
    series = recent[indicator]

    # The most recent hour does not always contain a reading, so find the last
    # one that does. If we used the final row blindly we would return NaN.
    measured = series.dropna()
    if len(measured) == 0:
        raise ValueError("No " + indicator + " readings at all in the history given "
                         "for " + SITE_NAME[site] + ".")

    last_reading_at = measured.index[-1]
    reading_age = (forecast_from - last_reading_at) / pd.Timedelta(hours=1)

    def value_at(timestamp):
        """Look a value up by time, and say clearly if it is missing."""
        if timestamp not in series.index or pd.isna(series.loc[timestamp]):
            raise ValueError(model_name + " needs the reading at " + str(timestamp)
                             + ", but it is missing.")
        return series.loc[timestamp]

    # ---- The baselines are rules, so we simply apply them ----------------
    if model_name == "persistence":
        value = measured.iloc[-1]

    elif model_name == "moving_average":
        window_starts = forecast_from - pd.Timedelta(hours=HORIZON_HOURS - 1)
        window = series.loc[window_starts:forecast_from]
        if window.notna().sum() == 0:
            raise ValueError("No readings in the last " + str(HORIZON_HOURS) + " hours.")
        value = window.mean()          # mean() ignores missing hours

    elif model_name == "seasonal_naive":
        value = value_at(forecast_from - pd.Timedelta(hours=24 - HORIZON_HOURS))

    elif model_name == "drift":
        now = value_at(forecast_from)
        before = value_at(forecast_from - pd.Timedelta(hours=HORIZON_HOURS))
        value = now + (now - before)

    # ---- The machine-learning models are loaded from disk ----------------
    elif model_name in ["ridge", "gradient_boosting"]:
        path = os.path.join(MODELS_DIR,
                            site + "_" + indicator + "_" + model_name + ".pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                path + " not found. Run 03_model_training.ipynb to create it.")

        bundle = joblib.load(path)
        row = build_features(recent, bundle["features"], bundle["feature_columns"])

        if model_name == "ridge":
            value = bundle["model"].predict(bundle["scaler"].transform(row))[0]
        else:
            value = bundle["model"].predict(row)[0]

    else:
        raise ValueError("Cannot apply model: " + str(model_name))

    if pd.isna(value):
        raise ValueError("Forecast came out missing for " + SITE_NAME[site]
                         + " / " + indicator + ". Check the history given.")

    return {
        "site": SITE_NAME[site],
        "indicator": indicator,
        "forecast": float(value),
        "forecast_from": forecast_from,
        "valid_for": valid_for,
        "model_used": model_name,
        "last_reading_at": last_reading_at,
        "reading_age_hours": round(reading_age, 1),
    }


def forecast_all(recent_by_site, indicators=("nh4", "cod", "tss")):
    """Forecast every indicator at every site, returning one tidy table.

    `recent_by_site` maps a site key to its recent hourly readings.
    """
    rows = []
    for site, recent in recent_by_site.items():
        for indicator in indicators:
            rows.append(forecast(site, indicator, recent))
    return pd.DataFrame(rows)


def _demo():
    """Run a forecast for every site and indicator using the processed data."""
    recent_by_site = {}
    for site in SITE_NAME:
        path = "processed/" + site + "_hourly.pkl"
        if not os.path.exists(path):
            print("Missing", path, "- run 02_data_preprocessing.ipynb first.")
            return
        # The last 48 hours is more than any model needs.
        recent_by_site[site] = pd.read_pickle(path)["values"].tail(48)

    results = forecast_all(recent_by_site)

    print("Forecasts", HORIZON_HOURS, "hours ahead")
    print()
    print(results.to_string(index=False))
    print()
    used = results["model_used"].value_counts()
    print("Models used:")
    for name, count in used.items():
        print("   ", name, "->", count, "of", len(results), "forecasts")


if __name__ == "__main__":
    _demo()
