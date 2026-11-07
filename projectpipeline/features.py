"""Shared feature engineering: turn the 5-minute table into model-ready rows."""
import numpy as np
import pandas as pd

STEP_MIN = 5
CO2_GOOD = 1000      # UBA guide value: below this indoor air is hygienically fine
CO2_BAD = 2000       # UBA: above this is hygienically unacceptable
TMP_MIN, TMP_MAX = 20.0, 26.0   # ASR A3.5: seated light work >= 20 C, should not exceed 26 C


def room_aliases(codes=None):
    """Map the original room codes to anonymous names Room 01 ... Room 23.

    The order follows the sorted original codes (read from the local room list, which is not
    published), so rooms on the same floor keep neighbouring numbers. No real code appears in
    this repository.
    """
    import os
    if codes is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        codes = pd.read_excel(os.path.join(root, "Data", "Classroom_List_FK10.xlsx")).dropna(subset=["Room"])["Room"]
    codes = sorted(set(codes))
    return {code: f"Room {i:02d}" for i, code in enumerate(codes, start=1)}


def anonymise_rooms(df):
    """Relabel rooms in a loaded table. Safe on data that is already anonymised."""
    if df["room"].astype(str).str.startswith("Room ").all():
        return df
    names = room_aliases()
    df = df.copy()
    df["room"] = df["room"].map(lambda r: names.get(r, r))
    if "sensor_key" in df:
        df["sensor_key"] = df["room"] + "#" + df["sensor_key"].astype(str).str.split("#").str[-1]
    return df


def complete_grid(df):
    """Give every sensor an unbroken 5-minute time index so that shifts are true time lags."""
    parts = []
    for key, g in df.groupby("sensor_key", sort=False):
        g = g.set_index("time").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq=f"{STEP_MIN}min")
        g = g[~g.index.duplicated()].reindex(idx)
        for c in ["room", "sensor_key", "volume_m3"]:
            g[c] = g[c].ffill().bfill()
        parts.append(g.rename_axis("time").reset_index())
    return pd.concat(parts, ignore_index=True)


def is_occupied_hours(t):
    """Teaching hours: Monday-Friday, 07:30-20:00."""
    mins = t.dt.hour * 60 + t.dt.minute
    return (t.dt.dayofweek < 5) & (mins >= 450) & (mins < 1200)


def add_features(df, horizons=(15, 30, 60)):
    df = complete_grid(df)
    g = df.groupby("sensor_key", sort=False)
    t = df["time"]

    # calendar
    hod = t.dt.hour + t.dt.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    df["dow"] = t.dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["occupied_hours"] = is_occupied_hours(t)

    # recent history (each step is 5 minutes)
    for col in ["co2", "tmp"]:
        for k in [1, 2, 3, 6, 12]:
            df[f"{col}_lag{k*STEP_MIN}"] = g[col].shift(k)
        df[f"{col}_d5"] = df[col] - df[f"{col}_lag5"]
        df[f"{col}_d15"] = df[col] - df[f"{col}_lag15"]
        df[f"{col}_d30"] = df[col] - df[f"{col}_lag30"]
    df["co2_max60"] = g["co2"].transform(lambda s: s.rolling(12, min_periods=3).max())
    df["tmp_mean24h"] = g["tmp"].transform(lambda s: s.rolling(288, min_periods=36).mean())
    df["pir_sum30"] = g["pir"].transform(lambda s: s.rolling(6, min_periods=2).sum())
    df["pir_sum60"] = g["pir"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    df["lux_mean30"] = g["lux"].transform(lambda s: s.rolling(6, min_periods=2).mean())
    df["rlf_d30"] = df["rlf"] - g["rlf"].shift(6)
    # CO2 above outdoor level, scaled by room size: a rough "people per m3" signal
    df["co2_excess_per_m3"] = (df["co2"] - 420) / df["volume_m3"]

    # Same time last week. Teaching follows a weekly timetable, so what happened in this room
    # one week ago (including in the 30-60 minutes after "now") is a strong hint of what comes next.
    W = 7 * 24 * 60 // STEP_MIN
    df["co2_wk"] = g["co2"].shift(W)
    df["co2_wk_d30"] = g["co2"].shift(W - 6) - df["co2_wk"]
    df["co2_wk_d60"] = g["co2"].shift(W - 12) - df["co2_wk"]
    df["pir_wk_next60"] = g["pir"].transform(lambda s: s.shift(W - 12).rolling(12, min_periods=3).sum())
    df["tmp_wk_d60"] = g["tmp"].shift(W - 12) - g["tmp"].shift(W)

    # targets
    for h in horizons:
        k = h // STEP_MIN
        df[f"y_co2_{h}"] = g["co2"].shift(-k)
        df[f"y_tmp_{h}"] = g["tmp"].shift(-k)
    # classification target: will CO2 go above 1000 ppm at any point in the next 30 minutes?
    fut = pd.concat([g["co2"].shift(-k) for k in range(1, 7)], axis=1)
    df["y_co2_max30"] = fut.max(axis=1, skipna=False)
    df["y_alert30"] = (df["y_co2_max30"] > CO2_GOOD).astype(float)
    df.loc[df["y_co2_max30"].isna(), "y_alert30"] = np.nan
    return df


class DeltaModel:
    """Predict the CHANGE from the current reading, then add it back.

    Tree models cannot predict values outside the range they were trained on. Training ran
    January-May and testing ran into a hot summer, so absolute-level targets failed for
    temperature. Predicting the change keeps the target in a stable range.
    """

    def __init__(self, estimator, base_col):
        self.estimator, self.base_col = estimator, base_col

    def fit(self, X, y):
        self.estimator.fit(X, np.asarray(y) - X[self.base_col].to_numpy())
        return self

    def predict(self, X):
        return X[self.base_col].to_numpy() + self.estimator.predict(X)


FULL_FEATURES = [
    "co2", "tmp", "rlf", "lux", "pir", "volume_m3",
    "hour_sin", "hour_cos", "dow", "weekend",
    "co2_lag5", "co2_lag10", "co2_lag15", "co2_lag30", "co2_lag60", "co2_d5", "co2_d15", "co2_d30", "co2_max60",
    "tmp_lag5", "tmp_lag10", "tmp_lag15", "tmp_lag30", "tmp_lag60", "tmp_d5", "tmp_d15", "tmp_d30", "tmp_mean24h",
    "pir_sum30", "pir_sum60", "lux_mean30", "rlf_d30", "co2_excess_per_m3",
    "co2_wk", "co2_wk_d30", "co2_wk_d60", "pir_wk_next60", "tmp_wk_d60",
]
# Features whose meaning does not change between the 2022 and 2023 data platforms
# (motion/light counts are reported differently in 2023, so they are left out for the cross-year test).
PORTABLE_FEATURES = [f for f in FULL_FEATURES if f not in ("lux", "pir", "pir_sum30", "pir_sum60", "lux_mean30", "pir_wk_next60")]
