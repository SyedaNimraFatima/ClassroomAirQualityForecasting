"""
Step 1: load and tidy the classroom sensor data.

Inputs  (relative to project root):
  Data/DF22/<DataSetID>.csv          2022, 1-minute, one file per room (built by the old notebook)
  Data/Sensor Data 2023/*.csv        2023, long format (phenomenonTime, result, name, thing_id)
  Data/Classroom_List_FK10.xlsx      room, DataSetID, sensor, Thing ID, room volume

Output:
  outputs/clean_2022.pkl, outputs/clean_2023.pkl
  One row per sensor per 5 minutes with columns:
  time, room, sensor_key, volume_m3, co2, tmp, rlf, lux, pir
"""
import glob
import os

import numpy as np
import pandas as pd

from features import room_aliases

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
OUT = os.path.join(ROOT, "outputs")
FREQ = "5min"

# Physical sensor limits from Data/sensor_info.txt (Elsys ERS-CO2)
LIMITS = {"co2": (300, 10000), "tmp": (0, 50), "rlf": (0, 100), "lux": (0, 2000), "pir": (0, 256)}


def load_rooms():
    rooms = pd.read_excel(os.path.join(DATA, "Classroom_List_FK10.xlsx"))
    rooms = rooms.dropna(subset=["Room"]).copy()
    rooms["DataSetID"] = rooms["DataSetID"].astype(int)
    rooms["Thing ID"] = rooms["Thing ID"].astype(int)
    # SensorNumber = order of appearance within a DataSetID (same rule as the original notebook)
    rooms["SensorNumber"] = rooms.groupby("DataSetID").cumcount() + 1
    rooms = rooms.rename(columns={"Room Volume": "volume_m3", "Room": "room"})
    # Anonymise room codes: the original codes are replaced by Room 01 ... Room 23 (in sorted
    # order of the original codes, so rooms on the same floor keep neighbouring numbers).
    names = room_aliases(rooms["room"])
    rooms["room"] = rooms["room"].map(names)
    return rooms[["room", "DataSetID", "SensorNumber", "Thing ID", "volume_m3", "Comments"]]


AUDIT = []   # one row per year x measurement, saved to results/data_audit.csv


def apply_limits(df, year=None):
    for col, (lo, hi) in LIMITS.items():
        if col in df:
            bad = (df[col] < lo) | (df[col] > hi)
            if year:
                AUDIT.append({"year": year, "measure": col, "readings_in": int(df[col].notna().sum()),
                              "removed_out_of_range": int(bad.sum()), "allowed_range": f"{lo} to {hi}"})
            df.loc[bad, col] = np.nan
    return df


def resample_sensor(g):
    """1-minute (or irregular) readings -> 5-minute means; PIR -> 5-minute sum."""
    g = g.set_index("time").sort_index()
    agg = {c: "mean" for c in ["co2", "tmp", "rlf", "lux"] if c in g}
    if "pir" in g:
        agg["pir"] = "sum"
    r = g.resample(FREQ).agg(agg)
    if "pir" in r:
        # sum() turns an all-missing window into 0; put NaN back where there was no data
        n = g["pir"].resample(FREQ).count()
        r.loc[n == 0, "pir"] = np.nan
    return r


def build_2022(rooms):
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "DF22", "*.csv"))):
        d = pd.read_csv(f, parse_dates=["Time"])
        d = d.rename(columns={"Time": "time", "CO2": "co2", "TMP": "tmp", "RLF": "rlf", "LUX": "lux", "PIR": "pir"})
        d = apply_limits(d, "2022")
        for (ds, sn), g in d.groupby(["DataSetID", "SensorNumber"]):
            r = resample_sensor(g)
            r["DataSetID"], r["SensorNumber"] = int(ds), int(sn)
            frames.append(r.reset_index())
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(rooms, on=["DataSetID", "SensorNumber"], how="inner")
    df["sensor_key"] = df["room"] + "#" + df["SensorNumber"].astype(str)
    return df


def build_2023(rooms):
    names = {"CO2": "co2", "Room Temperature": "tmp", "Humidity": "rlf", "Light": "lux", "Motion": "pir"}
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, "Sensor Data 2023", "*.csv"))):
        d = pd.read_csv(f)
        if d.empty:
            continue
        frames.append(d[d["name"].isin(names)])
    d = pd.concat(frames, ignore_index=True)
    n_before = len(d)
    d = d.drop_duplicates(["phenomenonTime", "thing_id", "name"])
    print(f"2023: {n_before - len(d)} exact duplicate readings removed")
    d["time"] = pd.to_datetime(d["phenomenonTime"], utc=True, format="ISO8601")
    # 2023 timestamps are UTC; 2022 are local German time. Convert so hours mean the same thing.
    d["time"] = d["time"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    d["var"] = d["name"].map(names)
    raw = d.pivot_table(index=d.index, columns="var", values="result")   # one column per measure, raw readings
    raw = apply_limits(raw, "2023")
    d["result"] = raw.stack().reindex(list(zip(d.index, d["var"]))).to_numpy()
    d = d.dropna(subset=["result"])
    out = []
    for (thing, var), g in d.groupby(["thing_id", "var"]):
        s = g.set_index("time")["result"].sort_index()
        s = s.resample(FREQ).sum(min_count=1) if var == "pir" else s.resample(FREQ).mean()
        out.append(s.rename(var).to_frame().assign(thing_id=int(thing)).reset_index())
    long = pd.concat(out).melt(id_vars=["time", "thing_id"], var_name="var").dropna(subset=["value"])
    wide = long.pivot_table(index=["time", "thing_id"], columns="var", values="value").reset_index()
    wide.columns.name = None
    wide = apply_limits(wide)   # no-op safety check after averaging
    wide = wide.merge(rooms, left_on="thing_id", right_on="Thing ID", how="inner")
    wide["sensor_key"] = wide["room"] + "#" + wide["SensorNumber"].astype(str)
    return wide


def main():
    os.makedirs(OUT, exist_ok=True)
    rooms = load_rooms()
    cols = ["time", "room", "sensor_key", "volume_m3", "co2", "tmp", "rlf", "lux", "pir"]
    d22 = build_2022(rooms)[cols].sort_values(["sensor_key", "time"])
    d22.to_pickle(os.path.join(OUT, "clean_2022.pkl"))
    print("2022:", d22.shape, d22.time.min(), "->", d22.time.max(), d22.sensor_key.nunique(), "sensors")
    d23 = build_2023(rooms)
    d23 = d23[[c for c in cols if c in d23]].sort_values(["sensor_key", "time"])
    d23.to_pickle(os.path.join(OUT, "clean_2023.pkl"))
    print("2023:", d23.shape, d23.time.min(), "->", d23.time.max(), d23.sensor_key.nunique(), "sensors")

    audit = pd.DataFrame(AUDIT).groupby(["year", "measure", "allowed_range"], as_index=False).sum()
    for year, d in [("2022", d22), ("2023", d23)]:
        for m in ["co2", "tmp", "rlf", "lux", "pir"]:
            sel = (audit.year == year) & (audit.measure == m)
            audit.loc[sel, "five_min_values_out"] = int(d[m].notna().sum())
            audit.loc[sel, "sensors"] = d.loc[d[m].notna(), "sensor_key"].nunique()
    audit["pct_removed"] = (100 * audit.removed_out_of_range / audit.readings_in).round(3)
    names = {"co2": "CO2 (ppm)", "tmp": "temperature (C)", "rlf": "humidity (%)", "lux": "light (lux)", "pir": "motion (count)"}
    audit["measure"] = audit["measure"].map(names)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    audit.to_csv(os.path.join(ROOT, "results", "data_audit.csv"), index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
