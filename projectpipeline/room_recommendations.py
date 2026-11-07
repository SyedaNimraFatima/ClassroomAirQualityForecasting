"""
Step 4: per-room facts and recommendations.

1. What actually happened in each room during teaching hours (2022):
   share of time above 1000 / 2000 ppm CO2, above 26 C, below 20 C, typical peak.
2. Room physics from the data:
   - air-change rate (ACH) when the room is empty, from how fast CO2 decays in the evening
   - how fast CO2 rises at the start of a busy class -> rough implied number of people
3. A simple, standard CO2 mass-balance model per room:
       dC/dt = n * G * 1e6 / V  -  ACH * (C - C_out)
   used to give: minutes until 1000 ppm for a given class size, and the largest class that
   stays under 1000 ppm for a 45- or 90-minute session without opening windows.

Assumptions (stated in the README): outdoor CO2 = 420 ppm; one seated adult exhales about
G = 0.020 m3 CO2 per hour; well-mixed air; the sensor stands for the whole room.
These are estimates to guide decisions, not measured head counts.
"""
import os

import numpy as np
import pandas as pd

from features import CO2_BAD, CO2_GOOD, TMP_MAX, TMP_MIN, anonymise_rooms, complete_grid, is_occupied_hours

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
RES = os.path.join(ROOT, "results")
C_OUT = 420.0
G = 0.020             # m3 CO2 per person per hour (seated adult, light activity)
DEFAULT_ACH = 0.5     # used when a room has too few clean evening decays


def evening_decay_ach(g):
    """Fit log(C - C_out) over 19:00-22:00 on weekdays where nobody moves and CO2 started > 600 ppm."""
    g = g.set_index("time")
    rates = []
    for day, d in g.groupby(g.index.date):
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        w = d.between_time("19:00", "22:00")
        if len(w) < 30 or w["co2"].isna().mean() > 0.1 or w["pir"].fillna(0).sum() > 0:
            continue
        excess = (w["co2"] - C_OUT).clip(lower=1)
        if excess.iloc[0] < 180:
            continue
        hours = (w.index - w.index[0]).total_seconds() / 3600
        ok = excess > 20
        if ok.sum() < 12:
            continue
        slope = np.polyfit(hours[ok], np.log(excess[ok]), 1)[0]
        if slope < 0:
            rates.append(-slope)
    return (float(np.median(rates)), len(rates)) if len(rates) >= 3 else (np.nan, len(rates))


def busy_rise_rate(g):
    """90th percentile of 15-minute CO2 rise (ppm/min) during teaching hours, rising periods only."""
    rise = (g["co2"] - g["co2"].shift(3)) / 15
    occ = is_occupied_hours(g["time"])
    r = rise[occ & (rise > 2)]
    return float(r.quantile(0.9)) if len(r) > 20 else np.nan


def minutes_to_limit(n, V, ach, c0=450.0, limit=CO2_GOOD):
    S = n * G * 1e6 / V                 # ppm per hour added by people
    c_ss = C_OUT + S / ach              # level the room settles at
    if c_ss <= limit:
        return np.inf
    t = -np.log((c_ss - limit) / (c_ss - c0)) / ach
    return t * 60


def max_people(V, ach, minutes, c0=450.0, limit=CO2_GOOD):
    T = minutes / 60
    e = np.exp(-ach * T)
    n = (limit - C_OUT - (c0 - C_OUT) * e) * ach * V / (G * 1e6 * (1 - e))
    return int(np.floor(n))


def main():
    os.makedirs(RES, exist_ok=True)
    df = complete_grid(anonymise_rooms(pd.read_pickle(os.path.join(OUT, "clean_2022.pkl"))))
    df["occ"] = is_occupied_hours(df["time"])
    rows = []
    for room, r in df.groupby("room"):
        V = float(r["volume_m3"].iloc[0])
        occ = r[r["occ"]]
        sensor_stats = [evening_decay_ach(g) for _, g in r.groupby("sensor_key")]
        achs = [a for a, _ in sensor_stats if not np.isnan(a)]
        ach = float(np.median(achs)) if achs else np.nan
        ach_used = ach if not np.isnan(ach) else DEFAULT_ACH
        rise = np.nanmax([busy_rise_rate(g) for _, g in r.groupby("sensor_key")])
        implied_people = rise * 60 * V / (G * 1e6) if not np.isnan(rise) else np.nan
        co2 = occ["co2"].dropna()
        tmp = occ["tmp"].dropna()
        rows.append({
            "room": room, "volume_m3": V, "sensors": r["sensor_key"].nunique(),
            "pct_teaching_time_co2_gt_1000": round(100 * (co2 > CO2_GOOD).mean(), 1),
            "pct_teaching_time_co2_gt_2000": round(100 * (co2 > CO2_BAD).mean(), 1),
            "co2_p99_ppm": round(co2.quantile(0.99)),
            "pct_teaching_time_tmp_gt_26": round(100 * (tmp > TMP_MAX).mean(), 1),
            "pct_teaching_time_tmp_lt_20": round(100 * (tmp < TMP_MIN).mean(), 1),
            "tmp_p95_C": round(tmp.quantile(0.95), 1),
            "empty_room_ach_per_h": round(ach, 2) if not np.isnan(ach) else None,
            "evening_decays_used": int(sum(n for _, n in sensor_stats)),
            "busy_rise_ppm_per_min_p90": round(rise, 1) if not np.isnan(rise) else None,
            "implied_people_at_busy_rise": int(round(implied_people)) if not np.isnan(implied_people) else None,
            "minutes_to_1000ppm_30_people": round(minutes_to_limit(30, V, ach_used), 0),
            "max_people_45min_under_1000": max_people(V, ach_used, 45),
            "max_people_90min_under_1000": max_people(V, ach_used, 90),
            "max_people_if_aired_every_20min": max_people(V, ach_used, 20),
            "air_changes_per_h_needed_30_people": round(30 * G * 1e6 / (CO2_GOOD - C_OUT) / V, 1),
            "ach_source": "measured" if not np.isnan(ach) else f"default {DEFAULT_ACH}",
        })
    t = pd.DataFrame(rows).sort_values("pct_teaching_time_co2_gt_1000", ascending=False)

    def advice(x):
        out = []
        if x["pct_teaching_time_co2_gt_2000"] >= 1:
            out.append("CO2 reaches unacceptable levels: open windows fully every 20 min, or cap class size")
        elif x["pct_teaching_time_co2_gt_1000"] >= 5:
            out.append("Ventilate mid-class (about every 20 min) and in every break")
        out.append(f"Keep below 1000 ppm: max ~{x['max_people_45min_under_1000']} people for 45 min with windows shut, "
                   f"~{x['max_people_if_aired_every_20min']} if fully aired every 20 min")
        if x["pct_teaching_time_tmp_gt_26"] >= 5:
            out.append("Often above 26 C: shading, cooling or early-morning airing")
        if x["pct_teaching_time_tmp_lt_20"] >= 5:
            out.append("Often below 20 C: check heating")
        return "; ".join(out)
    t["recommendation"] = t.apply(advice, axis=1)
    t.to_csv(os.path.join(RES, "room_recommendations.csv"), index=False)
    print(t.drop(columns="recommendation").to_string(index=False))


if __name__ == "__main__":
    main()
