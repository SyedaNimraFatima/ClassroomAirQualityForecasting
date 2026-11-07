"""
Use the trained models for one room at one moment, the way a building dashboard would.

Example:
    python pipeline/predict_room.py --room "Room 01" --time "2022-06-20 10:30"
"""
import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CO2_BAD, CO2_GOOD, FULL_FEATURES, TMP_MAX, TMP_MIN, add_features, anonymise_rooms  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, RES = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--time", required=True, help="local time, e.g. '2022-06-20 10:30'")
    ap.add_argument("--data", default=os.path.join(OUT, "clean_2022.pkl"))
    a = ap.parse_args()

    t = pd.Timestamp(a.time).floor("5min")
    d = anonymise_rooms(pd.read_pickle(a.data))
    d = d[(d.room == a.room) & (d.time > t - pd.Timedelta(days=8)) & (d.time <= t)]
    if d.empty:
        sys.exit(f"No data for {a.room} before {t}")
    f = add_features(d)
    now = f[f.time == t]
    co2_m, tmp_m = joblib.load(os.path.join(OUT, "model_co2_30min.joblib")), joblib.load(os.path.join(OUT, "model_tmp_30min.joblib"))
    alert = joblib.load(os.path.join(OUT, "model_alert_30min.joblib"))
    co2_now, tmp_now = now["co2"].mean(), now["tmp"].mean()
    co2_30 = co2_m.predict(now[FULL_FEATURES]).mean()
    tmp_30 = tmp_m.predict(now[FULL_FEATURES]).mean()
    p_alert = alert["model"].predict_proba(now[FULL_FEATURES])[:, 1].max()

    print(f"{a.room} at {t:%a %d %b %Y %H:%M}")
    print(f"  CO2 now {co2_now:.0f} ppm -> forecast in 30 min {co2_30:.0f} ppm")
    print(f"  Temperature now {tmp_now:.1f} C -> forecast in 30 min {tmp_30:.1f} C")
    print(f"  Chance CO2 passes {CO2_GOOD} ppm in the next 30 min: {100 * p_alert:.0f}%")
    actions = []
    if co2_now > CO2_BAD:
        actions.append("Open all windows fully now (CO2 above 2000 ppm)")
    elif co2_30 > CO2_BAD:
        actions.append("Open windows now: CO2 is forecast to pass 2000 ppm within 30 minutes")
    elif p_alert >= alert["threshold"] or co2_now > CO2_GOOD:
        actions.append("Plan to air the room within the next 15-20 minutes")
    if max(tmp_now, tmp_30) > TMP_MAX:
        actions.append("Room is above 26 C: shade windows, use cooling/fans, air early in the day")
    if min(tmp_now, tmp_30) < TMP_MIN:
        actions.append("Room is below 20 C: check heating; keep airing short (3-5 min, windows wide open)")
    recs = os.path.join(RES, "room_recommendations.csv")
    if os.path.exists(recs):
        r = pd.read_csv(recs).set_index("room").loc[a.room]
        actions.append(f"Capacity guide: about {r.max_people_45min_under_1000} people with windows shut for 45 min, "
                       f"{r.max_people_if_aired_every_20min} if fully aired every 20 min")
    print("  Advice:\n   - " + "\n   - ".join(actions or ["No action needed"]))


if __name__ == "__main__":
    main()
