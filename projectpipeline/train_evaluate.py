"""
Step 3: full evaluation of the chosen (tuned) models.

What is predicted (the re-framed task):
  A. Regression  - CO2 and temperature 15 / 30 / 60 minutes ahead in each classroom.
  B. Classification - "ventilation alert": will CO2 go above 1000 ppm within the next 30 minutes?
Only teaching hours (Mon-Fri 07:30-20:00) are used to train and score, because nights and
weekends are flat, trivially predictable, and would inflate every score.

Evaluation designs:
  1. Linear regression (the original 2023 method, reproduced): same-time temperature from other same-time readings.
  2. Time split: train 18 Jan - 15 May 2022, test 17 May - 24 Sep 2022.
  3. Unseen rooms: rooms held out entirely (5 folds by room), still chronological.
  4. Next year: train on all of 2022, test on 2023 (May - Oct).
Every model is compared with a no-skill baseline (persistence: "the next value equals the current one").
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, root_mean_squared_error)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from features import CO2_GOOD, FULL_FEATURES, PORTABLE_FEATURES, DeltaModel, add_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
RES = os.path.join(ROOT, "results")
SEED = 42
SPLIT_TRAIN_END = pd.Timestamp("2022-05-16")
SPLIT_TEST_START = pd.Timestamp("2022-05-17")   # one-day gap so no window straddles the split
HORIZONS = (15, 30, 60)


# Settings chosen by compare_models.py on the validation weeks (results/tuned_params.json).
# If that file is missing, sensible defaults are used.
DEFAULT_HGB = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=80, l2_regularization=1.0)
_TUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tuned_params.json")
TUNED = json.load(open(_TUNED_PATH)) if os.path.exists(_TUNED_PATH) else {}


def hgb_reg(var="co2"):
    p = TUNED.get(f"hist_gradient_boosting_regressor_{var}_30", DEFAULT_HGB)
    return HistGradientBoostingRegressor(random_state=SEED, **p)


def hgb_clf():
    p = {k: v for k, v in TUNED.get("hist_gradient_boosting_classifier_alert30", DEFAULT_HGB).items() if k != "threshold"}
    return HistGradientBoostingClassifier(random_state=SEED, **p)


def linear(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def reg_metrics(y, p, base=None):
    m = {"MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p), "R2": r2_score(y, p), "n": int(len(y))}
    if base is not None:
        m["skill_vs_persistence"] = 1 - m["MAE"] / mean_absolute_error(y, base)
    return {k: (round(float(v), 4) if k != "n" else v) for k, v in m.items()}


def clf_metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    m = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
         "F1": f1_score(y, pred, zero_division=0), "positives": int(y.sum()), "n": int(len(y))}
    if len(np.unique(y)) > 1 and len(np.unique(score)) > 2:
        m["PR_AUC"] = average_precision_score(y, score)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m.items()}


def best_threshold(y, score):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, score >= t, zero_division=0) for t in grid]))])


# ---------------------------------------------------------------------------------------------
def old_approach(df):
    """Reproduce what the original notebook did, then show what happens with an honest split."""
    cols = ["co2", "lux", "pir", "rlf", "volume_m3"]
    d = df.dropna(subset=cols + ["tmp"]).copy()
    d["sensor_code"] = d["sensor_key"].astype("category").cat.codes   # IDs used as numbers, as before
    X, y = d[cols + ["sensor_code"]], d["tmp"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    r_random = r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))
    tr, te = d["time"] < SPLIT_TRAIN_END, d["time"] >= SPLIT_TEST_START
    r_chrono = r2_score(y[te], LinearRegression().fit(X[tr], y[tr]).predict(X[te]))
    occ = d["occupied_hours"]
    r_chrono_occ = r2_score(y[te & occ], LinearRegression().fit(X[tr], y[tr]).predict(X[te & occ]))
    return {"same_time_temperature_LinearRegression": {
        "random_80_20_split_R2": round(r_random, 4),
        "chronological_split_R2": round(r_chrono, 4),
        "chronological_split_teaching_hours_R2": round(r_chrono_occ, 4)}}


def fit_predict_regression(tr, te, target, feats):
    out = {}
    var = target.replace("y_", "").rsplit("_", 1)[0]
    base = te[var]                                               # persistence: current value
    out["persistence"] = (base.values, None)
    lin = DeltaModel(linear(Ridge(alpha=1.0)), var).fit(tr[feats], tr[target])
    out["ridge"] = (lin.predict(te[feats]), lin)
    hgb = DeltaModel(hgb_reg(var), var).fit(tr[feats], tr[target])
    out["gradient_boosting"] = (hgb.predict(te[feats]), hgb)
    return out, base.values


def scored(te, target, out, base):
    """Metrics on all teaching-hour rows and on rows where people are present (motion in last 30 min)."""
    present = (te["pir_sum30"] > 0).values
    return {name: {"all_teaching_hours": reg_metrics(te[target], p, base),
                   "people_present": reg_metrics(te[target][present], p[present], base[present])}
            for name, (p, _) in out.items()}


def chronological(df, results, preds_store):
    occ = df[df["occupied_hours"]]
    tr_all, te_all = occ[occ["time"] < SPLIT_TRAIN_END], occ[occ["time"] >= SPLIT_TEST_START]
    res = {}
    for var in ["co2", "tmp"]:
        for h in HORIZONS:
            target = f"y_{var}_{h}"
            tr, te = tr_all.dropna(subset=[target, var]), te_all.dropna(subset=[target, var])
            out, base = fit_predict_regression(tr, te, target, FULL_FEATURES)
            res[target] = scored(te, target, out, base)
            if h == 30:
                joblib.dump(out["gradient_boosting"][1], os.path.join(OUT, f"model_{var}_30min.joblib"))
                keep = te[["time", "room", "sensor_key", var, target]].copy()
                keep["pred"] = out["gradient_boosting"][0]
                preds_store[target] = keep
            print(f"  {target}: " + ", ".join(f"{k} MAE={v['all_teaching_hours']['MAE']}/{v['people_present']['MAE']}" for k, v in res[target].items()))
    results["chronological_regression"] = res

    # --- classification: ventilation alert ---
    target = "y_alert30"
    tr, te = tr_all.dropna(subset=[target, "co2"]), te_all.dropna(subset=[target, "co2"])
    val = tr["time"] >= SPLIT_TRAIN_END - pd.Timedelta(days=21)            # last 3 weeks of training = validation
    fit, vset = tr[~val], tr[val]

    logit = linear(LogisticRegression(max_iter=2000, class_weight="balanced"))
    hgb = hgb_clf()
    scores = {}
    for name, model in [("logistic_regression", logit), ("gradient_boosting", hgb)]:
        model.fit(fit[FULL_FEATURES], fit[target])
        thr = best_threshold(vset[target], model.predict_proba(vset[FULL_FEATURES])[:, 1])
        model.fit(tr[FULL_FEATURES], tr[target])                               # refit on all training data
        scores[name] = (model.predict_proba(te[FULL_FEATURES])[:, 1], thr)
        if name == "gradient_boosting":
            joblib.dump({"model": model, "threshold": thr}, os.path.join(OUT, "model_alert_30min.joblib"))
    # rule baselines
    trend = te["co2"] + 6 * te["co2_d5"].fillna(0)
    scores["rule_current_co2_above_1000"] = ((te["co2"] > CO2_GOOD).astype(float).values, 0.5)
    scores["rule_linear_trend_extrapolation"] = ((trend > CO2_GOOD).astype(float).values, 0.5)

    res = {}
    early = (te["co2"] <= CO2_GOOD).values      # rows where the room is still fine right now
    for name, (s, thr) in scores.items():
        res[name] = {"threshold": thr,
                     "all_teaching_hours": clf_metrics(te[target].values, s, thr),
                     "early_warning_rows_co2_now_le_1000": clf_metrics(te[target].values[early], s[early], thr)}
        print(f"  alert {name}: F1={res[name]['all_teaching_hours']['F1']} early-F1={res[name]['early_warning_rows_co2_now_le_1000']['F1']}")
    results["chronological_alert"] = res
    keep = te[["time", "room", "sensor_key", "co2", target]].copy()
    keep["score"] = scores["gradient_boosting"][0]
    preds_store["alert"] = keep


def leakage_check(df, results):
    """Same gradient-boosting CO2 model, random split vs chronological split."""
    occ = df[df["occupied_hours"]].dropna(subset=["y_co2_30", "co2"])
    tr, te = train_test_split(occ, test_size=0.25, random_state=SEED)
    p = DeltaModel(hgb_reg(), "co2").fit(tr[FULL_FEATURES], tr["y_co2_30"]).predict(te[FULL_FEATURES])
    results["leakage_check_co2_30"] = {
        "random_split": reg_metrics(te["y_co2_30"], p, te["co2"]),
        "chronological_split": results["chronological_regression"]["y_co2_30"]["gradient_boosting"]["all_teaching_hours"]}


def unseen_rooms(df, results):
    occ = df[df["occupied_hours"]]
    rooms = occ["room"].unique()
    gkf = GroupKFold(n_splits=5)
    rows_reg, rows_clf = [], []
    for fold, (_, te_idx) in enumerate(gkf.split(rooms, groups=rooms)):
        test_rooms = set(rooms[te_idx])
        tr = occ[(~occ["room"].isin(test_rooms)) & (occ["time"] < SPLIT_TRAIN_END)]
        te = occ[(occ["room"].isin(test_rooms)) & (occ["time"] >= SPLIT_TEST_START)]
        a, b = tr.dropna(subset=["y_co2_30", "co2"]), te.dropna(subset=["y_co2_30", "co2"])
        p = DeltaModel(hgb_reg(), "co2").fit(a[FULL_FEATURES], a["y_co2_30"]).predict(b[FULL_FEATURES])
        rows_reg.append(reg_metrics(b["y_co2_30"], p, b["co2"]))
        a, b = tr.dropna(subset=["y_alert30", "co2"]), te.dropna(subset=["y_alert30", "co2"])
        clf = hgb_clf().fit(a[FULL_FEATURES], a["y_alert30"])
        thr = results["chronological_alert"]["gradient_boosting"]["threshold"]
        rows_clf.append(clf_metrics(b["y_alert30"].values, clf.predict_proba(b[FULL_FEATURES])[:, 1], thr))
        print(f"  fold {fold}: rooms={sorted(test_rooms)} MAE={rows_reg[-1]['MAE']} F1={rows_clf[-1]['F1']}")
    agg = lambda rows, k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                           "std": round(float(np.std([r[k] for r in rows])), 4)}
    results["unseen_rooms_5fold"] = {
        "co2_30": {k: agg(rows_reg, k) for k in ["MAE", "R2", "skill_vs_persistence"]},
        "alert30": {k: agg(rows_clf, k) for k in ["precision", "recall", "F1"]}}


def next_year(df22, results):
    path = os.path.join(OUT, "clean_2023.pkl")
    if not os.path.exists(path):
        return
    d23 = add_features(pd.read_pickle(path))
    tr = df22[df22["occupied_hours"]]
    te = d23[d23["occupied_hours"]]
    res = {}
    a, b = tr.dropna(subset=["y_co2_30", "co2"]), te.dropna(subset=["y_co2_30", "co2", "co2_lag5"])
    p = DeltaModel(hgb_reg(), "co2").fit(a[PORTABLE_FEATURES], a["y_co2_30"]).predict(b[PORTABLE_FEATURES])
    res["co2_30"] = {"persistence": reg_metrics(b["y_co2_30"], b["co2"]),
                     "gradient_boosting": reg_metrics(b["y_co2_30"], p, b["co2"])}
    a, b = tr.dropna(subset=["y_tmp_30", "tmp"]), te.dropna(subset=["y_tmp_30", "tmp", "tmp_lag5"])
    p = DeltaModel(hgb_reg("tmp"), "tmp").fit(a[PORTABLE_FEATURES], a["y_tmp_30"]).predict(b[PORTABLE_FEATURES])
    pr = DeltaModel(linear(Ridge(alpha=1.0)), "tmp").fit(a[PORTABLE_FEATURES], a["y_tmp_30"]).predict(b[PORTABLE_FEATURES])
    res["tmp_30"] = {"persistence": reg_metrics(b["y_tmp_30"], b["tmp"]),
                     "ridge": reg_metrics(b["y_tmp_30"], pr, b["tmp"]),
                     "gradient_boosting": reg_metrics(b["y_tmp_30"], p, b["tmp"])}
    a, b = tr.dropna(subset=["y_alert30", "co2"]), te.dropna(subset=["y_alert30", "co2", "co2_lag5"])
    clf = hgb_clf().fit(a[PORTABLE_FEATURES], a["y_alert30"])
    thr = results["chronological_alert"]["gradient_boosting"]["threshold"]
    res["alert30"] = {"gradient_boosting": clf_metrics(b["y_alert30"].values, clf.predict_proba(b[PORTABLE_FEATURES])[:, 1], thr),
                      "rule_current_co2_above_1000": clf_metrics(b["y_alert30"].values, (b["co2"] > CO2_GOOD).astype(float).values, 0.5)}
    results["next_year_2023"] = res
    print("  2023:", {k: v.get("gradient_boosting") for k, v in res.items()})


# Clear, model-based names used in results/metrics.json
SECTION_NAMES = {
    "old_approach": "linear_regression",
    "chronological_regression": "ridge_regression_and_hist_gradient_boosting_regressor",
    "chronological_alert": "logistic_regression_and_hist_gradient_boosting_classifier",
    "leakage_check_co2_30": "leakage_check_co2_30",
    "unseen_rooms_5fold": "unseen_rooms_5fold",
    "next_year_2023": "next_year_2023",
}
INNER_NAMES = {
    "same_time_temperature_LinearRegression": "predict_current_temperature_from_other_current_readings",
    "chronological_split_R2": "time_split_R2",
    "chronological_split_teaching_hours_R2": "time_split_teaching_hours_R2",
    "chronological_split": "time_split",
    "persistence": "persistence_baseline",
    "ridge": "ridge_regression",
}


def rename(results):
    def walk(obj, section):
        if not isinstance(obj, dict):
            return obj
        out = {}
        for k, v in obj.items():
            if k == "gradient_boosting":
                k = "hist_gradient_boosting_classifier" if section in ("chronological_alert", "alert30") else "hist_gradient_boosting_regressor"
            else:
                k = INNER_NAMES.get(k, k)
            out[k] = walk(v, k if k in ("alert30",) else section)
        return out
    renamed = {SECTION_NAMES[k]: walk(v, k) for k, v in results.items()}
    u = renamed.get("unseen_rooms_5fold")
    if u:
        renamed["unseen_rooms_5fold"] = {"co2_30_hist_gradient_boosting_regressor": u["co2_30"],
                                         "alert30_hist_gradient_boosting_classifier": u["alert30"]}
    return renamed


def main():
    os.makedirs(RES, exist_ok=True)
    print("Building features...")
    df = add_features(pd.read_pickle(os.path.join(OUT, "clean_2022.pkl")))
    results, preds = {}, {}
    print("1. Linear regression (original method)"); results["old_approach"] = old_approach(df); print(" ", results["old_approach"])
    print("2. Ridge regression, HistGradientBoosting regressor and classifier (time split)"); chronological(df, results, preds)
    print("   Leakage check"); leakage_check(df, results)
    print("3. Unseen rooms"); unseen_rooms(df, results)
    print("4. Next year (2023)"); next_year(df, results)
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(rename(results), f, indent=2)
    for k, v in preds.items():
        v.to_pickle(os.path.join(OUT, f"preds_{k}.pkl"))
    print("Saved results/metrics.json")


if __name__ == "__main__":
    main()
