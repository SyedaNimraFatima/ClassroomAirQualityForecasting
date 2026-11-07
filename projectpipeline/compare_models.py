"""
Step 2: tune and compare model families on exactly the same task.

Tasks (2022 data, teaching hours only, same rows for every model):
  - CO2 30 minutes ahead               (regression)
  - temperature 30 minutes ahead       (regression)
  - ventilation alert: will CO2 pass 1000 ppm in the next 30 minutes?  (classification)

Periods:
  fit         18 Jan - 24 Apr 2022   models are trained here while tuning
  validation  25 Apr - 15 May 2022   tuning choices are judged here (never on the test period)
  test        17 May - 24 Sep 2022   final scores; each model is refit on fit+validation first

Model families:
  persistence baseline, Ridge / LogisticRegression, RandomForest, HistGradientBoosting,
  SARIMAX (ARIMA with Fourier daily seasonality, one model per sensor), SVR (10,000-row sample),
  and three deep-learning models with identical inputs: LSTM, TCN and Transformer (PyTorch).

Outputs:
  results/model_comparison.json, results/tuned_params.json, results/feature_importance.csv
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CO2_GOOD, FULL_FEATURES, DeltaModel, add_features

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, RES = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "results")
SEED = 42
FIT_END = pd.Timestamp("2022-04-25")
TRAIN_END = pd.Timestamp("2022-05-16")
TEST_START = pd.Timestamp("2022-05-17")
REG_TARGETS = {"co2_30": ("y_co2_30", "co2"), "tmp_30": ("y_tmp_30", "tmp")}
ALERT = "y_alert30"
rng = np.random.RandomState(SEED)


# ------------------------------------------------------------------ helpers
def reg_scores(y, p, base, present):
    out = {}
    for name, m in [("all_teaching_hours", np.ones(len(y), bool)), ("people_present", present)]:
        mae, mae_b = mean_absolute_error(y[m], p[m]), mean_absolute_error(y[m], base[m])
        out[name] = {"MAE": round(float(mae), 4), "R2": round(float(r2_score(y[m], p[m])), 4),
                     "skill_vs_persistence": round(float(1 - mae / mae_b), 4)}
    return out


def clf_scores(y, s, thr, early):
    out = {}
    for name, m in [("all_teaching_hours", np.ones(len(y), bool)), ("early_warning_rows_co2_now_le_1000", early)]:
        pred = s[m] >= thr
        d = {"precision": precision_score(y[m], pred, zero_division=0), "recall": recall_score(y[m], pred, zero_division=0),
             "F1": f1_score(y[m], pred, zero_division=0)}
        if len(np.unique(s[m])) > 2:
            d["PR_AUC"] = average_precision_score(y[m], s[m])
        out[name] = {k: round(float(v), 4) for k, v in d.items()}
    return out


def best_thr(y, s):
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.995, 120)))
    f = [f1_score(y, s >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def linear(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def rf_pipe(m):
    return make_pipeline(SimpleImputer(strategy="median"), m)


# ------------------------------------------------------------------ data
def load():
    df = add_features(pd.read_pickle(os.path.join(OUT, "clean_2022.pkl")))
    df = df.reset_index(drop=True)
    df["pos"] = df.groupby("sensor_key", sort=False).cumcount()
    occ = df["occupied_hours"] & df["co2"].notna() & df["tmp"].notna()
    need = df["y_co2_30"].notna() & df["y_tmp_30"].notna() & df[ALERT].notna()
    rows = df.index[occ & need & (df["pos"] >= 11)]
    t = df.loc[rows, "time"]
    split = {"fit": rows[t < FIT_END], "val": rows[(t >= FIT_END) & (t < TRAIN_END)], "test": rows[t >= TEST_START]}
    split["train"] = split["fit"].append(split["val"])
    return df, split


# ------------------------------------------------------------------ tabular families
def tabular_family(df, split, results, tuned):
    X = lambda idx: df.loc[idx, FULL_FEATURES]
    te = split["test"]
    present = (df.loc[te, "pir_sum30"] > 0).to_numpy()
    early = (df.loc[te, "co2"] <= CO2_GOOD).to_numpy()

    spaces = {
        "ridge_regression": [{"alpha": a} for a in [0.1, 1, 10, 100]],
        "random_forest": [{"max_depth": d, "min_samples_leaf": l} for d, l in [(12, 50), (20, 20), (None, 10)]],
        "hist_gradient_boosting": list(ParameterSampler({
            "learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [15, 31, 48, 63], "min_samples_leaf": [20, 80, 200],
            "l2_regularization": [0.0, 1.0, 5.0], "max_iter": [200, 400, 700]}, n_iter=10, random_state=SEED)),
    }

    def make_reg(fam, p):
        if fam == "ridge_regression":
            return linear(Ridge(**p))
        if fam == "random_forest":
            return rf_pipe(RandomForestRegressor(n_estimators=100, max_samples=0.3, n_jobs=-1, random_state=SEED, **p))
        return HistGradientBoostingRegressor(random_state=SEED, **p)

    def make_clf(fam, p):
        if fam == "ridge_regression":       # linear family for classification = logistic regression (C = 1/alpha)
            return linear(LogisticRegression(C=1 / p["alpha"], max_iter=2000, class_weight="balanced"))
        if fam == "random_forest":
            return rf_pipe(RandomForestClassifier(n_estimators=100, max_samples=0.3, n_jobs=-1, random_state=SEED,
                                                  class_weight="balanced_subsample", **p))
        return HistGradientBoostingClassifier(random_state=SEED, **p)

    for task, (target, base) in REG_TARGETS.items():
        y_te, b_te = df.loc[te, target].to_numpy(), df.loc[te, base].to_numpy()
        results[task]["persistence_baseline"] = {"scores": reg_scores(y_te, b_te, b_te, present)}
        for fam, space in spaces.items():
            t0 = time.time()
            val_mae = []
            for p in space:
                m = DeltaModel(make_reg(fam, p), base).fit(X(split["fit"]), df.loc[split["fit"], target])
                val_mae.append(mean_absolute_error(df.loc[split["val"], target], m.predict(X(split["val"]))))
            best = space[int(np.argmin(val_mae))]
            t1 = time.time()
            m = DeltaModel(make_reg(fam, best), base).fit(X(split["train"]), df.loc[split["train"], target])
            fit_s = time.time() - t1
            t2 = time.time(); p_te = m.predict(X(te)); pred_s = time.time() - t2
            name = {"ridge_regression": "ridge_regression", "random_forest": "random_forest_regressor",
                    "hist_gradient_boosting": "hist_gradient_boosting_regressor"}[fam]
            results[task][name] = {"best_params": best, "configs_tried": len(space),
                                   "validation_MAE_best": round(float(min(val_mae)), 4),
                                   "fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 2),
                                   "scores": reg_scores(y_te, p_te, b_te, present)}
            tuned[f"{name}_{task}"] = best
            if fam == "hist_gradient_boosting":
                FINAL[task] = m
            print(f"  {task} {name}: best {best} val MAE {min(val_mae):.3f} test MAE "
                  f"{results[task][name]['scores']['all_teaching_hours']['MAE']} ({time.time() - t0:.0f}s)", flush=True)

    # alert
    y_te = df.loc[te, ALERT].to_numpy().astype(int)
    trend = (df.loc[te, "co2"] + 6 * df.loc[te, "co2_d5"].fillna(0)).to_numpy()
    results["alert30"]["rule_current_co2_above_1000"] = {"scores": clf_scores(y_te, df.loc[te, "co2"].to_numpy(), CO2_GOOD + 1e-9, early)}
    results["alert30"]["rule_linear_trend_extrapolation"] = {"scores": clf_scores(y_te, trend, CO2_GOOD + 1e-9, early)}
    y_fit, y_val = df.loc[split["fit"], ALERT], df.loc[split["val"], ALERT].to_numpy().astype(int)
    for fam, space in spaces.items():
        t0 = time.time()
        fits = []
        for p in space:
            m = make_clf(fam, p).fit(X(split["fit"]), y_fit)
            s = m.predict_proba(X(split["val"]))[:, 1]
            thr = best_thr(y_val, s)
            fits.append((f1_score(y_val, s >= thr), thr, p))
        f1v, thr, best = max(fits, key=lambda z: z[0])
        t1 = time.time()
        m = make_clf(fam, best).fit(X(split["train"]), df.loc[split["train"], ALERT])
        fit_s = time.time() - t1
        t2 = time.time(); s_te = m.predict_proba(X(te))[:, 1]; pred_s = time.time() - t2
        name = {"ridge_regression": "logistic_regression", "random_forest": "random_forest_classifier",
                "hist_gradient_boosting": "hist_gradient_boosting_classifier"}[fam]
        bp = {"C": 1 / best["alpha"]} if fam == "ridge_regression" else best
        results["alert30"][name] = {"best_params": bp, "threshold": round(thr, 4), "configs_tried": len(space),
                                    "validation_F1_best": round(float(f1v), 4), "fit_seconds": round(fit_s, 1),
                                    "predict_seconds": round(pred_s, 2), "scores": clf_scores(y_te, s_te, thr, early)}
        tuned[f"{name}_alert30"] = {**bp, "threshold": thr}
        if fam == "hist_gradient_boosting":
            FINAL["alert30"] = m
        print(f"  alert {name}: best {bp} val F1 {f1v:.3f} test F1 "
              f"{results['alert30'][name]['scores']['all_teaching_hours']['F1']} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ SARIMAX (one model per sensor)
def fourier(idx, K=3):
    h = (idx.hour * 60 + idx.minute) / 1440
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"], cols[f"cos{k}"] = np.sin(2 * np.pi * k * h), np.cos(2 * np.pi * k * h)
    cols["weekend"] = (idx.dayofweek >= 5).astype(float)
    return pd.DataFrame(cols, index=idx)


def sarimax_forecasts(series, train_end, order, fit_days=28, horizons=range(1, 7)):
    """Fit on the last `fit_days` before train_end, then filter the later data with fixed
    parameters and return h-step-ahead forecasts made at every time step (no refitting)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    s = series.asfreq("5min")
    fit = s[train_end - pd.Timedelta(days=fit_days):train_end - pd.Timedelta(minutes=5)]
    model = SARIMAX(fit, exog=fourier(fit.index), order=order)
    res = model.fit(disp=False, maxiter=50)
    later = s[train_end - pd.Timedelta(days=2):]
    r2 = res.apply(later, exog=fourier(later.index))
    a = r2.filtered_state                                       # state after seeing data up to each step
    Z, T = np.asarray(r2.model["design"]), np.asarray(r2.model["transition"])
    Z, T = (Z[:, :, 0] if Z.ndim == 3 else Z), (T[:, :, 0] if T.ndim == 3 else T)
    beta = r2.params[[c for c in fourier(later.index[:1]).columns]].to_numpy()
    X = fourier(later.index).to_numpy()
    out = {}
    Th = np.eye(T.shape[0])
    for h in horizons:
        Th = Th @ T
        level = (Z @ Th @ a).ravel()
        reg = np.full(len(later), np.nan)
        reg[:-h] = X[h:] @ beta
        out[h] = pd.Series(level + reg, index=later.index)
    return out, res.aic


def sarimax_family(df, split, results, tuned):
    te = split["test"]
    present = (df.loc[te, "pir_sum30"] > 0).to_numpy()
    early = (df.loc[te, "co2"] <= CO2_GOOD).to_numpy()
    sensors = df["sensor_key"].unique()
    if os.environ.get("SMOKE"):
        sensors = sensors[:4]
    # order chosen by validation MAE on 6 sensors (fit before 25 Apr, forecast 25 Apr - 15 May)
    orders = [(1, 1, 1), (2, 1, 1), (2, 1, 2)]
    pick = rng.choice(sensors, min(6, len(sensors)), replace=False)
    val_idx = split["val"]
    choice = {}
    for var, target in [("co2", "y_co2_30"), ("tmp", "y_tmp_30")]:
        maes = []
        for order in orders:
            err = []
            for sk in pick:
                g = df[df.sensor_key == sk].set_index("time")[var]
                fc, _ = sarimax_forecasts(g, FIT_END, order)
                rows = val_idx[df.loc[val_idx, "sensor_key"] == sk]
                p = fc[6].reindex(df.loc[rows, "time"]).to_numpy()
                y = df.loc[rows, target].to_numpy()
                ok = ~np.isnan(p)
                err.append(np.abs(p[ok] - y[ok]))
            maes.append(float(np.mean(np.concatenate(err))))
        choice[var] = orders[int(np.argmin(maes))]
        tuned[f"sarimax_{var}_order"] = list(choice[var])
        print(f"  SARIMAX {var}: validation MAE by order {dict(zip(map(str, orders), np.round(maes, 3)))}", flush=True)

    t0 = time.time()
    preds = {"co2": {}, "tmp": {}, "co2_max": {}}
    for sk in sensors:
        g = df[df.sensor_key == sk].set_index("time")
        for var in ["co2", "tmp"]:
            try:
                fc, _ = sarimax_forecasts(g[var], TRAIN_END, choice[var])
            except Exception:
                continue
            preds[var][sk] = fc[6]
            if var == "co2":
                preds["co2_max"][sk] = pd.concat(fc.values(), axis=1).max(axis=1, skipna=False)
    secs = time.time() - t0

    def collect(key):
        out = np.full(len(te), np.nan)
        sk_arr, t_arr = df.loc[te, "sensor_key"].to_numpy(), df.loc[te, "time"].to_numpy()
        for sk, ser in preds[key].items():
            m = sk_arr == sk
            out[m] = ser.reindex(t_arr[m]).to_numpy()
        return out

    for task, (target, base), var in [("co2_30", REG_TARGETS["co2_30"], "co2"), ("tmp_30", REG_TARGETS["tmp_30"], "tmp")]:
        p = collect(var)
        b = df.loc[te, base].to_numpy()
        bad = np.isnan(p) | (np.abs(p - b) > (2000 if var == "co2" else 10))   # gaps or a diverged forecast
        p = np.where(bad, b, p)                                                   # fall back to the current value
        results[task]["sarimax"] = {"best_params": {"order": list(choice[var]), "seasonality": "Fourier daily (3 harmonics) + weekend",
                                                    "fitted_on": "last 28 days before test, one model per sensor"},
                                    "fit_and_forecast_seconds_all_sensors": round(secs, 1),
                                    "scores": reg_scores(df.loc[te, target].to_numpy(), p, b, present)}
        print(f"  {task} SARIMAX: test MAE {results[task]['sarimax']['scores']['all_teaching_hours']['MAE']}", flush=True)
    # alert from SARIMAX: forecast max over the next 30 min, threshold chosen on validation sensors' run
    s = collect("co2_max")
    s = np.where(np.isnan(s), df.loc[te, "co2"].to_numpy(), s)
    y_te = df.loc[te, ALERT].to_numpy().astype(int)
    thr = CO2_GOOD   # the forecast is in ppm, so the natural threshold is the limit itself
    results["alert30"]["sarimax"] = {"threshold_ppm": thr, "scores": clf_scores(y_te, s, thr, early)}
    print(f"  alert SARIMAX: test F1 {results['alert30']['sarimax']['scores']['all_teaching_hours']['F1']}", flush=True)


# ------------------------------------------------------------------ LSTM
SEQ_COLS = ["co2", "tmp", "rlf", "lux", "pir"]
CTX_COLS = ["hour_sin", "hour_cos", "dow", "weekend", "volume_m3", "co2_wk", "co2_wk_d30", "co2_wk_d60",
            "pir_wk_next60", "tmp_wk_d60"]
SEQ_LEN = 12   # last 60 minutes


def deep_family(df, split, results, tuned, kind="lstm", seed=SEED):
    """LSTM, TCN or Transformer. All three get identical inputs (last 60 min), outputs
    (CO2 change, temperature change, alert) and tuning; only the sequence encoder differs."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    drng = np.random.RandomState(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    fit_rows = split["fit"]
    mu = df.loc[fit_rows, SEQ_COLS + CTX_COLS].mean()
    sd = df.loc[fit_rows, SEQ_COLS + CTX_COLS].std().replace(0, 1)
    base = ((df[SEQ_COLS] - mu[SEQ_COLS]) / sd[SEQ_COLS]).to_numpy(np.float32)
    mask = np.isnan(base).astype(np.float32)
    base = np.nan_to_num(base)
    seq_all = np.concatenate([base, mask], axis=1)                       # 10 channels per step
    ctx_all = np.nan_to_num(((df[CTX_COLS] - mu[CTX_COLS]) / sd[CTX_COLS]).to_numpy(np.float32))
    SC_CO2, SC_TMP = 50.0, 0.1
    y1 = ((df["y_co2_30"] - df["co2"]) / SC_CO2).to_numpy(np.float32)
    y2 = ((df["y_tmp_30"] - df["tmp"]) / SC_TMP).to_numpy(np.float32)
    y3 = df[ALERT].to_numpy(np.float32)
    offs = np.arange(-SEQ_LEN + 1, 1)

    def batch(idx):
        idx = np.asarray(idx)
        return (torch.from_numpy(seq_all[idx[:, None] + offs]), torch.from_numpy(ctx_all[idx]),
                torch.from_numpy(y1[idx]), torch.from_numpy(y2[idx]), torch.from_numpy(y3[idx]))

    n_in = seq_all.shape[1]

    class CausalConv(nn.Module):
        def __init__(self, cin, cout, dilation):
            super().__init__()
            self.pad = 2 * dilation
            self.conv = nn.Conv1d(cin, cout, kernel_size=3, dilation=dilation)

        def forward(self, x):                        # x: batch, channels, time
            return self.conv(nn.functional.pad(x, (self.pad, 0)))

    class TCN(nn.Module):
        """Temporal Convolutional Network: causal dilated convolutions (dilation 1, 2, 4 -> sees 15 steps)."""
        def __init__(self, hidden):
            super().__init__()
            self.layers = nn.ModuleList([CausalConv(n_in if i == 0 else hidden, hidden, d) for i, d in enumerate([1, 2, 4])])
            self.skip = nn.Conv1d(n_in, hidden, 1)

        def forward(self, s):
            x = s.transpose(1, 2)
            h = x
            for layer in self.layers:
                h = torch.relu(layer(h))
            return (h + self.skip(x))[:, :, -1]     # output at the latest time step

    class TransformerEnc(nn.Module):
        """Transformer encoder: self-attention over the 12 time steps, learned position embedding."""
        def __init__(self, hidden):
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden))
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, s):
            return self.enc(self.inp(s) + self.pos)[:, -1]

    class LSTMEnc(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)

        def forward(self, s):
            _, (h, _) = self.lstm(s)
            return h[-1]

    ENCODERS = {"lstm": LSTMEnc, "tcn": TCN, "transformer": TransformerEnc}

    class Net(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.enc = ENCODERS[kind](hidden)
            self.head = nn.Sequential(nn.Linear(hidden + ctx_all.shape[1], 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, s, c):
            return self.head(torch.cat([self.enc(s), c], dim=1))

    pos_w = torch.tensor(float((1 - y3[fit_rows].mean()) / max(y3[fit_rows].mean(), 1e-3)))

    def train(rows, hidden, lr, epochs, val_rows=None):
        net = Net(hidden)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state, history = np.inf, None, []
        for ep in range(epochs):
            net.train()
            perm = drng.permutation(rows)
            for i in range(0, len(perm), 1024):
                s, c, a, b, z = batch(perm[i:i + 1024])
                o = net(s, c)
                loss = (o[:, 0] - a).abs().mean() + (o[:, 1] - b).abs().mean() + 0.5 * bce(o[:, 2], z)
                opt.zero_grad(); loss.backward(); opt.step()
            if val_rows is not None:
                p = predict(net, val_rows)
                v = mean_absolute_error(df.loc[val_rows, "y_co2_30"], p[0]) / 50 + \
                    mean_absolute_error(df.loc[val_rows, "y_tmp_30"], p[1]) / 0.1
                history.append(v)
                if v < best:
                    best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best, (int(np.argmin(history)) + 1 if history else epochs)

    def predict(net, rows):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(rows), 4096):
                s, c, *_ = batch(rows[i:i + 4096])
                outs.append(net(s, c).numpy())
        o = np.concatenate(outs)
        r = np.asarray(rows)
        return (df.loc[r, "co2"].to_numpy() + o[:, 0] * SC_CO2, df.loc[r, "tmp"].to_numpy() + o[:, 1] * SC_TMP,
                1 / (1 + np.exp(-o[:, 2])))

    space = [{"hidden": h, "lr": lr} for h in [32, 64] for lr in [1e-3, 3e-3]]
    # Deep models use the validation weeks for early stopping, so (unlike the tabular models) they are
    # not refit on fit+validation: the network from the best training round of the best setting is kept.
    # Retraining from scratch for a fixed number of rounds proved unstable from run to run.
    tried = []
    t0 = time.time()
    for p in space:
        net_p, v, ep = train(split["fit"].to_numpy(), p["hidden"], p["lr"], epochs=20, val_rows=split["val"].to_numpy())
        tried.append((v, ep, p, net_p))
        print(f"  {kind} {p}: validation score {v:.3f} best epoch {ep} ({time.time() - t0:.0f}s)", flush=True)
    v, ep, best, net = min(tried, key=lambda z: z[0])
    fit_s = (time.time() - t0) / len(space)
    pv = predict(net, split["val"].to_numpy())
    thr = best_thr(df.loc[split["val"], ALERT].to_numpy().astype(int), pv[2])
    te = split["test"].to_numpy()
    t2 = time.time(); p = predict(net, te); pred_s = time.time() - t2
    present = (df.loc[te, "pir_sum30"] > 0).to_numpy()
    early = (df.loc[te, "co2"] <= CO2_GOOD).to_numpy()
    info = {"best_params": {**best, "epochs": ep, "sequence": "last 60 min (12 steps)", "outputs": "CO2, temperature, alert (shared network)"},
            "configs_tried": len(space), "fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 2),
            "trained_on": "fit period only; best validation round kept (early stopping)"}
    for task, (target, basecol), pi in [("co2_30", REG_TARGETS["co2_30"], 0), ("tmp_30", REG_TARGETS["tmp_30"], 1)]:
        b = df.loc[te, basecol].to_numpy()
        results[task][kind] = {**info, "seed": seed, "scores": reg_scores(df.loc[te, target].to_numpy(), p[pi], b, present)}
    results["alert30"][kind] = {**info, "seed": seed, "threshold": round(thr, 4),
                                  "scores": clf_scores(df.loc[te, ALERT].to_numpy().astype(int), p[2], thr, early)}
    tuned[kind] = {**best, "epochs": ep, "alert_threshold": thr}
    print(f"  {kind} test: CO2 MAE {results['co2_30'][kind]['scores']['all_teaching_hours']['MAE']}, "
          f"temp MAE {results['tmp_30'][kind]['scores']['all_teaching_hours']['MAE']}, "
          f"alert F1 {results['alert30'][kind]['scores']['all_teaching_hours']['F1']}", flush=True)


DEEP_SEEDS = [42, 43, 44]


def deep_family_seeds(df, split, results, tuned, kind):
    """Deep networks vary from one training run to the next, so each is trained with 3 random
    starts. Reported scores are the average over the 3 runs; the spread is kept as *_std."""
    runs = []
    for sd in DEEP_SEEDS:
        r = {"co2_30": {}, "tmp_30": {}, "alert30": {}}
        t = {}
        deep_family(df, split, r, t, kind, seed=sd)
        runs.append((r, t))
    for task in ["co2_30", "tmp_30", "alert30"]:
        per = [r[task][kind] for r, _ in runs]
        mean, std = {}, {}
        for subset in per[0]["scores"]:
            keys = per[0]["scores"][subset].keys()
            mean[subset] = {k: round(float(np.mean([q["scores"][subset][k] for q in per])), 4) for k in keys}
            std[subset] = {k: round(float(np.std([q["scores"][subset][k] for q in per])), 4) for k in keys}
        results[task][kind] = {"seeds": DEEP_SEEDS, "scores": mean, "scores_std": std,
                               "fit_seconds": round(float(np.mean([q["fit_seconds"] for q in per])), 1),
                               "per_seed": [{"seed": q["seed"], "best_params": q["best_params"], "scores": q["scores"]} for q in per]}
    tuned[kind] = [t[kind] for _, t in runs]
    print(f"  {kind} average of {len(DEEP_SEEDS)} runs: CO2 MAE {results['co2_30'][kind]['scores']['all_teaching_hours']['MAE']} "
          f"(+/- {results['co2_30'][kind]['scores_std']['all_teaching_hours']['MAE']}), temp MAE "
          f"{results['tmp_30'][kind]['scores']['all_teaching_hours']['MAE']}, alert F1 "
          f"{results['alert30'][kind]['scores']['all_teaching_hours']['F1']}", flush=True)


# ------------------------------------------------------------------ SVR (on a sample: kernel SVR does not scale)
SVR_ROWS = 10000


def svr_family(df, split, results, tuned):
    """Support Vector Regression (RBF kernel). Training time grows roughly with the square of the
    number of rows, so it is trained on a random sample of 10,000 rows. Scores on the full test set."""
    from sklearn.svm import SVR
    te = split["test"]
    present = (df.loc[te, "pir_sum30"] > 0).to_numpy()
    r = np.random.RandomState(SEED)
    fit_s = r.choice(split["fit"], min(SVR_ROWS, len(split["fit"])), replace=False)
    train_s = r.choice(split["train"], min(SVR_ROWS, len(split["train"])), replace=False)
    X = lambda idx: df.loc[idx, FULL_FEATURES]
    space = [{"C": c, "epsilon": e} for c in [1, 10, 100] for e in [0.1, 1.0]]
    for task, (target, base) in REG_TARGETS.items():
        scale = 50.0 if base == "co2" else 0.1       # SVR works on standardised changes
        def make(p):
            return linear(SVR(kernel="rbf", gamma="scale", C=p["C"], epsilon=p["epsilon"]))
        val = []
        t0 = time.time()
        for p in space:
            m = make(p).fit(X(fit_s), (df.loc[fit_s, target] - df.loc[fit_s, base]) / scale)
            pv = df.loc[split["val"], base].to_numpy() + m.predict(X(split["val"])) * scale
            val.append(mean_absolute_error(df.loc[split["val"], target], pv))
        best = space[int(np.argmin(val))]
        t1 = time.time()
        m = make(best).fit(X(train_s), (df.loc[train_s, target] - df.loc[train_s, base]) / scale)
        fit_sec = time.time() - t1
        t2 = time.time()
        p_te = df.loc[te, base].to_numpy() + m.predict(X(te)) * scale
        pred_sec = time.time() - t2
        b = df.loc[te, base].to_numpy()
        results[task]["svr_10k_sample"] = {"best_params": best, "configs_tried": len(space), "training_rows": SVR_ROWS,
                                           "validation_MAE_best": round(float(min(val)), 4),
                                           "fit_seconds": round(fit_sec, 1), "predict_seconds": round(pred_sec, 1),
                                           "scores": reg_scores(df.loc[te, target].to_numpy(), p_te, b, present)}
        tuned[f"svr_{task}"] = best
        print(f"  {task} SVR: best {best} test MAE {results[task]['svr_10k_sample']['scores']['all_teaching_hours']['MAE']} "
              f"({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ feature importance
def importance(df, split):
    te = split["test"]
    sample = rng.choice(te, size=min(40000, len(te)), replace=False)
    X = df.loc[sample, FULL_FEATURES]
    rows = []
    for task, (target, base) in REG_TARGETS.items():
        # The model predicts the CHANGE and adds the current value back. Importance is measured on the
        # change model, otherwise shuffling "co2"/"tmp" would also break that final addition.
        m = FINAL[task].estimator
        change = df.loc[sample, target] - df.loc[sample, base]
        r = permutation_importance(m, X, change, scoring=lambda est, X_, y_: -mean_absolute_error(y_, est.predict(X_)),
                                   n_repeats=3, random_state=SEED, n_jobs=1)
        for f, v, s in zip(FULL_FEATURES, r.importances_mean, r.importances_std):
            rows.append({"model": f"hist_gradient_boosting_regressor_{task}", "feature": f,
                         "importance": round(float(v), 5), "std": round(float(s), 5),
                         "meaning": "increase in MAE when this feature is shuffled (ppm or C)"})
    m = FINAL["alert30"]
    r = permutation_importance(m, X, df.loc[sample, ALERT].astype(int), scoring="average_precision",
                               n_repeats=3, random_state=SEED, n_jobs=1)
    for f, v, s in zip(FULL_FEATURES, r.importances_mean, r.importances_std):
        rows.append({"model": "hist_gradient_boosting_classifier_alert30", "feature": f, "importance": round(float(v), 5),
                     "std": round(float(s), 5), "meaning": "drop in PR-AUC when this feature is shuffled"})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "feature_importance.csv"), index=False)
    print("  saved results/feature_importance.csv", flush=True)


FINAL = {}


def main():
    """python compare_models.py            -> everything
       python compare_models.py svr deep      -> only those parts (results merged into the existing files)
       parts: tabular, importance, sarimax, svr, lstm, tcn, transformer, deep (= all three deep models)"""
    import sys
    only = set(sys.argv[1:])
    run = lambda part: not only or part in only
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    print("Building features...", flush=True)
    df, split = load()
    if os.environ.get("SMOKE"):                       # quick code test on a small slice
        split = {k: v[::60] for k, v in split.items()}
    print({k: len(v) for k, v in split.items()}, flush=True)
    cpath, tpath = os.path.join(RES, "model_comparison.json"), os.path.join(RES, "tuned_params.json")
    results = json.load(open(cpath)) if only and os.path.exists(cpath) else {"co2_30": {}, "tmp_30": {}, "alert30": {}}
    tuned = json.load(open(tpath)) if only and os.path.exists(tpath) else {}
    t_prev = results.get("setup", {}).get("runtime_minutes", 0) if only else 0
    if run("tabular") or run("importance"):
        print("Tabular models (Ridge/Logistic, RandomForest, HistGradientBoosting)", flush=True)
        if run("tabular"):
            tabular_family(df, split, results, tuned)
        else:   # importance only: refit the tuned HistGradientBoosting models
            for task, (target, base) in REG_TARGETS.items():
                FINAL[task] = DeltaModel(HistGradientBoostingRegressor(random_state=SEED, **tuned[f"hist_gradient_boosting_regressor_{task}"]),
                                         base).fit(df.loc[split["train"], FULL_FEATURES], df.loc[split["train"], target])
            p = {k: v for k, v in tuned["hist_gradient_boosting_classifier_alert30"].items() if k != "threshold"}
            FINAL["alert30"] = HistGradientBoostingClassifier(random_state=SEED, **p).fit(
                df.loc[split["train"], FULL_FEATURES], df.loc[split["train"], ALERT])
    if run("importance"):
        print("Feature importance", flush=True)
        importance(df, split)
    if run("sarimax"):
        print("SARIMAX", flush=True)
        sarimax_family(df, split, results, tuned)
    if run("svr"):
        print("SVR (10,000-row sample)", flush=True)
        svr_family(df, split, results, tuned)
    for kind in ["lstm", "tcn", "transformer"]:
        if run(kind) or run("deep"):
            print(kind.upper(), flush=True)
            try:
                import torch  # noqa: F401
            except ImportError:
                print("  PyTorch is not installed (it needs Python 3.12/3.13); deep models skipped.", flush=True)
                break
            deep_family_seeds(df, split, results, tuned, kind)
    results["setup"] = {"fit_period": "18 Jan - 24 Apr 2022", "validation_period": "25 Apr - 15 May 2022",
                        "test_period": "17 May - 24 Sep 2022", "rows": {k: int(len(v)) for k, v in split.items()},
                        "note_true_sarima": "A seasonal SARIMA with a daily season (96 steps at 15-min resolution) did not finish "
                                            "fitting one sensor within 15 minutes, so the daily cycle is modelled with Fourier terms instead.",
                        "runtime_minutes": round(t_prev + (time.time() - t0) / 60, 1)}
    with open(os.path.join(RES, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(RES, "tuned_params.json"), "w") as f:
        json.dump(tuned, f, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(f"Saved results/model_comparison.json and results/tuned_params.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
