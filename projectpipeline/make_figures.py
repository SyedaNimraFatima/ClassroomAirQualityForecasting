"""Step 5: figures for the README (saved to results/figures/)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import anonymise_rooms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, RES = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
WARN, CRIT = "#fab219", "#d03b3b"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5cf",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "axes.axisbelow": True, "savefig.dpi": 150, "savefig.bbox": "tight"})


def thresholds(ax):
    ax.axhline(1000, color=WARN, lw=1.2, ls="--")
    ax.axhline(2000, color=CRIT, lw=1.2, ls="--")
    x = ax.get_xlim()[1]
    ax.text(x, 1000, " 1000 ppm (good-air limit)", va="center", fontsize=8, color=INK2)
    ax.text(x, 2000, " 2000 ppm (unacceptable)", va="center", fontsize=8, color=INK2)


def fig_example_day():
    d = anonymise_rooms(pd.read_pickle(os.path.join(OUT, "clean_2022.pkl")))
    d = d[(d.room == "Room 01") & (d.time >= "2022-03-17 07:00") & (d.time <= "2022-03-17 19:00")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (k, g), c in zip(d.groupby("sensor_key"), [BLUE, ORANGE]):
        ax.plot(g.time, g.co2, color=c, lw=2, label=k.replace("#1", " front sensor").replace("#2", " back sensor"))
    thresholds(ax)
    ax.set_title("Room 01 (311 m³), Thursday 17 March 2022: CO2 during a teaching day")
    ax.set_ylabel("CO2 (ppm)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(FIG, "01_example_day.png")); plt.close(fig)


def fig_honest_scores(m):
    old = m["linear_regression"]["predict_current_temperature_from_other_current_readings"]
    leak = m["leakage_check_co2_30"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    vals = [old["random_80_20_split_R2"], old["time_split_R2"]]
    ax.bar(["Random 80/20 split\n(original 2023 notebook)", "Train on past,\ntest on future"], vals,
           color=[GREY, BLUE], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.04 if v >= 0 else -0.1), f"R² = {v:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_title("Original 2023 method: linear regression,\nsame-time temperature from other readings")
    ax.set_ylim(-1.2, 0.6)
    ax = axes[1]
    vals = [100 * leak["random_split"]["skill_vs_persistence"], 100 * leak["time_split"]["skill_vs_persistence"]]
    ax.bar(["Random split", "Train on past,\ntest on future"], vals, color=[GREY, BLUE], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.0f}% better\nthan persistence", ha="center", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_title("HistGradientBoosting CO2 forecast, two ways of testing it\n(30 min ahead, teaching hours)")
    ax.set_ylabel("Error reduction vs 'no change' (%)")
    ax.set_ylim(0, 40)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "02_why_scores_were_misleading.png")); plt.close(fig)


def fig_forecast_week():
    p = anonymise_rooms(pd.read_pickle(os.path.join(OUT, "preds_y_co2_30.pkl")))
    p["week"] = p.time.dt.to_period("W")
    busiest = p[p.y_co2_30 > 1000].groupby(["sensor_key", "week"]).size().idxmax()
    w = p[(p.sensor_key == busiest[0]) & (p.week == busiest[1])].sort_values("time")
    w = w[w.time.dt.dayofweek < 5]
    days = sorted(w.time.dt.date.unique())[:3]
    fig, axes = plt.subplots(1, len(days), figsize=(11, 3.4), sharey=True)
    for ax, day in zip(np.atleast_1d(axes), days):
        x = w[w.time.dt.date == day]
        t_target = x.time + pd.Timedelta(minutes=30)
        ax.plot(t_target, x.y_co2_30, color=INK2, lw=2, label="Actual CO2")
        ax.plot(t_target, x.pred, color=BLUE, lw=2, label="Forecast made 30 min earlier")
        ax.axhline(1000, color=WARN, lw=1.2, ls="--")
        ax.set_title(pd.Timestamp(day).strftime("%a %d %b %Y"))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("CO2 (ppm)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"30-minute CO2 forecast on unseen test data: {busiest[0].split('#')[0]}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_forecast_examples.png")); plt.close(fig)


def fig_horizons(m):
    r = m["ridge_regression_and_hist_gradient_boosting_regressor"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, var, title in [(axes[0], "co2", "CO2"), (axes[1], "tmp", "Temperature")]:
        hs = [15, 30, 60]
        for model, c, lab in [("hist_gradient_boosting_regressor", BLUE, "HistGradientBoosting"), ("ridge_regression", ORANGE, "Ridge regression")]:
            v = [100 * r[f"y_{var}_{h}"][model]["people_present"]["skill_vs_persistence"] for h in hs]
            ax.plot(hs, v, marker="o", ms=7, lw=2, color=c, label=lab)
            ax.text(hs[-1] + 2, v[-1], lab, color=INK2, fontsize=8, va="center")
        ax.axhline(0, color=INK2, lw=1)
        ax.text(40, -0.3, "no-change baseline", fontsize=8, color=INK2, va="top", ha="center")
        ax.set_xticks(hs); ax.set_xlim(12, 78)
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_title(f"{title}: error reduction vs 'no change'\n(test period, people in the room)")
    axes[0].set_ylabel("%")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "04_skill_by_horizon.png")); plt.close(fig)


def fig_rooms():
    t = anonymise_rooms(pd.read_csv(os.path.join(RES, "room_recommendations.csv")))
    t = t.sort_values("pct_teaching_time_co2_gt_1000")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True)
    axes[0].barh(t.room, t.pct_teaching_time_co2_gt_1000, color=BLUE, height=0.7)
    axes[0].set_title("Teaching time with CO2 above 1000 ppm (%)")
    axes[1].barh(t.room, t.pct_teaching_time_tmp_gt_26, color=ORANGE, height=0.7)
    axes[1].set_title("Teaching time above 26 °C (%)")
    for ax, col in [(axes[0], "pct_teaching_time_co2_gt_1000"), (axes[1], "pct_teaching_time_tmp_gt_26")]:
        for y, v in zip(t.room, t[col]):
            ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=7, color=INK2)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Where the problems are: 23 rooms, Jan-Sep 2022, Mon-Fri 07:30-20:00", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "05_rooms_co2_and_heat.png")); plt.close(fig)


def fig_capacity():
    t = anonymise_rooms(pd.read_csv(os.path.join(RES, "room_recommendations.csv")))
    t = t[t.room != "Room 07"].sort_values("volume_m3")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.max_people_45min_under_1000, 0.38, color=GREY, label="Windows shut for 45 min")
    ax.bar(x + 0.2, t.max_people_if_aired_every_20min, 0.38, color=AQUA, label="Fully aired every 20 min")
    ax.set_xticks(x, [f"{r}\n{int(v)} m³" for r, v in zip(t.room, t.volume_m3)], fontsize=7, rotation=90)
    ax.set_ylabel("People")
    ax.set_title("Estimated largest group that keeps CO2 under 1000 ppm (lecture hall Room 07 not shown: ~170 / ~340)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "06_room_capacity.png")); plt.close(fig)


NICE = {"persistence_baseline": "Persistence (no change)", "ridge_regression": "Ridge regression",
        "logistic_regression": "Logistic regression", "random_forest_regressor": "Random Forest",
        "random_forest_classifier": "Random Forest", "hist_gradient_boosting_regressor": "HistGradientBoosting",
        "hist_gradient_boosting_classifier": "HistGradientBoosting", "sarimax": "SARIMAX (per sensor)",
        "lstm": "LSTM (avg of 3 runs)", "tcn": "TCN (avg of 3 runs)", "transformer": "Transformer (avg of 3 runs)",
        "svr_10k_sample": "SVR (10k-row sample)", "rule_current_co2_above_1000": "Rule: CO2 already > 1000",
        "rule_linear_trend_extrapolation": "Rule: extend last trend"}


def fig_model_comparison():
    path = os.path.join(RES, "model_comparison.json")
    if not os.path.exists(path):
        return
    c = json.load(open(path))
    panels = [("co2_30", "people_present", "MAE", "CO2 30 min ahead, people present\nMAE (ppm), lower is better"),
              ("tmp_30", "all_teaching_hours", "MAE", "Temperature 30 min ahead\nMAE (°C), lower is better"),
              ("alert30", "early_warning_rows_co2_now_le_1000", "F1", "Ventilation alert, early warnings\nF1, higher is better")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (task, subset, metric, title) in zip(axes, panels):
        items = [(NICE.get(k, k), v["scores"][subset][metric]) for k, v in c[task].items() if subset in v["scores"]]
        items.sort(key=lambda z: z[1], reverse=(metric == "F1"))
        names, vals = zip(*items)
        best = vals[0]
        colors = [BLUE if v == best else GREY for v in vals]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors, height=0.65)
        ax.set_yticks(y, names, fontsize=8)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.3f}" if metric != "MAE" or task == "tmp_30" else f" {v:.1f}", va="center", fontsize=7, color=INK2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.22)
    fig.suptitle("Model comparison on the same 2022 test period (17 May - 24 Sep), best in blue", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "07_model_comparison.png")); plt.close(fig)


FEATURE_LABELS = {
    "co2": "CO2 now", "tmp": "temperature now", "rlf": "humidity now", "lux": "light now", "pir": "motion now",
    "volume_m3": "room volume", "hour_sin": "time of day (sin)", "hour_cos": "time of day (cos)", "dow": "weekday",
    "weekend": "weekend", "co2_max60": "CO2 max last 60 min", "tmp_mean24h": "temperature mean last 24 h",
    "pir_sum30": "motion last 30 min", "pir_sum60": "motion last 60 min", "lux_mean30": "light last 30 min",
    "rlf_d30": "humidity change 30 min", "co2_excess_per_m3": "CO2 above outdoor per m³",
    "co2_wk": "CO2 same time last week", "co2_wk_d30": "last week: CO2 change next 30 min",
    "co2_wk_d60": "last week: CO2 change next 60 min", "pir_wk_next60": "last week: motion next 60 min",
    "tmp_wk_d60": "last week: temperature change next 60 min"}


def label(f):
    if f in FEATURE_LABELS:
        return FEATURE_LABELS[f]
    for var, nm in [("co2", "CO2"), ("tmp", "temperature")]:
        if f.startswith(f"{var}_lag"):
            return f"{nm} {f.split('lag')[1]} min ago"
        if f.startswith(f"{var}_d"):
            return f"{nm} change last {f.split('_d')[1]} min"
    return f


def fig_importance():
    path = os.path.join(RES, "feature_importance.csv")
    if not os.path.exists(path):
        return
    t = pd.read_csv(path)
    panels = [("hist_gradient_boosting_regressor_co2_30", "CO2 forecast\n(increase in MAE, ppm)"),
              ("hist_gradient_boosting_regressor_tmp_30", "Temperature forecast\n(increase in MAE, °C)"),
              ("hist_gradient_boosting_classifier_alert30", "Ventilation alert\n(drop in PR-AUC)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (model, title) in zip(axes, panels):
        d = t[t.model == model].sort_values("importance", ascending=False).head(10)[::-1]
        ax.barh([label(f) for f in d.feature], d.importance, xerr=d["std"], color=BLUE, height=0.65,
                error_kw={"ecolor": INK2, "lw": 1})
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("What the HistGradientBoosting models rely on (top 10 features, permutation importance on test data)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "08_feature_importance.png")); plt.close(fig)


def fig_alert(m):
    a = m["logistic_regression_and_hist_gradient_boosting_classifier"]
    names = [("rule_current_co2_above_1000", "Rule: CO2 already > 1000"), ("rule_linear_trend_extrapolation", "Rule: extend last trend"),
             ("logistic_regression", "Logistic regression"), ("hist_gradient_boosting_classifier", "HistGradientBoosting")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, subset, title in [(axes[0], "all_teaching_hours", "All teaching-hour moments"),
                              (axes[1], "early_warning_rows_co2_now_le_1000", "Early warnings only (CO2 still ≤ 1000 ppm now)")]:
        y = np.arange(len(names))
        for k, (metric, col) in enumerate([("precision", GREY), ("recall", AQUA), ("F1", BLUE)]):
            vals = [a[n][subset][metric] for n, _ in names]
            ax.barh(y + (k - 1) * 0.27, vals, height=0.25, color=col, label=metric.capitalize() if metric != "F1" else "F1")
            for yi, v in zip(y + (k - 1) * 0.27, vals):
                ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y, [lab for _, lab in names], fontsize=8)
        ax.set_xlim(0, 1.12); ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Ventilation alert: will CO2 pass 1000 ppm in the next 30 minutes? (test period)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "09_alert_results.png")); plt.close(fig)


def fig_generalise(m):
    r = m["ridge_regression_and_hist_gradient_boosting_regressor"]
    u = m["unseen_rooms_5fold"]; ny = m["next_year_2023"]
    a = m["logistic_regression_and_hist_gradient_boosting_classifier"]["hist_gradient_boosting_classifier"]
    tests = ["Later months\n(May–Sep 2022)", "Unseen rooms\n(5 folds)", "Next year\n(2023)"]
    co2 = [100 * r["y_co2_30"]["hist_gradient_boosting_regressor"]["all_teaching_hours"]["skill_vs_persistence"],
           100 * u["co2_30_hist_gradient_boosting_regressor"]["skill_vs_persistence"]["mean"],
           100 * ny["co2_30"]["hist_gradient_boosting_regressor"]["skill_vs_persistence"]]
    co2_err = [0, 100 * u["co2_30_hist_gradient_boosting_regressor"]["skill_vs_persistence"]["std"], 0]
    f1 = [a["all_teaching_hours"]["F1"], u["alert30_hist_gradient_boosting_classifier"]["F1"]["mean"],
          ny["alert30"]["hist_gradient_boosting_classifier"]["F1"]]
    f1_err = [0, u["alert30_hist_gradient_boosting_classifier"]["F1"]["std"], 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(3)
    axes[0].bar(x, co2, yerr=co2_err, color=BLUE, width=0.55, error_kw={"ecolor": INK2})
    for xi, v, e in zip(x, co2, co2_err):
        axes[0].text(xi, v + e + 0.8, f"+{v:.0f}%", ha="center", fontsize=9)
    axes[0].set_xticks(x, tests, fontsize=8); axes[0].set_ylim(0, 22)
    axes[0].set_title("CO2 30 min ahead: error reduction\nvs 'no change' (all teaching hours)", fontsize=10)
    axes[0].set_ylabel("%"); axes[0].grid(axis="x", visible=False)
    axes[1].bar(x, f1, yerr=f1_err, color=AQUA, width=0.55, error_kw={"ecolor": INK2})
    for xi, v, e in zip(x, f1, f1_err):
        axes[1].text(xi, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(x, tests, fontsize=8); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Ventilation alert F1", fontsize=10); axes[1].grid(axis="x", visible=False)
    fig.suptitle("Does the chosen model (HistGradientBoosting) still work on new data?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "10_generalisation.png")); plt.close(fig)


def fig_coverage():
    rows = []
    for year in ["2022", "2023"]:
        path = os.path.join(OUT, f"clean_{year}.pkl")
        if not os.path.exists(path):
            return
        d = anonymise_rooms(pd.read_pickle(path))[["time", "sensor_key", "co2"]].dropna()
        d["day"] = d["time"].dt.floor("D")
        rows.append(d.groupby(["sensor_key", "day"]).size().rename("n").reset_index().assign(year=year))
    c = pd.concat(rows)
    sensors = sorted(c.sensor_key.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    for ax, year, col in [(axes[0], "2022", BLUE), (axes[1], "2023", ORANGE)]:
        d = c[c.year == year]
        ypos = d.sensor_key.map({s: i for i, s in enumerate(sensors)})
        ax.scatter(d.day, ypos, s=4, c=col, alpha=np.clip(d.n / 288, 0.15, 1), marker="s", linewidths=0)
        ax.set_yticks(range(len(sensors)), sensors, fontsize=7)
        ax.set_title(f"{year}: days with CO2 data per sensor (darker = more complete day)", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Data coverage: 2022 is complete; 2023 is thinner and has gaps", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "11_data_coverage.png")); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    m = json.load(open(os.path.join(RES, "metrics.json")))
    fig_example_day(); fig_honest_scores(m); fig_forecast_week(); fig_horizons(m); fig_rooms(); fig_capacity()
    fig_model_comparison(); fig_importance(); fig_alert(m); fig_generalise(m); fig_coverage()
    print("Figures saved to", FIG)


if __name__ == "__main__":
    main()
