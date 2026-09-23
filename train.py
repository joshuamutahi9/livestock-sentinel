"""
Livestock Sentinel AI - model training and back-test (Sprint 1)

Target: will a NEW lab-confirmed FMD outbreak be declared in this sub-county
        in the next 2 weeks?
Split : train on years 1-2, test on year 3 (time-based, never random).
Model : LightGBM, compared against two simple baselines.
Output: risk score 0-100 per sub-county-week, band, top drivers in plain language.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

D = Path(__file__).parent / "data"
from features import build_features
aw, FEATURES = build_features(pd.read_csv(D / "area_week_raw.csv"))
g = aw.groupby("area_id")

valid = (aw.week >= 8) & (aw.week <= aw.week.max() - 2)
train = aw[valid & (aw.week < 104)]
test = aw[valid & (aw.week >= 104)]

# ---------- models ----------
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1)
model.fit(train[FEATURES], train.target)
p_lgb = model.predict_proba(test[FEATURES])[:, 1]

# baseline 1: seasonal/historical rate per area and month
hist = train.assign(m=train.week % 52 // 4.35).groupby(["area_id", "m"]).target.mean()
p_seas = test.assign(m=test.week % 52 // 4.35).set_index(["area_id", "m"]).index.map(hist).fillna(train.target.mean()).values
# baseline 2: logistic regression on the same features
sc = StandardScaler().fit(train[FEATURES])
lr = LogisticRegression(max_iter=2000).fit(sc.transform(train[FEATURES]), train.target)
p_lr = lr.predict_proba(sc.transform(test[FEATURES]))[:, 1]

# ---------- risk score (0-100) ----------
p_train = model.predict_proba(train[FEATURES])[:, 1]
p_mod, p_high = np.quantile(p_train, [0.75, 0.92])
def to_score(p):
    s = np.where(p < p_mod, p / p_mod * 40,
         np.where(p < p_high, 40 + (p - p_mod) / (p_high - p_mod) * 30,
                  70 + np.minimum(1, (p - p_high) / (1 - p_high) * 3) * 30))
    return np.clip(np.round(s), 0, 100)

# ---------- metrics ----------
def recall_top_k(p, df, k=5):
    hits = tot = 0
    for w, idx in df.groupby("week").groups.items():
        sub = df.loc[idx]; order = np.argsort(-p[df.index.get_indexer(idx)])[:k]
        hits += sub.target.values[order].sum(); tot += sub.target.sum()
    return hits / max(tot, 1)

test = test.copy()
test["p"] = p_lgb
test["score"] = to_score(p_lgb)

# lead time: for each confirmed outbreak in the test year, how many weeks before
# confirmation did the score first reach High (>=70) within the preceding 6 weeks?
leads, caught = [], 0
events = aw[(aw.confirmed_outbreak == 1) & (aw.week >= 112)]
for ev in events.itertuples():
    win = test[(test.area_id == ev.area_id) & (test.week >= ev.week - 6) & (test.week < ev.week)]
    hi = win[win.score >= 70]
    if len(hi):
        caught += 1; leads.append(ev.week - hi.week.min())
a_leads, a_caught = [], 0
for ev in events.itertuples():
    win = test[(test.area_id == ev.area_id) & (test.week >= ev.week - 6) & (test.week < ev.week) & (test.anomaly_flags.str.len() > 0)]
    if len(win):
        a_caught += 1; a_leads.append(ev.week - win.week.min())
metrics = {
    "test_period": "Year 3 (weeks 104-155), never seen in training",
    "n_outbreaks_test": int(len(events)),
    "roc_auc": {"lightgbm": roc_auc_score(test.target, p_lgb), "logistic": roc_auc_score(test.target, p_lr),
                "seasonal_baseline": roc_auc_score(test.target, p_seas)},
    "pr_auc": {"lightgbm": average_precision_score(test.target, p_lgb),
               "logistic": average_precision_score(test.target, p_lr),
               "seasonal_baseline": average_precision_score(test.target, p_seas),
               "random": float(test.target.mean())},
    "recall_top5_areas_per_week": {"lightgbm": recall_top_k(p_lgb, test), "logistic": recall_top_k(p_lr, test),
                                   "seasonal_baseline": recall_top_k(p_seas, test)},
    "outbreaks_with_high_alert_before_confirmation": f"{caught}/{len(events)}",
    "median_lead_time_weeks": float(np.median(leads)) if leads else None,
    "mean_lead_time_weeks": float(np.mean(leads)) if leads else None,
    "high_alerts_per_week": float((test.score >= 70).groupby(test.week).sum().mean()),
    "anomaly_flag_before_confirmation": f"{a_caught}/{len(events)}",
    "anomaly_median_lead_time_weeks": float(np.median(a_leads)) if a_leads else None,
    "anomaly_flags_per_week": float((test.anomaly_flags.str.len() > 0).groupby(test.week).sum().mean()),
}
metrics = json.loads(json.dumps(metrics, default=float))
(D / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# ---------- explanations (SHAP -> plain language) ----------
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(test[FEATURES])
sv = sv[1] if isinstance(sv, list) else sv

def describe(f, r):
    return {
        "farmer_reports": f"{int(r.farmer_reports)} farmer symptom reports this week",
        "farmer_reports_2w": f"{int(r.farmer_reports_2w)} farmer symptom reports in the last 2 weeks",
        "vet_reports": f"{int(r.vet_reports)} suspected cases reported by vets this week",
        "vet_reports_2w": f"{int(r.vet_reports_2w)} vet-reported suspected cases in the last 2 weeks",
        "report_anomaly_z": "symptom reports well above this area's normal level" if r.report_anomaly_z > 0 else "symptom reports at or below normal",
        "inbound_animals": f"{int(r.inbound_animals)} cattle recorded arriving this week",
        "inbound_change_pct": f"cattle arrivals {r.inbound_change_pct:+.0f}% compared with recent weeks",
        "inbound_external": f"{int(r.inbound_external)} cattle arrived from other sub-counties",
        "inbound_from_outbreak_areas_2w": f"{int(r.inbound_from_outbreak_areas_2w)} cattle arrived from areas with recent outbreaks",
        "neighbour_outbreaks_4w": f"{int(r.neighbour_outbreaks_4w)} confirmed outbreaks in nearby sub-counties (last 4 weeks)",
        "own_outbreak_8w": "outbreak declared here in the last 8 weeks" if r.own_outbreak_8w else None,
        "vacc_coverage": f"vaccination coverage about {r.vacc_coverage:.0%}",
        "weeks_since_campaign": f"{int(r.weeks_since_campaign)} weeks since last vaccination campaign" if r.weeks_since_campaign < 99 else "no recent vaccination campaign",
        "rain_8w_anomaly_pct": (f"rainfall {abs(r.rain_8w_anomaly_pct):.0f}% below normal over the last 8 weeks" if r.rain_8w_anomaly_pct < -10
                                else f"rainfall {r.rain_8w_anomaly_pct:.0f}% above normal over the last 8 weeks" if r.rain_8w_anomaly_pct > 10
                                else "rainfall close to normal over the last 8 weeks"),
        "et0_anomaly": "hotter, drier air than normal (high evaporation)" if r.et0_anomaly > 0 else "cooler, less evaporative conditions than normal",
        "density_index": "high cattle density" if r.density_index >= 1.1 else "moderate cattle density",
        "market_hub": "major livestock market in the area" if r.market_hub else "no major market",
        "border": "border area with cross-border movement" if r.border else "not a border area",
        "woy_sin": "time of year when outbreaks have been more common", "woy_cos": "time of year when outbreaks have been more common",
    }[f]

drivers_up, drivers_down = [], []
for i, r in enumerate(test.itertuples()):
    order = np.argsort(-sv[i])
    up = [describe(FEATURES[j], r) for j in order[:6] if sv[i, j] > 0.02]
    up = [u for u in dict.fromkeys(up) if u][:4]
    down = [x for x in (describe(FEATURES[j], r) for j in np.argsort(sv[i])[:3] if sv[i, j] < -0.05) if x][:2]
    drivers_up.append(" | ".join(up)); drivers_down.append(" | ".join(down))
test["drivers_up"] = drivers_up
test["drivers_down"] = drivers_down
test["band"] = np.where(test.score >= 70, "High", np.where(test.score >= 40, "Elevated", "Low"))
test["data_confidence"] = np.where(test.reporting_prob < 0.3, "Low data", "Adequate")

keep = ["area_id", "week", "date", "score", "band", "data_confidence", "drivers_up", "drivers_down",
        "farmer_reports", "vet_reports", "inbound_animals", "inbound_change_pct",
        "inbound_from_outbreak_areas_2w", "vacc_coverage", "rain_mm", "rain_8w_anomaly_pct", "anomaly_flags",
        "confirmed_outbreak", "target"]
test[keep].to_csv(D / "risk_scores.csv", index=False)
model.booster_.save_model(str(D / "model.txt"))
(D / "score_thresholds.json").write_text(json.dumps({"p_mod": float(p_mod), "p_high": float(p_high)}))
print("risk_scores.csv rows:", len(test))
