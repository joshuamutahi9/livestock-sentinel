"""
Livestock Sentinel AI - counterfactual impact simulation

For each simulated world (seed), the same outbreak season is run twice:
  A. Status quo : authorities respond only after lab confirmation.
  B. Sentinel   : same response to confirmations, plus early targeted action
                  (verification, movement checks, targeted vaccination) when
                  the model raises a High alert.
Alerts for B come from the trained model scoring world A's observable data.
This is conservative: new outbreak chains that appear only in B get the
status-quo response. Results are MODELLED ESTIMATES, not observed impact.
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import build_features

HERE = Path(__file__).parent
D = HERE / "data"
N_SEEDS = int(os.environ.get("N_SEEDS", 12))
booster = lgb.Booster(model_file=str(D / "model.txt"))
p_high = json.loads((D / "score_thresholds.json").read_text())["p_high"]


def run(seed, policy, out, alerts=None):
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SEED": str(seed), "POLICY": policy, "OUT": str(out)}
    if alerts:
        env["ALERTS"] = str(alerts)
    subprocess.run([sys.executable, str(HERE / "simulate.py")], env=env, check=True, capture_output=True)
    return json.loads((out / "impact_summary.json").read_text())


rows = []
for seed in range(100, 100 + N_SEEDS):
    base = HERE / "tmp_impact" / str(seed)
    sq = run(seed, "status_quo", base / "sq")
    aw, FEATURES = build_features(pd.read_csv(base / "sq" / "area_week_raw.csv"))
    aw["p"] = booster.predict(aw[FEATURES])
    alerts = aw[(aw.week >= 104) & (aw.p >= p_high)][["area_id", "week"]]
    alerts.to_csv(base / "alerts.csv", index=False)
    se = run(seed, "sentinel", base / "se", base / "alerts.csv")
    rows += [sq, se]
    print(seed, "status quo:", sq["herds_infected"], "herds | sentinel:", se["herds_infected"], "herds")

df = pd.DataFrame(rows)
cols = ["herds_infected", "animals_infected", "infected_herd_weeks", "confirmed_outbreaks",
        "market_closure_weeks", "quarantine_weeks", "early_action_weeks"]
mean = df.groupby("policy")[cols].mean()
paired = df.pivot(index="seed", columns="policy", values=cols)
summary = {"n_worlds": N_SEEDS, "period": "one simulated year (year 3), 26 sub-counties", "metrics": {}}
for c in cols:
    sqv, sev = paired[(c, "status_quo")], paired[(c, "sentinel")]
    red = (sqv - sev) / sqv.replace(0, np.nan) * 100
    summary["metrics"][c] = {"status_quo": float(sqv.mean()), "sentinel": float(sev.mean()),
                             "reduction_pct_mean": float(red.mean()),
                             "reduction_pct_p10": float(red.quantile(0.1)),
                             "reduction_pct_p90": float(red.quantile(0.9)),
                             "worlds_improved": int((sev < sqv).sum())}
(D / "impact_results.json").write_text(json.dumps(summary, indent=2))
df.to_csv(D / "impact_runs.csv", index=False)
print(json.dumps(summary, indent=2))
