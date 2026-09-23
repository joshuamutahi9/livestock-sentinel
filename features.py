"""Feature engineering and anomaly detection, shared by train.py and impact.py."""
import numpy as np
import pandas as pd


def build_features(aw):
    aw = aw.sort_values(["area_id", "week"]).reset_index(drop=True)
    g = aw.groupby("area_id")

    # ---------- features (observable signals only) ----------
    def roll(col, w):
        return g[col].transform(lambda s: s.rolling(w, min_periods=1).sum())

    aw["farmer_reports_2w"] = roll("farmer_reports", 2)
    aw["vet_reports_2w"] = roll("vet_reports", 2)
    base_rep = g["farmer_reports"].transform(lambda s: s.shift(1).rolling(12, min_periods=4).mean())
    sd_rep = g["farmer_reports"].transform(lambda s: s.shift(1).rolling(12, min_periods=4).std())
    aw["report_anomaly_z"] = ((aw.farmer_reports - base_rep) / (sd_rep.fillna(1) + 0.5)).fillna(0)
    base_in = g["inbound_animals"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    aw["inbound_change_pct"] = ((aw.inbound_animals - base_in) / (base_in + 5) * 100).fillna(0)
    aw["inbound_from_outbreak_areas_2w"] = roll("inbound_from_outbreak_areas", 2)
    aw["own_outbreak_8w"] = g["confirmed_outbreak"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).sum()).fillna(0)
    aw["woy_sin"] = np.sin(2 * np.pi * (aw.week % 52) / 52)
    aw["woy_cos"] = np.cos(2 * np.pi * (aw.week % 52) / 52)

    # ---------- anomaly detection (separate early tripwire, not the ML model) ----------
    def ewma_z(col, span=8):
        mu = g[col].transform(lambda x: x.shift(1).ewm(span=span, min_periods=4).mean())
        sd = g[col].transform(lambda x: x.shift(1).ewm(span=span, min_periods=4).std())
        return ((aw[col] - mu) / (sd.fillna(0) + 1)).fillna(0)
    aw["z_farmer"] = ewma_z("farmer_reports")
    aw["z_vet"] = ewma_z("vet_reports")
    aw["z_inbound"] = ewma_z("inbound_animals")
    flags = []
    for r in aw[["z_farmer", "z_vet", "z_inbound", "farmer_reports", "vet_reports", "inbound_change_pct"]].itertuples(index=False):
        f = []
        if r.z_farmer > 2 and r.farmer_reports >= 2: f.append(f"Farmer symptom reports unusually high ({int(r.farmer_reports)} this week)")
        if r.z_vet > 2 and r.vet_reports >= 1: f.append(f"Vet-reported suspected cases unusually high ({int(r.vet_reports)} this week)")
        if r.z_inbound > 2.5 and r.inbound_change_pct > 40: f.append(f"Cattle arrivals unusually high ({r.inbound_change_pct:+.0f}% on recent weeks)")
        flags.append(" | ".join(f))
    aw["anomaly_flags"] = flags

    # target: new confirmed outbreak in weeks t+1..t+2
    aw["target"] = (g["confirmed_outbreak"].shift(-1).fillna(0) + g["confirmed_outbreak"].shift(-2).fillna(0) > 0).astype(int)

    FEATURES = ["farmer_reports", "farmer_reports_2w", "vet_reports", "vet_reports_2w", "report_anomaly_z",
                "inbound_animals", "inbound_change_pct", "inbound_external", "inbound_from_outbreak_areas_2w",
                "neighbour_outbreaks_4w", "own_outbreak_8w", "vacc_coverage", "weeks_since_campaign",
                "rain_8w_anomaly_pct", "et0_anomaly", "density_index", "market_hub", "border", "woy_sin", "woy_cos"]
    return aw, FEATURES
