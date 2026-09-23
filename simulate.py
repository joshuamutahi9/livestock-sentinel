"""
Livestock Sentinel AI - data simulator (Sprint 1, walking skeleton)

ALL DATA PRODUCED HERE IS SIMULATED. It is not government surveillance data.

How it works
- ~29 real sub-counties in 6 Kenyan counties, each with cattle holdings (farms, markets).
- FMD spreads between herds: locally within a sub-county, and through animal movements.
- The hidden "true" infection state is NEVER given to the AI model.
- The AI only sees what would be observable in real life: recorded movements,
  farmer/vet symptom reports (with under-reporting and false reports),
  vaccination records, rainfall, and lab-confirmed outbreaks (with delays).

Sprint 2 will replace the synthetic rainfall with real CHIRPS data and
calibrate cattle densities with KNBS 2019 census figures.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

N_WEEKS = 156                      # 3 years of weekly data
START = pd.Timestamp("2023-01-02")

# county, sub_county, lat, lon, zone, reporting_prob, density_index, market_hub, border
AREAS = [
    ("Narok", "Narok North", -1.08, 35.87, "agro-pastoral", 0.45, 1.1, 1, 0),
    ("Narok", "Narok South", -1.43, 35.65, "pastoral", 0.30, 1.2, 0, 0),
    ("Narok", "Narok East", -1.20, 36.20, "pastoral", 0.35, 1.0, 1, 0),
    ("Narok", "Narok West", -1.50, 35.20, "pastoral", 0.25, 1.1, 0, 1),
    ("Narok", "Transmara West", -1.00, 34.88, "mixed", 0.50, 1.3, 0, 1),
    ("Narok", "Transmara East", -1.20, 35.05, "mixed", 0.45, 1.2, 0, 0),
    ("Kajiado", "Kajiado Central", -1.85, 36.78, "pastoral", 0.40, 0.9, 1, 0),
    ("Kajiado", "Kajiado North", -1.36, 36.66, "peri-urban", 0.60, 0.8, 1, 0),
    ("Kajiado", "Kajiado East", -1.55, 36.95, "pastoral", 0.45, 0.8, 0, 0),
    ("Kajiado", "Kajiado West", -1.90, 36.30, "pastoral", 0.25, 0.7, 0, 0),
    ("Kajiado", "Loitokitok", -2.93, 37.51, "pastoral", 0.30, 0.9, 1, 1),
    ("Nakuru", "Naivasha", -0.72, 36.43, "mixed", 0.60, 1.0, 1, 0),
    ("Nakuru", "Gilgil", -0.50, 36.32, "mixed", 0.60, 1.0, 0, 0),
    ("Nakuru", "Molo", -0.25, 35.73, "mixed", 0.65, 1.2, 0, 0),
    ("Nakuru", "Njoro", -0.33, 35.94, "mixed", 0.65, 1.1, 0, 0),
    ("Nakuru", "Rongai", -0.17, 35.86, "mixed", 0.60, 1.0, 1, 0),
    ("Laikipia", "Laikipia East", 0.01, 37.07, "mixed", 0.55, 0.9, 1, 0),
    ("Laikipia", "Laikipia West", 0.27, 36.54, "agro-pastoral", 0.45, 1.0, 0, 0),
    ("Laikipia", "Laikipia North", 0.40, 37.15, "pastoral", 0.25, 0.8, 0, 0),
    ("Isiolo", "Isiolo", 0.35, 37.58, "pastoral", 0.40, 0.8, 1, 0),
    ("Isiolo", "Garbatulla", 0.53, 38.52, "pastoral", 0.20, 0.7, 0, 0),
    ("Isiolo", "Merti", 1.07, 38.67, "pastoral", 0.15, 0.6, 0, 0),
    ("Garissa", "Garissa Township", -0.45, 39.65, "pastoral", 0.45, 0.9, 1, 0),
    ("Garissa", "Dadaab", 0.06, 40.31, "pastoral", 0.20, 0.8, 0, 1),
    ("Garissa", "Fafi", -0.90, 40.10, "pastoral", 0.15, 0.7, 0, 1),
    ("Garissa", "Ijara", -1.60, 40.52, "pastoral", 0.20, 0.7, 0, 1),
    ("Garissa", "Lagdera", 0.74, 39.18, "pastoral", 0.15, 0.6, 0, 0),
    ("Garissa", "Balambala", -0.04, 39.33, "pastoral", 0.20, 0.7, 0, 0),
]
areas = pd.DataFrame(AREAS, columns=["county", "sub_county", "lat", "lon", "zone",
                                     "reporting_prob", "density_index", "market_hub", "border"])
areas["area_id"] = [f"A{i:02d}" for i in range(len(areas))]
A = len(areas)

# distance matrix (km, approx)
lat = np.radians(areas.lat.values); lon = np.radians(areas.lon.values)
dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
DIST = 6371 * 2 * np.arcsin(np.sqrt(h))

# movement destination weights between areas (gravity model; markets attract)
attract = 1 + 2.5 * areas.market_hub.values
W = attract[None, :] / (1 + DIST / 60) ** 2
np.fill_diagonal(W, 0)
W = W / W.sum(axis=1, keepdims=True)

# ---------- holdings ----------
hold_rows = []
for a in range(A):
    n = int(RNG.integers(35, 55) * areas.density_index[a])
    for j in range(n):
        htype = "market" if (areas.market_hub[a] and j == 0) else "farm"
        pastoral = areas.zone[a] == "pastoral"
        size = int(RNG.lognormal(4.0 if pastoral else 3.0, 0.6)) + 3
        hold_rows.append((f"H{len(hold_rows):05d}", areas.area_id[a], a, htype, size))
holdings = pd.DataFrame(hold_rows, columns=["holding_id", "area_id", "a", "holding_type", "herd_size"])
H = len(holdings)
h_area = holdings.a.values

# ---------- environment (synthetic placeholder; real CHIRPS in Sprint 2) ----------
weeks = np.arange(N_WEEKS)
dates = START + pd.to_timedelta(weeks * 7, unit="D")
woy = dates.isocalendar().week.values.astype(int)
aridity = np.array([0.5 if z == "pastoral" and c in ("Garissa", "Isiolo") else 1.0
                    for z, c in zip(areas.zone, areas.county)])
seasonal = (np.exp(-((woy - 16) ** 2) / 30) * 60 + np.exp(-((woy - 45) ** 2) / 25) * 45 + 5)
year_factor = np.where(dates.year == 2023, 1.25, np.where(dates.year == 2024, 1.0, 0.65))  # 2025 dry year
rain = np.maximum(0, seasonal[None, :] * year_factor[None, :] * aridity[:, None]
                  * RNG.lognormal(0, 0.35, (A, N_WEEKS)))
rain_clim = (seasonal[None, :] * aridity[:, None])
dry_factor = 1 + 0.8 * np.clip(1 - rain / (rain_clim + 1), 0, 1.5)  # drier -> more mixing

# ---------- vaccination ----------
county_list = areas.county.unique()
vacc = np.zeros((A, N_WEEKS))
base_cov = RNG.uniform(0.15, 0.45, A)
campaign_weeks = {c: sorted(RNG.choice(np.arange(5, N_WEEKS), size=3, replace=False)) for c in county_list}
last_campaign = np.full((A, N_WEEKS), 99.0)
for a in range(A):
    cov = base_cov[a]; since = 99
    for t in range(N_WEEKS):
        if t in campaign_weeks[areas.county[a]]:
            cov = min(0.85, cov + RNG.uniform(0.25, 0.4)); since = 0
        else:
            cov = max(base_cov[a] * 0.7, cov * 0.985); since += 1
        vacc[a, t] = cov; last_campaign[a, t] = since

# ---------- epidemic simulation ----------
S, I, R = 0, 1, 2
state = np.zeros(H, dtype=int)
timer = np.zeros(H, dtype=int)
herd_vacc = RNG.random(H) < base_cov[h_area]

inf_count = np.zeros((A, N_WEEKS))
moves_log = []
farmer_rep = np.zeros((A, N_WEEKS)); vet_rep = np.zeros((A, N_WEEKS))
detected_week = {}   # area -> week truth first detected in current wave
confirm_events = []  # (area, week_confirmed)
active_until = np.full(A, -1)

BETA_LOCAL = 1.3
for t in range(N_WEEKS):
    # re-sample herd vaccination to follow coverage
    herd_vacc = RNG.random(H) < vacc[h_area, t]
    susc = np.where(herd_vacc, 0.15, 1.0)

    # introductions (rare; more at borders)
    intro_p = 0.004 + 0.01 * areas.border.values
    for a in np.where(RNG.random(A) < intro_p)[0]:
        cand = np.where((h_area == a) & (state == S))[0]
        if len(cand):
            k = RNG.choice(cand); state[k] = I; timer[k] = RNG.integers(2, 5)

    infected = state == I
    # movements
    p_move = 0.08 * dry_factor[h_area, t] * np.where(holdings.holding_type.values == "market", 4, 1)
    movers = np.where(RNG.random(H) < p_move)[0]
    new_inf = np.zeros(H, dtype=bool)
    for k in movers:
        a = h_area[k]
        dest_area = a if RNG.random() < 0.6 else RNG.choice(A, p=W[a])
        cand = np.where(h_area == dest_area)[0]
        d = RNG.choice(cand)
        n_anim = int(RNG.integers(2, 25))
        recorded = RNG.random() < (0.85 if holdings.holding_type.values[k] == "market" else 0.6)
        moves_log.append((t, holdings.holding_id.values[k], holdings.holding_id.values[d],
                          a, dest_area, n_anim, int(recorded)))
        if infected[k] and state[d] == S and RNG.random() < 0.55 * susc[d]:
            new_inf[d] = True

    # local transmission within area (density- and dryness-dependent)
    n_area = np.bincount(h_area, minlength=A)
    i_area = np.bincount(h_area, weights=infected, minlength=A)
    force = BETA_LOCAL * i_area / n_area * areas.density_index.values * dry_factor[:, t]
    # spillover from nearby areas
    near = (DIST < 80) & (DIST > 0)
    force += 0.15 * (near @ (i_area / n_area))
    p_inf = 1 - np.exp(-force[h_area] * susc)
    new_inf |= (state == S) & (RNG.random(H) < p_inf)

    # update states
    timer[infected] -= 1
    recover = infected & (timer <= 0)
    state[recover] = R; timer[recover] = RNG.integers(20, 40, recover.sum())
    wane = (state == R) & (~recover)
    timer[wane] -= 1
    state[wane & (timer <= 0)] = S
    state[new_inf] = I; timer[new_inf] = RNG.integers(2, 5, new_inf.sum())

    infected = state == I
    inf_count[:, t] = np.bincount(h_area, weights=infected, minlength=A)

    # observable reports (under-reporting + false alarms)
    for a in range(A):
        n_i = int(inf_count[a, t])
        rp = areas.reporting_prob[a]
        farmer_rep[a, t] = RNG.binomial(n_i, rp) + RNG.poisson(0.25)
        vet_rep[a, t] = RNG.binomial(n_i, rp * 0.4) + RNG.poisson(0.03)
        # official detection -> lab confirmation with delay
        if n_i > 0 and t > active_until[a] and a not in detected_week:
            if RNG.random() < 1 - (1 - rp * 0.5) ** n_i:
                detected_week[a] = t
        if a in detected_week:
            dt = detected_week.pop(a)
            conf = dt + int(RNG.integers(1, 4))
            if conf < N_WEEKS:
                confirm_events.append((a, conf))
            active_until[a] = conf + 8   # no "new" outbreak declared within 8 weeks

# ---------- assemble observable area-week table ----------
confirmed = np.zeros((A, N_WEEKS))
for a, t in confirm_events:
    confirmed[a, t] = 1

mv = pd.DataFrame(moves_log, columns=["week", "from_holding", "to_holding",
                                      "from_area", "to_area", "n_animals", "permit_recorded"])
rec = mv[mv.permit_recorded == 1]
inbound = np.zeros((A, N_WEEKS)); inbound_ext = np.zeros((A, N_WEEKS)); inbound_risky = np.zeros((A, N_WEEKS))
recent_conf = np.zeros((A, N_WEEKS))
for t in range(N_WEEKS):
    recent_conf[:, t] = confirmed[:, max(0, t - 4):t + 1].sum(axis=1) > 0
for r in rec.itertuples():
    inbound[r.to_area, r.week] += r.n_animals
    if r.from_area != r.to_area:
        inbound_ext[r.to_area, r.week] += r.n_animals
        if recent_conf[r.from_area, r.week]:
            inbound_risky[r.to_area, r.week] += r.n_animals

rows = []
for a in range(A):
    for t in range(N_WEEKS):
        rows.append(dict(
            area_id=areas.area_id[a], week=t, date=dates[t].date(),
            rain_mm=rain[a, t], rain_anomaly=rain[a, t] - rain_clim[a, t],
            vacc_coverage=vacc[a, t], weeks_since_campaign=min(last_campaign[a, t], 99),
            inbound_animals=inbound[a, t], inbound_external=inbound_ext[a, t],
            inbound_from_outbreak_areas=inbound_risky[a, t],
            farmer_reports=farmer_rep[a, t], vet_reports=vet_rep[a, t],
            confirmed_outbreak=confirmed[a, t],
            true_infected_herds=inf_count[a, t],   # hidden truth: NOT used by the model
        ))
aw = pd.DataFrame(rows)
aw = aw.merge(areas[["area_id", "density_index", "market_hub", "border", "reporting_prob", "lat", "lon"]], on="area_id")

# neighbouring confirmed outbreaks (<150 km) in last 4 weeks
near150 = (DIST < 150) & (DIST > 0)
nb = np.zeros((A, N_WEEKS))
for t in range(N_WEEKS):
    nb[:, t] = near150 @ confirmed[:, max(0, t - 3):t + 1].sum(axis=1)
aw["neighbour_outbreaks_4w"] = nb[aw.area_id.str[1:].astype(int), aw.week]

# ---------- sample animal-level records (~10,000) ----------
animal_rows = []
per_hold = np.maximum(1, (10000 * holdings.herd_size / holdings.herd_size.sum()).round().astype(int))
for hid, a_id, n in zip(holdings.holding_id, holdings.area_id, per_hold):
    for _ in range(n):
        animal_rows.append((f"KE-{len(animal_rows):06d}", "cattle",
                            RNG.choice(["Boran", "Zebu", "Sahiwal", "Friesian cross", "Ayrshire cross"]),
                            RNG.choice(["F", "M"], p=[0.65, 0.35]), int(RNG.integers(6, 120)),
                            hid, a_id, int(RNG.random() < 0.35)))
animals = pd.DataFrame(animal_rows, columns=["animal_id", "species", "breed", "sex", "age_months",
                                             "holding_id", "area_id", "vaccinated_fmd"])

areas.to_csv(OUT / "areas.csv", index=False)
holdings.drop(columns="a").to_csv(OUT / "holdings.csv", index=False)
animals.to_csv(OUT / "animals.csv", index=False)
mv.assign(date=(START + pd.to_timedelta(mv.week * 7, unit="D")).dt.date).to_csv(OUT / "movements.csv", index=False)
aw.to_csv(OUT / "area_week_raw.csv", index=False)
print(f"areas={A} holdings={H} animals={len(animals)} movements={len(mv)} "
      f"confirmed_outbreaks={int(confirmed.sum())}")
