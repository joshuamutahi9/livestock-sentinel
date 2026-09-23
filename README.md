# Livestock Sentinel AI

AI-enabled early warning for foot-and-mouth disease (FMD) in cattle in Kenya.
It combines animal movements, symptom reports, vaccination and environmental data
to predict which sub-counties are likely to see a new outbreak in the next 2 weeks,
and explains why.

> **Prototype notice:** all animal, movement, report and outbreak data in this
> repository is **simulated**. It is not government surveillance data.

## What is in this repository

| File | Purpose |
|---|---|
| `simulate.py` | Simulates FMD spreading between cattle herds through local contact and animal movement across 26 real Kenyan sub-counties. Produces only the signals that would be observable in reality. |
| `train.py` | Builds features, trains the model (LightGBM), compares it with simple baselines on a held-out year, and creates risk scores with plain-language explanations (SHAP). |
| `features.py` | Feature engineering and the anomaly-detection layer, shared by training and impact runs. |
| `impact.py` | Runs the same simulated year twice in 20 simulated worlds (act after confirmation vs act on alerts) to estimate impact. |
| `sms.py` | Understands farmer SMS reports in Swahili/English (Claude language model, with a keyword fallback), decides what needs human review, and writes the reply. |
| `app.py` | The dashboard. |
| `colab/sprint2_real_data.ipynb` | Downloads the real boundary and weather data. |
| `data/` | Generated data and model outputs. |

## Run the dashboard

```
pip install -r requirements.txt
streamlit run app.py
```

## Regenerate data and retrain

```
pip install -r requirements-dev.txt
python simulate.py
python train.py
python impact.py   # about 2 minutes
```

## Sprint log

| Sprint | Dates | Goal | Delivered |
|---|---|---|---|
| 0 | 23 Sep 2026 | Set-up | Backlog, tools, accounts |
| 1 | 23 Sep 2026 | Walking skeleton | Simulator, model and back-test, risk map, sub-county detail, model performance page |
| 3a | 23 Sep 2026 | Usable product, part 1 | Alerts feed with verification status, impact simulator (common random numbers across 20 worlds), map polish |
| 3b | 23 Sep 2026 | Usable product, part 2 | Farmer SMS reports with review queue and replies, week-by-week replay, movement view |
| 2 | 24-25 Sep 2026 | Credible AI | Real weather (ERA5) and boundaries, 8-week drought mechanism, anomaly detection layer, data sources page |

## Data sources

- **Real:** constituency/sub-county boundaries from geoBoundaries (gbOpen, KEN ADM2); weekly rainfall and reference evapotranspiration 2013-2025 from ERA5 reanalysis via the Open-Meteo archive API (2013-2022 used as the normal baseline).
- **Simulated:** cattle holdings, animals, movements, farmer and vet reports, vaccination and outbreaks.
- Merti and Transmara East were removed because they have no separate polygon in the boundary file. Molo, Isiolo and Garbatulla use the names of the boundary polygons they sit in (Kuresoi North, Isiolo North, Isiolo South).

## Switching on the AI language engine

Farmer SMS understanding uses a Claude model when an Anthropic API key is available; otherwise a keyword fallback runs.
On Streamlit Community Cloud: app menu → Settings → Secrets, and add:

```
ANTHROPIC_API_KEY = "your-key-here"
```

Never put the key in the code or commit it to GitHub.
