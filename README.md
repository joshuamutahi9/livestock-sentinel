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
| `simulate.py` | Simulates FMD spreading between cattle herds through local contact and animal movement across 28 real Kenyan sub-counties. Produces only the signals that would be observable in reality. |
| `train.py` | Builds features, trains the model (LightGBM), compares it with simple baselines on a held-out year, and creates risk scores with plain-language explanations (SHAP). |
| `app.py` | The dashboard. |
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
```

## Sprint log

| Sprint | Dates | Goal | Delivered |
|---|---|---|---|
| 0 | 23 Sep 2026 | Set-up | Backlog, tools, accounts |
| 1 | 23-25 Sep 2026 | Walking skeleton | Simulator, model and back-test, risk map, sub-county detail, model performance page |
