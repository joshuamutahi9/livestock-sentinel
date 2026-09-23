"""Livestock Sentinel AI - dashboard (Sprint 1 walking skeleton). Run: streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

D = Path(__file__).parent / "data"
BAND_COLORS = {"High": "#B3261E", "Elevated": "#D98A1A", "Low": "#2F7D4F"}
ACTIONS = {
    "High": "Prioritise veterinary verification and sampling in this sub-county. Trace cattle that arrived in the "
            "last 2 weeks and check their areas of origin. Consider a movement advisory for local markets.",
    "Elevated": "Increase reporting: contact farmers and animal health assistants in the area. Review recent "
                "arrivals and check vaccination coverage.",
    "Low": "Routine surveillance.",
}

st.set_page_config(page_title="Livestock Sentinel AI", page_icon="🐄", layout="wide")


@st.cache_data
def load():
    areas = pd.read_csv(D / "areas.csv")
    risk = pd.read_csv(D / "risk_scores.csv", parse_dates=["date"]).merge(
        areas[["area_id", "county", "sub_county", "lat", "lon"]], on="area_id")
    metrics = json.loads((D / "model_metrics.json").read_text())
    return areas, risk, metrics


areas, risk, metrics = load()

st.title("Livestock Sentinel AI")
st.caption("Early warning for foot-and-mouth disease in cattle. Prototype, sprint 1.")
st.warning("All animal, movement, report and outbreak data in this prototype is simulated. "
           "It is not government surveillance data.")

with st.sidebar:
    st.header("View")
    dates = sorted(risk.date.dt.date.unique())
    week = st.select_slider("Week", options=dates, value=dates[20])
    counties = st.multiselect("Counties", sorted(areas.county.unique()), default=sorted(areas.county.unique()))

now = risk[(risk.date.dt.date == week) & (risk.county.isin(counties))].copy()

tab_overview, tab_area, tab_model = st.tabs(["Risk overview", "Sub-county detail", "Model performance"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-risk sub-counties", int((now.band == "High").sum()))
    c2.metric("Elevated", int((now.band == "Elevated").sum()))
    c3.metric("Farmer symptom reports", int(now.farmer_reports.sum()))
    c4.metric("Confirmed outbreaks this week", int(now.confirmed_outbreak.sum()))

    left, right = st.columns([3, 2])
    with left:
        now["size"] = now.score.clip(lower=15)
        fig = px.scatter_map(now, lat="lat", lon="lon", color="band", size="size", size_max=28,
                             color_discrete_map=BAND_COLORS, hover_name="sub_county",
                             hover_data={"county": True, "score": True, "data_confidence": True,
                                         "lat": False, "lon": False, "size": False, "band": False},
                             zoom=5.3, center={"lat": -0.6, "lon": 37.6}, height=520, map_style="open-street-map")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Risk")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Highest risk this week")
        top = now.sort_values("score", ascending=False)[["sub_county", "county", "score", "band", "drivers_up", "data_confidence"]]
        top = top.rename(columns={"sub_county": "Sub-county", "county": "County", "score": "Score",
                                  "band": "Risk", "drivers_up": "Main signals", "data_confidence": "Data"})
        top["Main signals"] = top["Main signals"].fillna("").str.split(" | ", regex=False).str[0]
        st.dataframe(top, hide_index=True, width="stretch", height=470)

with tab_area:
    pick = st.selectbox("Sub-county", now.sort_values("score", ascending=False).sub_county.tolist())
    row = now[now.sub_county == pick].iloc[0]
    a, b = st.columns([1, 2])
    with a:
        st.metric("Risk score", f"{int(row.score)}/100", help="Chance of a new confirmed FMD outbreak in the next 2 weeks, scaled 0-100")
        st.markdown(f"**Risk level:** :{'red' if row.band == 'High' else 'orange' if row.band == 'Elevated' else 'green'}[{row.band}]")
        if row.data_confidence == "Low data":
            st.info("Few reports come from this area, so the score is less reliable. Low risk here may mean missing data.")
        st.markdown("**Why the risk is at this level**")
        for d in str(row.drivers_up).split(" | "):
            if d and d != "nan":
                st.markdown(f"- {d}")
        if isinstance(row.drivers_down, str) and row.drivers_down:
            st.markdown("**Factors lowering risk**")
            for d in row.drivers_down.split(" | "):
                st.markdown(f"- {d}")
        st.markdown("**Recommended action**")
        st.write(ACTIONS[row.band])
    with b:
        hist = risk[risk.sub_county == pick].sort_values("date")
        fig2 = go.Figure()
        fig2.add_hrect(y0=70, y1=100, fillcolor="#B3261E", opacity=0.07, line_width=0)
        fig2.add_hrect(y0=40, y1=70, fillcolor="#D98A1A", opacity=0.07, line_width=0)
        fig2.add_trace(go.Scatter(x=hist.date, y=hist.score, mode="lines", name="Risk score", line=dict(color="#1E3D32", width=2)))
        ob = hist[hist.confirmed_outbreak == 1]
        fig2.add_trace(go.Scatter(x=ob.date, y=[98] * len(ob), mode="markers", name="Outbreak confirmed",
                                  marker=dict(symbol="x", size=12, color="#B3261E")))
        fig2.add_vline(x=pd.Timestamp(week), line_dash="dot", line_color="#555")
        fig2.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(range=[0, 100], title="Risk score"),
                           title="Risk score over the test year", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Look for the risk line rising into the red band before an outbreak is confirmed (✕).")

with tab_model:
    st.subheader("How well does it predict?")
    st.write(f"Tested on {metrics['test_period'].lower()}. {metrics['n_outbreaks_test']} simulated outbreaks in the test year.")
    comp = pd.DataFrame({
        "Model": ["Livestock Sentinel (LightGBM)", "Logistic regression", "Seasonal average", "Random guess"],
        "PR-AUC (higher is better)": [metrics["pr_auc"]["lightgbm"], metrics["pr_auc"]["logistic"],
                                      metrics["pr_auc"]["seasonal_baseline"], metrics["pr_auc"]["random"]],
        "ROC-AUC": [metrics["roc_auc"]["lightgbm"], metrics["roc_auc"]["logistic"], metrics["roc_auc"]["seasonal_baseline"], 0.5],
        "Outbreaks caught in top 5 areas each week": [metrics["recall_top5_areas_per_week"]["lightgbm"],
                                                      metrics["recall_top5_areas_per_week"]["logistic"],
                                                      metrics["recall_top5_areas_per_week"]["seasonal_baseline"], 5 / 28],
    }).round(2)
    st.dataframe(comp, hide_index=True, width="stretch")
    st.write(f"High alert raised before lab confirmation for **{metrics['outbreaks_with_high_alert_before_confirmation']}** "
             f"test-year outbreaks, a median of **{metrics['median_lead_time_weeks']:.0f} weeks** ahead.")
    st.caption("These results show the approach works on simulated data. Real-world accuracy can only be measured in a pilot with real surveillance data.")
