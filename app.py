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
    geo = json.loads((D / "boundaries.geojson").read_text()) if (D / "boundaries.geojson").exists() else None
    return areas, risk, metrics, geo


areas, risk, metrics, geo = load()
risk["anomaly_flags"] = risk.anomaly_flags.fillna("")

st.title("Livestock Sentinel AI")
st.caption("Early warning for foot-and-mouth disease in cattle. Working prototype.")
st.warning("All animal, movement, report and outbreak data in this prototype is simulated. "
           "It is not government surveillance data.")

with st.sidebar:
    st.header("View")
    dates = sorted(risk.date.dt.date.unique())
    week = st.select_slider("Week", options=dates, value=dates[20])
    counties = st.multiselect("Counties", sorted(areas.county.unique()), default=sorted(areas.county.unique()))

now = risk[(risk.date.dt.date == week) & (risk.county.isin(counties))].copy()

tab_overview, tab_alerts, tab_area, tab_impact, tab_model = st.tabs(
    ["Risk overview", "Alerts", "Sub-county detail", "Impact", "Model performance"])

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("High-risk sub-counties", int((now.band == "High").sum()))
    c2.metric("Elevated", int((now.band == "Elevated").sum()))
    c3.metric("Unusual signals", int((now.anomaly_flags.str.len() > 0).sum()),
              help="Sub-counties where reports or cattle arrivals jumped well above their own normal level")
    c4.metric("Farmer symptom reports", int(now.farmer_reports.sum()))
    c5.metric("Confirmed outbreaks", int(now.confirmed_outbreak.sum()))

    left, right = st.columns([3, 2])
    with left:
        hover = {"county": True, "score": True, "data_confidence": True}
        if geo:
            fig = px.choropleth_map(now, geojson=geo, locations="area_id", featureidkey="properties.area_id",
                                    color="band", color_discrete_map=BAND_COLORS, hover_name="sub_county",
                                    category_orders={"band": ["High", "Elevated", "Low"]},
                                    hover_data={**hover, "area_id": False, "band": False}, opacity=0.75,
                                    zoom=5.9, center={"lat": -0.5, "lon": 37.9}, height=520, map_style="carto-positron")
            flagged = now[now.anomaly_flags.str.len() > 0]
            if len(flagged):
                fig.add_trace(go.Scattermap(lat=flagged.lat, lon=flagged.lon, mode="markers", name="Unusual signal",
                                            marker=dict(size=11, color="#1B2420", symbol="circle"),
                                            text=flagged.sub_county, hoverinfo="text"))
        else:
            now["size"] = now.score.clip(lower=15)
            fig = px.scatter_map(now, lat="lat", lon="lon", color="band", size="size", size_max=28,
                                 color_discrete_map=BAND_COLORS, hover_name="sub_county",
                                 category_orders={"band": ["High", "Elevated", "Low"]},
                                 hover_data={**hover, "lat": False, "lon": False, "size": False, "band": False},
                                 zoom=5.9, center={"lat": -0.5, "lon": 37.9}, height=520, map_style="open-street-map")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Risk")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Highest risk this week")
        top = now.sort_values("score", ascending=False)[["sub_county", "county", "score", "band", "drivers_up", "data_confidence"]]
        top = top.rename(columns={"sub_county": "Sub-county", "county": "County", "score": "Score",
                                  "band": "Risk", "drivers_up": "Main signals", "data_confidence": "Data"})
        top["Main signals"] = top["Main signals"].fillna("").str.split(" | ", regex=False).str[0]
        st.dataframe(top, hide_index=True, width="stretch", height=470)

with tab_alerts:
    st.subheader("Alerts to follow up")
    st.caption("High-risk scores and unusual signals from the last 4 weeks. A veterinary officer verifies each one "
               "and records the outcome. The AI supports the decision; people make it.")
    STATUSES = ["New", "Investigating", "Confirmed", "Not confirmed"]
    if "alert_status" not in st.session_state:
        st.session_state.alert_status = {}
    win = risk[(risk.date.dt.date <= week) & (risk.date.dt.date > pd.Timestamp(week).date() - pd.Timedelta(weeks=4))
               & (risk.county.isin(counties))]
    al = win[(win.band == "High") | (win.anomaly_flags.str.len() > 0)].sort_values(["date", "score"], ascending=[False, False])
    status_now = [st.session_state.alert_status.get(f"{r.area_id}-{r.week}", "New") for r in al.itertuples()]
    k1, k2, k3, k4 = st.columns(4)
    for col, name in zip([k1, k2, k3, k4], STATUSES):
        col.metric(name, status_now.count(name))
    if al.empty:
        st.info("No alerts in the last 4 weeks for the selected counties. Routine surveillance continues.")
    for r in al.itertuples():
        key = f"{r.area_id}-{r.week}"
        kind = "High risk" if r.band == "High" else "Unusual signal"
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{r.sub_county}, {r.county}**: {kind} (score {int(r.score)}), week of {r.date:%d %b %Y}")
                reasons = [x for x in (r.anomaly_flags.split(" | ") if r.anomaly_flags else []) +
                           (str(r.drivers_up).split(" | ") if isinstance(r.drivers_up, str) else []) if x][:3]
                st.markdown("  \n".join(f"- {x}" for x in reasons))
                st.caption("Recommended: " + ACTIONS["High" if r.band == "High" else "Elevated"])
            with c2:
                cur = st.session_state.alert_status.get(key, "New")
                st.session_state.alert_status[key] = st.selectbox("Status", STATUSES, index=STATUSES.index(cur), key=f"sel-{key}")
                with st.popover("What happened next?"):
                    st.write("In the simulation, a new outbreak " +
                             ("**was confirmed** in the following 2 weeks." if r.target else "was **not** confirmed in the following 2 weeks."))
                    st.caption("Shown only in this prototype, to demonstrate accuracy. A real officer would not know this in advance.")
    st.caption("In this prototype, statuses are kept only for your browser session. In a pilot they would be stored "
               "and used as new training data, so the model learns from every verified alert.")

with tab_area:
    pick = st.selectbox("Sub-county", now.sort_values("score", ascending=False).sub_county.tolist())
    row = now[now.sub_county == pick].iloc[0]
    a, b = st.columns([1, 2])
    with a:
        st.metric("Risk score", f"{int(row.score)}/100", help="Chance of a new confirmed FMD outbreak in the next 2 weeks, scaled 0-100")
        st.markdown(f"**Risk level:** :{'red' if row.band == 'High' else 'orange' if row.band == 'Elevated' else 'green'}[{row.band}]")
        if row.data_confidence == "Low data":
            st.info("Few reports come from this area, so the score is less reliable. Low risk here may mean missing data.")
        if row.anomaly_flags:
            st.markdown("**Unusual signals this week**")
            for f in row.anomaly_flags.split(" | "):
                st.markdown(f"- ⚠️ {f}")
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

with tab_impact:
    st.subheader("What difference could early action make?")
    imp_path = D / "impact_results.json"
    if not imp_path.exists():
        st.info("Impact results have not been generated yet.")
    else:
        imp = json.loads(imp_path.read_text()); m = imp["metrics"]
        st.write(f"We ran the same simulated year in **{imp['n_worlds']} different simulated worlds**, twice each:")
        st.markdown("- **Status quo:** authorities act only after lab confirmation (quarantine, ring vaccination, market closure).\n"
                    "- **With Livestock Sentinel:** the same response to confirmations, plus 4 weeks of early targeted action "
                    "(verification, movement checks, targeted vaccination) whenever a High alert fires.")
        def pct(k): return (1 - m[k]["sentinel"] / m[k]["status_quo"]) * 100 if m[k]["status_quo"] else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Herd infections avoided", f"{m['herds_infected']['status_quo'] - m['herds_infected']['sentinel']:.0f} per year",
                  f"-{pct('herds_infected'):.0f}%", delta_color="inverse")
        c2.metric("Cattle in infected herds avoided", f"{m['animals_infected']['status_quo'] - m['animals_infected']['sentinel']:,.0f} per year",
                  f"-{pct('animals_infected'):.0f}%", delta_color="inverse")
        c3.metric("Worlds where early action helped", f"{m['herds_infected']['worlds_improved']} of {imp['n_worlds']}")
        runs = pd.read_csv(D / "impact_runs.csv")
        pv = runs.pivot(index="seed", columns="policy", values="herds_infected").reset_index()
        pv["World"] = [f"W{i+1}" for i in range(len(pv))]
        fig3 = go.Figure([go.Bar(name="Status quo", x=pv.World, y=pv.status_quo, marker_color="#8A8F8C"),
                          go.Bar(name="With Livestock Sentinel", x=pv.World, y=pv.sentinel, marker_color="#1E3D32")])
        fig3.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=30, b=0),
                           title="Herd infections in each simulated world", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, width="stretch")
        st.markdown("**Estimated economic value**")
        loss = st.number_input("Assumed loss per infected animal (KSh): milk, weight, treatment and sales lost",
                               min_value=0, value=10000, step=1000)
        avoided = m["animals_infected"]["status_quo"] - m["animals_infected"]["sentinel"]
        st.write(f"At this assumption, early action avoids roughly **KSh {avoided * loss:,.0f}** in losses per year "
                 f"across these 26 sub-counties. Change the figure above to test other assumptions.")
        st.markdown("**The honest picture**")
        st.markdown(f"- The benefit comes from fewer infections. Confirmed outbreaks and market closures stayed roughly the same "
                    f"(status quo {m['market_closure_weeks']['status_quo']:.0f} vs {m['market_closure_weeks']['sentinel']:.0f} market-closure weeks), "
                    "because outbreaks were still detected and responded to.\n"
                    f"- Early action has a cost: about **{m['early_action_weeks']['sentinel']:.0f} sub-county-weeks** of extra "
                    "verification work per year, roughly 3 sub-counties a week.\n"
                    "- These are modelled estimates from simulated data. A pilot would measure the real effect.")

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
    st.write(f"The separate anomaly check flagged **{metrics.get('anomaly_flag_before_confirmation', 'n/a')}** outbreaks before "
             f"confirmation. On average the system raises about **{metrics.get('high_alerts_per_week', 0):.1f} high alerts** and "
             f"**{metrics.get('anomaly_flags_per_week', 0):.1f} unusual-signal flags** per week across 28 sub-counties.")
    st.subheader("Data sources")
    st.markdown("- **Real:** sub-county boundaries (geoBoundaries), weekly rainfall and evaporation 2013-2025 (ERA5 reanalysis via Open-Meteo).\n"
                "- **Simulated:** cattle holdings, animals, movements, farmer and vet reports, vaccination, outbreaks.")
    st.caption("These results show the approach works on simulated data. Real-world accuracy can only be measured in a pilot with real surveillance data.")
