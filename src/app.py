"""
app.py  —  Route Delay Predictor + LLM Explainer  (Streamlit dashboard)

Run:
    streamlit run src/app.py

Pages
-----
1. Predict & Explain   — fill inputs → get delay prediction + Gemini explanation
2. SHAP Feature Impact — global feature importance from the saved SHAP values
3. Model Performance   — eval metrics card + prediction distribution
4. Chat with RouteBot  — multi-turn Gemini agent about route delays
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from explainer_agent import build_context, chat_agent, explain_prediction
from train_model import load_model, predict_single

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="NYC Bus Delay Predictor",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_shap_artefacts():
    sv_path = MODELS_DIR / "shap_values.npy"
    ev_path = MODELS_DIR / "shap_expected_value.npy"
    fn_path = MODELS_DIR / "feature_names.json"
    if not sv_path.exists():
        return None, None, None
    sv = np.load(sv_path)
    ev = float(np.load(ev_path)[0])
    with open(fn_path) as f:
        fn = json.load(f)
    return sv, ev, fn


@st.cache_data(show_spinner=False)
def get_eval_metrics():
    path = MODELS_DIR / "eval_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar — navigation + global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚌 Route Delay Predictor")
    st.caption("XGBoost + Gemini · NYC MTA Bus")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Predict & Explain", "SHAP Feature Impact", "Model Performance", "Chat with RouteBot"],
        index=0,
    )
    st.divider()

    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
    if gemini_ok:
        st.success("Gemini API connected", icon="✅")
    else:
        st.warning("GEMINI_API_KEY not set — AI explanations disabled", icon="⚠️")

    model_ready = (MODELS_DIR / "xgb_delay_model.json").exists()
    if model_ready:
        st.success("Model loaded", icon="✅")
    else:
        st.error("No trained model found.\nRun `python src/train_model.py` first.", icon="🔴")

    st.divider()
    st.caption("Built with XGBoost · SHAP · Gemini · Streamlit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROUTE_OPTIONS = [
    "M15", "M15+", "M86", "M104", "M11",
    "B46", "B44", "B35",
    "Q58", "Q44",
    "Bx12", "Bx41",
]

# Only conditions that actually appear in the model (based on what OWM returned
# during collection). Others exist in the UI for UX but have no model effect.
WEATHER_CONDITIONS      = ["Clear", "Clouds", "Rain", "Snow", "Fog", "Drizzle", "Thunderstorm"]
WEATHER_IN_MODEL        = {"Clear", "Clouds", "Rain"}   # wx_* one-hots present in trained model

# direction_code: derived alphabetically from the raw "direction" string column.
# Mapping approximated from common MTA direction labels in the speeds dataset.
DIRECTION_OPTIONS = {
    "Inbound (toward Manhattan / hub)": 0,
    "Outbound (away from hub)":         1,
    "Northbound / Eastbound":           2,
    "Southbound / Westbound":           3,
}

# borough_code: 0=Brooklyn, 1=Manhattan, 2=Queens  (alphabetical cat.codes)
# route_type_code: 0=Limited, 1=Local, 2=SBS, 3=School
# Values derived from actual training data: avg_road_speed mean=6.86, avg_travel_time mean=9.95
ROUTE_PROFILES: dict[str, dict] = {
    #              borough  rtype  speed  travel_t  trips  cjt_hr  add_stop
    "M15":  dict(borough_code=1, route_type_code=1, avg_speed=6.5,  avg_tt=9.5,  trips=18, cjt_hr=0.72, add_stop=2.8),
    "M15+": dict(borough_code=1, route_type_code=2, avg_speed=8.0,  avg_tt=7.5,  trips=14, cjt_hr=0.66, add_stop=2.2),
    "M86":  dict(borough_code=1, route_type_code=1, avg_speed=6.0,  avg_tt=10.5, trips=10, cjt_hr=0.74, add_stop=3.0),
    "M104": dict(borough_code=1, route_type_code=1, avg_speed=5.8,  avg_tt=11.0, trips=10, cjt_hr=0.76, add_stop=3.2),
    "M11":  dict(borough_code=1, route_type_code=1, avg_speed=7.0,  avg_tt=8.5,  trips=12, cjt_hr=0.70, add_stop=2.6),
    "B46":  dict(borough_code=0, route_type_code=1, avg_speed=6.8,  avg_tt=10.0, trips=20, cjt_hr=0.73, add_stop=2.9),
    "B44":  dict(borough_code=0, route_type_code=1, avg_speed=7.2,  avg_tt=9.0,  trips=18, cjt_hr=0.71, add_stop=2.7),
    "B35":  dict(borough_code=0, route_type_code=1, avg_speed=6.5,  avg_tt=10.5, trips=16, cjt_hr=0.74, add_stop=3.0),
    "Q58":  dict(borough_code=2, route_type_code=1, avg_speed=7.0,  avg_tt=11.5, trips=14, cjt_hr=0.75, add_stop=3.1),
    "Q44":  dict(borough_code=2, route_type_code=2, avg_speed=8.5,  avg_tt=8.0,  trips=12, cjt_hr=0.67, add_stop=2.3),
    "Bx12": dict(borough_code=1, route_type_code=1, avg_speed=7.0,  avg_tt=9.0,  trips=20, cjt_hr=0.72, add_stop=2.8),
    "Bx41": dict(borough_code=1, route_type_code=1, avg_speed=6.5,  avg_tt=10.0, trips=18, cjt_hr=0.73, add_stop=2.9),
}


def delay_status(minutes: float) -> dict:
    """Return emoji, label, and Streamlit colour key for a predicted delay."""
    if minutes < -5:
        return dict(emoji="🟢", label="Running well early",       color="green",  st_color="normal")
    elif minutes < -2:
        return dict(emoji="🟢", label="Running slightly early",   color="green",  st_color="normal")
    elif minutes < 0:
        return dict(emoji="🟢", label="On time / slightly ahead", color="green",  st_color="normal")
    elif minutes < 2:
        return dict(emoji="🟡", label="On time / minor variance", color="yellow", st_color="off")
    elif minutes < 5:
        return dict(emoji="🟠", label="Slight delay",             color="orange", st_color="off")
    elif minutes < 10:
        return dict(emoji="🔴", label="Moderate delay",           color="red",    st_color="inverse")
    else:
        return dict(emoji="🚨", label="Significant delay",        color="red",    st_color="inverse")


# Keep thin wrappers so any other call sites still work
def delay_color(minutes: float) -> str:
    return delay_status(minutes)["color"]


def delay_label(minutes: float) -> str:
    return delay_status(minutes)["label"]


def build_input_dict(inputs: dict, feature_names: list[str]) -> dict[str, float]:
    """Map UI widget values onto the full feature vector.

    Keys must exactly match what feature_engineering.py produces.
    Route-specific defaults come from ROUTE_PROFILES so that changing the
    route dropdown actually changes the model input.
    """
    route   = inputs["route"]
    profile = ROUTE_PROFILES.get(route, ROUTE_PROFILES["M15"])

    h     = inputs["hour"]
    dow   = inputs["day_of_week"]       # 0=Mon … 6=Sun (int)
    month = inputs["month"]
    rain  = inputs["rain_1h_mm"]
    snow  = inputs["snow_1h_mm"]
    precip = rain + snow

    avg_tt    = inputs.get("average_travel_time", profile["avg_tt"])
    avg_speed = inputs.get("average_road_speed",  profile["avg_speed"])
    trip_cnt  = inputs.get("bus_trip_count",       profile["trips"])

    # customer_journey_time is stored in the model as hours (0.53–0.87).
    # UI collects it in minutes → divide by 60.
    cjt_min = inputs.get("customer_journey_time_min", profile["cjt_hr"] * 60)
    cjt_hr  = cjt_min / 60.0

    # additional_bus_stop_time lives in the model in minutes (0.95–4.49).
    add_stop = inputs.get("additional_bus_stop_time", profile["add_stop"])

    d: dict[str, float] = {
        # ── Temporal ──────────────────────────────────────────────────────
        "hour_of_day":  h,
        "hour_sin":     np.sin(2 * np.pi * h / 24),
        "hour_cos":     np.cos(2 * np.pi * h / 24),
        "dow_int":      dow,
        "dow_sin":      np.sin(2 * np.pi * dow / 7),
        "dow_cos":      np.cos(2 * np.pi * dow / 7),
        "is_weekend":   int(dow >= 5),
        "is_rush_hour": int(7 <= h <= 9 or 16 <= h <= 19),
        "month_num":    month,          # BUG FIX: was "month" — model key is "month_num"
        "month_sin":    np.sin(2 * np.pi * month / 12),
        "month_cos":    np.cos(2 * np.pi * month / 12),
        "year":         2026,

        # ── Route operational features ────────────────────────────────────
        # IMPORTANT: average_travel_time, average_road_speed, road_distance,
        # speed_ratio, trip_volume_norm were REMOVED from the model as data
        # leakage. Do NOT add them back. Only bus_trip_count remains.
        "bus_trip_count": trip_cnt,

        # ── Journey / delay history ───────────────────────────────────────
        "route_delay_roll3":        inputs.get("route_delay_roll3", 0.57),
        "additional_bus_stop_time": add_stop,
        "customer_journey_time":    cjt_hr,

        # ── Categoricals ─────────────────────────────────────────────────
        "borough_code":    profile["borough_code"],
        "route_type_code": profile["route_type_code"],
        "direction_code":  inputs.get("direction_code", 1),  # set from UI widget

        # ── Weather ───────────────────────────────────────────────────────
        "temp_c":              inputs["temp_c"],
        "humidity_pct":        inputs["humidity_pct"],
        "pressure_hpa":        inputs.get("pressure_hpa", 1013.0),
        "wind_speed_ms":       inputs["wind_speed_ms"],
        "clouds_pct":          inputs.get("clouds_pct", 0),
        "rain_1h_mm":          rain,
        "snow_1h_mm":          snow,
        "pop":                 inputs.get("pop", 0.0),
        "is_raining":          int(rain > 0),
        "is_snowing":          int(snow > 0),
        "precip_intensity": (
            0 if precip == 0 else 1 if precip <= 2.5 else 2 if precip <= 7.6 else 3
        ),
        "apparent_temp_delta": inputs.get("wind_chill_delta", 0),
        "visibility_bucket":   inputs.get("visibility_bucket", 2),
    }

    # Only emit wx_* one-hots for conditions actually in the model.
    # wx_Snow, wx_Fog, wx_Thunderstorm, wx_Drizzle aren't in feature_names
    # (OWM only returned Clear/Clouds/Rain during data collection) so they
    # would be silently dropped by the return statement anyway.
    for cond in WEATHER_IN_MODEL:
        d[f"wx_{cond}"] = int(inputs.get("weather_condition") == cond)

    # Return only features the model knows, in exact training order
    return {f: d.get(f, 0.0) for f in feature_names}


# ===========================================================================
# Page 1 — Predict & Explain
# ===========================================================================

if page == "Predict & Explain":
    st.header("Predict Bus Delay & Get AI Explanation")

    if not model_ready:
        st.error("Train the model first: `python src/train_model.py`")
        st.stop()

    model, feature_names = get_model()

    # ── What delay range to expect ──────────────────────────────────────────
    with st.expander("💡 Understanding predictions — why are delays small?", expanded=False):
        st.markdown("""
The model predicts **deviation from each route's historical median travel time**.
Most segments run within ±3 minutes of their median, so predictions in the 0–2 min
range are realistic for average conditions. The full target distribution is:

| Range | Meaning | % of data |
|---|---|---|
| `< −5 min` | Bus running well ahead of schedule | 11% |
| `−5 to 0 min` | Slightly ahead of typical | 32% |
| `0 to +2 min` | On time / minor variance | 18% |
| `+2 to +5 min` | Moderate delay | 24% |
| `+5 to +10 min` | Significant delay | 10% |
| `> +10 min` | Heavy delay | 4% |

**To see larger delays (+5–10 min), try:**
- 🕗 Hour: **8** (morning rush) or **17** (evening rush)
- 🚌 Route: **M15+** or **Q44** (SBS — highest delay variance)
- 📅 Day: **Monday or Friday**
- 📈 Route delay history: slide **3-month rolling delay** toward **1.5**
- 🌧️ Weather: Rain + high wind speed *(once weather data is in model)*

**Note on weather:** weather inputs only affect predictions if the model was
trained with weather data. Re-run the full pipeline after your OWM key activates.
        """)

    # ── Input form ──────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("Route & Timing")
        c1, c2, c3 = st.columns(3)
        route = c1.selectbox("Route", ROUTE_OPTIONS)
        hour = c2.slider("Hour of day", 0, 23, 8)
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_of_week = c3.selectbox("Day of week", range(7), format_func=lambda i: dow_names[i])

        c4, _ = st.columns(2)
        month = c4.slider("Month", 1, 12, 3)

        # Show route profile so the user knows what's being auto-filled
        _p = ROUTE_PROFILES.get(route, ROUTE_PROFILES["M15"])
        borough_labels = {0: "Brooklyn", 1: "Manhattan", 2: "Queens"}
        rtype_labels   = {0: "Limited", 1: "Local", 2: "SBS", 3: "School"}
        st.caption(
            f"📍 **{route}** profile — Borough: **{borough_labels[_p['borough_code']]}** · "
            f"Type: **{rtype_labels[_p['route_type_code']]}** · "
            f"Typical speed: **{_p['avg_speed']} km/h** · Seg. travel time: **{_p['avg_tt']} min**"
        )

        # Check if the loaded model has any weather features
        _weather_in_model = any(
            any(w in feat for w in ["rain", "temp_c", "wind", "wx_", "snow", "humidity"])
            for feat in feature_names
        )
        st.subheader("Weather Conditions")
        if not _weather_in_model:
            st.warning(
                "⚠️ **Weather inputs have no effect on this prediction.** "
                "The model was trained without weather data because the OpenWeatherMap API "
                "key was not active during data collection. "
                "Once your OWM key activates, re-run `data_collection.py` → `feature_engineering.py` → `train_model.py` "
                "to include weather features.",
                icon="🌧️",
            )
        w1, w2, w3, w4 = st.columns(4)
        temp_c       = w1.number_input("Temp (°C)", -20.0, 40.0, 8.0, step=0.5)
        wind_speed   = w2.number_input("Wind speed (m/s)", 0.0, 30.0, 4.0, step=0.5)
        humidity     = w3.slider("Humidity (%)", 0, 100, 65)
        weather_cond = w4.selectbox("Condition", WEATHER_CONDITIONS)

        w5, w6, w7 = st.columns(3)
        rain_mm    = w5.number_input("Rain last 1h (mm)", 0.0, 50.0, 0.0, step=0.5)
        snow_mm    = w6.number_input("Snow last 1h (mm)", 0.0, 30.0, 0.0, step=0.5)
        visibility = w7.selectbox(
            "Visibility", [0, 1, 2],
            format_func=lambda v: ["Poor (<1km)", "Moderate (1–5km)", "Good (>5km)"][v],
            index=2,
        )

        st.subheader("Route Conditions")
        rc1, rc2 = st.columns(2)
        trip_count = rc1.number_input(
            "Active trip count", 1, 60, int(_p["trips"]), step=1,
            help="Number of buses running on this route right now. More trips = more congestion."
        )
        direction = rc2.selectbox(
            "Direction of travel",
            options=list(DIRECTION_OPTIONS.keys()),
            index=1,
            help="Inbound/Outbound affects direction_code — the #2 SHAP feature in this model."
        )

        st.subheader("Journey History")
        r1, r2 = st.columns(2)
        roll3 = r1.number_input(
            "3-month rolling avg delay (min)", -0.4, 1.6, 0.57, step=0.05,
            help="Recent delay trend for this route. Training range −0.36 to 1.55, mean 0.57."
        )
        cust_journey_min = r2.number_input(
            "Customer journey time (min)", 30.0, 55.0, float(round(_p["cjt_hr"] * 60, 1)), step=1.0,
            help="Typical end-to-end journey time (stored internally as hours)."
        )

        submitted = st.form_submit_button("Predict Delay", type="primary", use_container_width=True)

    # ── Prediction ──────────────────────────────────────────────────────────
    if submitted:
        inputs = {
            "route":                     route,
            "hour":                      hour,
            "day_of_week":               day_of_week,
            "month":                     month,
            # route conditions
            "bus_trip_count":            trip_count,
            "direction_code":            DIRECTION_OPTIONS[direction],
            # weather
            "temp_c":                    temp_c,
            "wind_speed_ms":             wind_speed,
            "humidity_pct":              humidity,
            "weather_condition":         weather_cond,
            "rain_1h_mm":                rain_mm,
            "snow_1h_mm":                snow_mm,
            "visibility_bucket":         visibility,
            # journey history
            "route_delay_roll3":         roll3,
            "customer_journey_time_min": cust_journey_min,
        }

        input_dict = build_input_dict(inputs, feature_names)
        predicted_delay, sv = predict_single(model, feature_names, input_dict)
        status = delay_status(predicted_delay)

        # ── Metric cards ────────────────────────────────────────────────────
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Delay", f"{predicted_delay:+.1f} min")
        m2.metric("Status", f"{status['emoji']} {status['label']}")
        m3.metric("Route", route)

        # ── Colour-coded status banner ───────────────────────────────────────
        banner_styles = {
            "green":  ("🟢", "#1a472a", "#d4edda"),
            "yellow": ("🟡", "#856404", "#fff3cd"),
            "orange": ("🟠", "#7d3a00", "#ffe5cc"),
            "red":    ("🔴", "#721c24", "#f8d7da"),
        }
        icon, text_col, bg_col = banner_styles[status["color"]]
        st.markdown(
            f"""<div style="background:{bg_col};border-radius:8px;padding:12px 18px;
                margin:6px 0;color:{text_col};font-size:1.05rem;font-weight:600;">
                {icon}&nbsp;&nbsp;{status['label']}
                &nbsp;—&nbsp; {predicted_delay:+.1f} min vs this route's typical travel time
                </div>""",
            unsafe_allow_html=True,
        )

        # ── SHAP waterfall bar chart ─────────────────────────────────────────
        st.subheader("What's driving this prediction?")
        from explainer_agent import FEATURE_LABELS
        top_n = 8
        pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        feat_labels = [FEATURE_LABELS.get(f, f.replace("_", " ")) for f, _ in pairs]
        shap_vals = [v for _, v in pairs]

        fig = go.Figure(go.Bar(
            x=shap_vals,
            y=feat_labels,
            orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in shap_vals],
        ))
        fig.update_layout(
            title="Feature impact on predicted delay (minutes)",
            xaxis_title="SHAP value (minutes added to delay)",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Gemini explanation ───────────────────────────────────────────────
        st.subheader("AI Explanation")
        if gemini_ok:
            with st.spinner("RouteBot is thinking…"):
                ctx = build_context(
                    route=route,
                    predicted_delay=predicted_delay,
                    feature_names=feature_names,
                    feature_values=input_dict,
                    shap_values=sv,
                )
                try:
                    explanation = explain_prediction(ctx)
                    st.info(explanation, icon="🤖")
                except Exception as e:
                    st.warning(f"Gemini unavailable: {e}")
        else:
            st.warning("Set GEMINI_API_KEY in .env to enable AI explanations.")

        # Store context for chat page
        st.session_state["last_ctx_inputs"] = inputs
        st.session_state["last_ctx_sv"] = sv
        st.session_state["last_ctx_delay"] = predicted_delay
        st.session_state["last_ctx_feature_names"] = feature_names
        st.session_state["last_ctx_input_dict"] = input_dict


# ===========================================================================
# Page 2 — SHAP Feature Impact
# ===========================================================================

elif page == "SHAP Feature Impact":
    st.header("Global Feature Importance (SHAP)")

    shap_values, expected_value, feat_names = get_shap_artefacts()

    if shap_values is None:
        st.error("No SHAP artefacts found. Run `python src/train_model.py` first.")
        st.stop()

    from explainer_agent import FEATURE_LABELS

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_n = st.slider("Number of features to show", 5, min(30, len(feat_names)), 15)

    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    labels = [FEATURE_LABELS.get(feat_names[i], feat_names[i].replace("_", " ")) for i in top_idx]
    values = [mean_abs_shap[i] for i in top_idx]

    # ── Bar chart ───────────────────────────────────────────────────────────
    fig = px.bar(
        x=values, y=labels,
        orientation="h",
        labels={"x": "Mean |SHAP| (minutes)", "y": ""},
        color=values,
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        title=f"Top {top_n} features by mean absolute SHAP value",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=max(400, top_n * 28),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP distribution beeswarm approximation (violin) ───────────────────
    st.subheader("SHAP Value Distribution — Top 10 Features")
    top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
    rows = []
    for i in top10_idx:
        label = FEATURE_LABELS.get(feat_names[i], feat_names[i].replace("_", " "))
        for v in shap_values[:, i]:
            rows.append({"Feature": label, "SHAP value (min)": v})

    df_violin = pd.DataFrame(rows)
    fig2 = px.violin(
        df_violin, x="SHAP value (min)", y="Feature",
        orientation="h", box=True, points=False,
        color="Feature", color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig2.update_layout(
        showlegend=False, height=500,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        f"Base prediction (SHAP expected value): **{expected_value:+.2f} minutes**  "
        f"· Computed on {shap_values.shape[0]:,} test samples."
    )


# ===========================================================================
# Page 3 — Model Performance
# ===========================================================================

elif page == "Model Performance":
    st.header("Model Performance")

    metrics = get_eval_metrics()
    if metrics is None:
        st.error("No eval metrics found. Run `python src/train_model.py` first.")
        st.stop()

    # ── Metric cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE",    f"{metrics['mae_minutes']} min",
              help="Mean Absolute Error — average prediction error in minutes")
    c2.metric("RMSE",   f"{metrics['rmse_minutes']} min",
              help="Root Mean Squared Error — penalises large errors more")
    c3.metric("R²",     metrics["r2_score"],
              help="0 = predicts mean, 1 = perfect. 0.3+ is reasonable for route-level aggregates.")
    c4.metric("Within 2 min", f"{metrics.get('within_2min_pct', '—')}%",
              help="% of predictions within 2 minutes of actual delay")
    c5.metric("Within 5 min", f"{metrics.get('within_5min_pct', '—')}%",
              help="% of predictions within 5 minutes of actual delay")

    st.info(
        "**Why no MAPE?** This target (`travel_time_vs_median`) contains near-zero "
        "and negative values. MAPE divides by the true value, so it explodes to "
        "1000%+ when the true delay is close to 0 — even if the model is accurate. "
        "**Within-N-min accuracy** is a much more meaningful metric here.",
        icon="ℹ️",
    )
    st.caption(f"Evaluated on {metrics['n_test_samples']:,} held-out test samples (most recent months, never seen during training).")
    st.divider()

    # ── Gauge: MAE ──────────────────────────────────────────────────────────
    st.subheader("MAE Gauge — how many minutes off on average?")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics["mae_minutes"],
        title={"text": "Mean Absolute Error (minutes)"},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, 2], "color": "#2ecc71"},
                {"range": [2, 5], "color": "#f39c12"},
                {"range": [5, 10], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": metrics["mae_minutes"],
            },
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP summary numbers ─────────────────────────────────────────────────
    shap_values, expected_value, feat_names = get_shap_artefacts()
    if shap_values is not None:
        st.subheader("SHAP Summary")
        from explainer_agent import FEATURE_LABELS
        mean_abs = np.abs(shap_values).mean(axis=0)
        top3_idx = np.argsort(mean_abs)[::-1][:3]
        st.write("**Top 3 most impactful features:**")
        for rank, i in enumerate(top3_idx, 1):
            label = FEATURE_LABELS.get(feat_names[i], feat_names[i])
            st.write(f"{rank}. **{label}** — avg impact: {mean_abs[i]:.3f} min")


# ===========================================================================
# Page 4 — Chat with RouteBot
# ===========================================================================

elif page == "Chat with RouteBot":
    st.header("Chat with RouteBot 🤖")
    st.caption("Ask anything about your bus delay prediction. Multi-turn conversation powered by Gemini.")

    if not gemini_ok:
        st.error("GEMINI_API_KEY is not set. Add it to your .env file.")
        st.stop()

    # ── Initialise session state ─────────────────────────────────────────────
    if "chat_session" not in st.session_state:
        last_inputs = st.session_state.get("last_ctx_inputs")
        last_sv = st.session_state.get("last_ctx_sv")
        last_delay = st.session_state.get("last_ctx_delay")
        last_fn = st.session_state.get("last_ctx_feature_names")
        last_dict = st.session_state.get("last_ctx_input_dict")

        if last_inputs and last_sv is not None:
            ctx = build_context(
                route=last_inputs["route"],
                predicted_delay=last_delay,
                feature_names=last_fn,
                feature_values=last_dict,
                shap_values=last_sv,
            )
        else:
            ctx = None

        with st.spinner("Starting RouteBot…"):
            st.session_state["chat_session"] = chat_agent(ctx)
            st.session_state["chat_history"] = []
            if ctx:
                seed_msg = (
                    f"I just ran a prediction for route **{last_inputs['route']}** "
                    f"showing a **{last_delay:+.1f} min** delay. Ask me anything!"
                )
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": seed_msg}
                )

    # ── Render history ───────────────────────────────────────────────────────
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask RouteBot about delays, weather, or your route…"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("RouteBot is thinking…"):
                try:
                    reply = st.session_state["chat_session"].send(prompt)
                except Exception as e:
                    reply = f"Sorry, I ran into an error: {e}"
            st.markdown(reply)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    # ── Reset button ─────────────────────────────────────────────────────────
    if st.button("Reset conversation", type="secondary"):
        del st.session_state["chat_session"]
        del st.session_state["chat_history"]
        st.rerun()
