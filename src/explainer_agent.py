"""
explainer_agent.py

LLM-powered explanation layer on top of the XGBoost model.

Uses the google-genai SDK (Gemini 2.0 Flash) to turn raw SHAP values +
model predictions into plain-English explanations a commuter can understand.

Exposes:
  explain_prediction()   — one-shot explanation for a single prediction
  chat_agent()           — stateful multi-turn agent for route Q&A
  batch_explain()        — explain a list of predictions (returns DataFrame)

Environment variables:
  GEMINI_API_KEY  — required
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import google.genai as genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

GEMINI_MODEL = "gemini-3.1-flash-lite-preview" 

SYSTEM_PROMPT = """You are RouteBot, an expert NYC transit analyst embedded in a
bus delay prediction dashboard. Your job is to explain machine-learning predictions
to everyday commuters in clear, friendly language.

Guidelines:
- Give a structured explanation in 4-6 sentences covering:
    1. The overall delay verdict and what it means practically.
    2. The single biggest factor driving the delay (from the top SHAP contributors).
    3. One or two secondary factors worth mentioning.
    4. A practical takeaway or tip for the commuter (e.g. leave earlier, check SBS).
- Always translate technical feature names into plain English:
    "hour_sin / hour_cos"       → "time of day"
    "rain_1h_mm"                → "recent rainfall"
    "route_delay_roll3"         → "this route's recent delay history"
    "average_travel_time"       → "how long buses take between stops"
    "average_road_speed"        → "current traffic speed"
    "customer_journey_time"     → "typical end-to-end journey time"
    "additional_bus_stop_time"  → "extra time spent at bus stops"
    "borough_code"              → "the borough this route runs through"
    "bus_trip_count"            → "how many buses are running"
- If a delay is under 2 minutes: reassure the commuter — this is within normal variation.
- If a delay is 2–5 minutes: acknowledge it, name the cause, give a simple tip.
- If a delay is over 5 minutes: be empathetic, explain the main causes clearly,
  and suggest a concrete action (e.g. check MTA app, consider alternate route).
- Never mention SHAP, XGBoost, machine learning, or model internals
  unless the user explicitly asks.
- Tone: friendly, calm, informative — like a knowledgeable friend, not a robot.
"""

# Human-readable labels for feature names shown to the model
FEATURE_LABELS: dict[str, str] = {
    "hour": "hour of day",
    "hour_sin": "time of day (cyclical)",
    "hour_cos": "time of day (cyclical)",
    "day_of_week": "day of the week",
    "dow_sin": "day of week (cyclical)",
    "dow_cos": "day of week (cyclical)",
    "is_weekend": "weekend flag",
    "is_rush_hour": "rush-hour period",
    "month": "month of year",
    "distance_from_stop_m": "distance from next stop",
    "stops_away": "stops away from destination",
    "bearing": "vehicle heading",
    "longitude": "vehicle longitude",
    "latitude": "vehicle latitude",
    "direction": "route direction",
    "route_code": "bus route",
    "route_delay_roll5": "recent delay trend (last 5 obs)",
    "route_delay_roll15": "recent delay trend (last 15 obs)",
    "route_hour_avg_delay": "typical delay for this route at this hour",
    "temp_c": "temperature",
    "feels_like_c": "feels-like temperature",
    "humidity_pct": "humidity",
    "pressure_hpa": "atmospheric pressure",
    "wind_speed_ms": "wind speed",
    "wind_deg": "wind direction",
    "clouds_pct": "cloud cover",
    "rain_1h_mm": "recent rainfall",
    "snow_1h_mm": "recent snowfall",
    "pop": "chance of precipitation",
    "is_raining": "currently raining",
    "is_snowing": "currently snowing",
    "precip_intensity": "precipitation intensity",
    "apparent_temp_delta": "wind chill / heat index effect",
    "visibility_bucket": "road visibility",
    "uvi": "UV index",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionContext:
    """Everything needed to explain one prediction."""
    route: str
    predicted_delay_minutes: float
    feature_values: dict[str, float]
    shap_values: np.ndarray
    feature_names: list[str]
    top_n: int = 5

    @property
    def top_factors(self) -> list[dict]:
        """Return top-N features sorted by |SHAP| descending."""
        pairs = list(zip(self.feature_names, self.shap_values))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name.replace("_", " ")),
                "value": round(float(self.feature_values.get(name, 0)), 3),
                "shap_impact_minutes": round(float(shap_val), 3),
            }
            for name, shap_val in pairs[: self.top_n]
        ]

    def to_prompt_context(self) -> str:
        """Serialise context into a structured string for the LLM prompt."""
        delay_str = (
            f"{self.predicted_delay_minutes:+.1f} minutes"
            if self.predicted_delay_minutes != 0
            else "on time"
        )
        factors_str = "\n".join(
            f"  • {f['label']}: {f['value']} "
            f"(adds {f['shap_impact_minutes']:+.2f} min to delay)"
            for f in self.top_factors
        )
        return (
            f"Route: {self.route}\n"
            f"Predicted delay: {delay_str}\n"
            f"Top contributing factors:\n{factors_str}"
        )


@dataclass
class ChatSession:
    """Wraps a stateful Gemini multi-turn chat."""
    client: genai.Client
    history: list = field(default_factory=list)
    _chat: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._chat = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=512,
            ),
        )

    def send(self, message: str) -> str:
        response = self._chat.send_message(message)
        return response.text


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def explain_prediction(ctx: PredictionContext) -> str:
    """
    Generate a plain-English explanation for a single bus delay prediction.

    Parameters
    ----------
    ctx : PredictionContext with prediction + SHAP values.

    Returns
    -------
    str — natural-language explanation from Gemini.
    """
    client = _build_client()
    prompt = (
        "A commuter is asking why their bus is delayed. "
        "Here is the prediction data:\n\n"
        f"{ctx.to_prompt_context()}\n\n"
        "Please explain this prediction to the commuter in plain English."
    )

    log.info(
        "Requesting Gemini explanation for route %s (delay=%.1f min)",
        ctx.route,
        ctx.predicted_delay_minutes,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.5,
            max_output_tokens=512,   # increased from 256 for detailed explanations
        ),
    )
    explanation = response.text.strip()
    log.info("Explanation received (%d chars)", len(explanation))
    return explanation


def explain_with_comparison(
    ctx: PredictionContext,
    historical_avg_delay: float,
) -> str:
    """
    Explain the prediction relative to the historical average for that route/hour.

    Parameters
    ----------
    historical_avg_delay : typical delay in minutes for this route at this hour.
    """
    client = _build_client()
    delta = ctx.predicted_delay_minutes - historical_avg_delay
    direction = "worse than" if delta > 0 else "better than"

    prompt = (
        f"A commuter is checking their bus status.\n\n"
        f"{ctx.to_prompt_context()}\n\n"
        f"Historical average delay for this route at this hour: "
        f"{historical_avg_delay:+.1f} minutes\n"
        f"Today's prediction is {abs(delta):.1f} minutes {direction} usual.\n\n"
        "Explain this to the commuter, including whether today is unusually bad or normal."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=300,
        ),
    )
    return response.text.strip()


def chat_agent(initial_context: PredictionContext | None = None) -> ChatSession:
    """
    Start a stateful multi-turn chat session about route delays.

    Parameters
    ----------
    initial_context : optional PredictionContext to seed the first message.

    Returns
    -------
    ChatSession — call .send(message) to continue the conversation.

    Example
    -------
    >>> session = chat_agent(ctx)
    >>> print(session.send("Why is my bus late?"))
    >>> print(session.send("What about tomorrow morning?"))
    """
    client = _build_client()
    session = ChatSession(client=client)

    if initial_context:
        seed_message = (
            "Here is the current prediction context for our conversation:\n\n"
            f"{initial_context.to_prompt_context()}\n\n"
            "Please acknowledge this and be ready to answer commuter questions."
        )
        ack = session.send(seed_message)
        log.info("Chat session seeded. Model ack: %s", ack[:80])

    return session


def batch_explain(
    contexts: list[PredictionContext],
) -> pd.DataFrame:
    """
    Generate explanations for multiple predictions.

    Returns a DataFrame with columns:
      route, predicted_delay_minutes, explanation, top_factors_json
    """
    client = _build_client()
    rows = []

    for i, ctx in enumerate(contexts):
        log.info("Batch explain %d/%d — route %s", i + 1, len(contexts), ctx.route)
        prompt = (
            "Explain this bus delay prediction briefly (2 sentences max):\n\n"
            f"{ctx.to_prompt_context()}"
        )
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=150,
                ),
            )
            explanation = response.text.strip()
        except Exception as exc:
            log.warning("Gemini error for route %s: %s", ctx.route, exc)
            explanation = "Explanation unavailable."

        rows.append(
            {
                "route": ctx.route,
                "predicted_delay_minutes": round(ctx.predicted_delay_minutes, 2),
                "explanation": explanation,
                "top_factors_json": json.dumps(ctx.top_factors),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience builder  (used by app.py)
# ---------------------------------------------------------------------------

def build_context(
    route: str,
    predicted_delay: float,
    feature_names: list[str],
    feature_values: dict[str, float],
    shap_values: np.ndarray,
    top_n: int = 5,
) -> PredictionContext:
    """Thin constructor — avoids importing the dataclass in app.py."""
    return PredictionContext(
        route=route,
        predicted_delay_minutes=predicted_delay,
        feature_values=feature_values,
        shap_values=shap_values,
        feature_names=feature_names,
        top_n=top_n,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Minimal smoke-test with fake data so the file can be run without a
    # trained model. Swap in real values from train_model.predict_single().
    fake_features = [
        "hour", "is_rush_hour", "is_weekend", "rain_1h_mm",
        "temp_c", "wind_speed_ms", "route_delay_roll5", "stops_away",
    ]
    fake_values = {
        "hour": 8, "is_rush_hour": 1, "is_weekend": 0,
        "rain_1h_mm": 3.2, "temp_c": 4.5, "wind_speed_ms": 6.1,
        "route_delay_roll5": 4.8, "stops_away": 3,
    }
    fake_shap = np.array([0.3, 2.1, 0.0, 1.8, -0.4, 0.9, 3.2, 0.5])

    ctx = build_context(
        route="M15",
        predicted_delay=7.4,
        feature_names=fake_features,
        feature_values=fake_values,
        shap_values=fake_shap,
    )

    print("=== Prediction Context ===")
    print(ctx.to_prompt_context())
    print()

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set — skipping live API call.")
        print("Set it in .env and re-run to see a live explanation.")
        sys.exit(0)

    print("=== One-shot explanation ===")
    print(explain_prediction(ctx))
    print()

    print("=== Multi-turn chat ===")
    session = chat_agent(ctx)
    for question in [
        "Should I leave earlier tomorrow to avoid this delay?",
        "Is rain the main reason for the delay?",
    ]:
        print(f"You: {question}")
        print(f"RouteBot: {session.send(question)}")
        print()
