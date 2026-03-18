"""
feature_engineering.py

Joins the three raw CSVs and engineers a model-ready feature set.

Inputs  (latest file by mtime in data/raw/):
  mta_speeds_raw_*.csv    — segment-level travel times  (kufs-yh3x)
  mta_journey_raw_*.csv   — monthly route delay metrics (8mkn-d32t)
  weather_raw_*.csv       — OWM hourly / 3-hour weather (optional)

Output:
  data/processed/features.parquet

Target variable:
  delay_score  — additional_travel_time (minutes over schedule) joined from
                 the journey metrics onto each speeds row by route + period

Run standalone:
    python src/feature_engineering.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _latest_csv(directory: Path, glob: str) -> Path:
    files = sorted(directory.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob}' in {directory}. Run data_collection.py first."
        )
    log.info("Loading %s", files[0].name)
    return files[0]


def load_speeds(path: Path | None = None) -> pd.DataFrame:
    path = path or _latest_csv(RAW_DIR, "mta_speeds_raw_*.csv")
    df = pd.read_csv(path, low_memory=False)
    log.info("Segment Speeds raw shape: %s", df.shape)
    return df


def load_journey(path: Path | None = None) -> pd.DataFrame:
    path = path or _latest_csv(RAW_DIR, "mta_journey_raw_*.csv")
    df = pd.read_csv(path, low_memory=False)
    log.info("Journey Metrics raw shape: %s", df.shape)
    return df


def load_weather(path: Path | None = None) -> pd.DataFrame | None:
    """Returns None (silently) if no weather file exists yet."""
    try:
        path = path or _latest_csv(RAW_DIR, "weather_raw_*.csv")
    except FileNotFoundError:
        log.warning("No weather file found — weather features will be skipped.")
        return None
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        log.warning("Weather file is empty — weather features will be skipped.")
        return None
    log.info("Weather raw shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Segment Speeds feature engineering
# ---------------------------------------------------------------------------

def engineer_speeds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric casts
    for col in ("average_travel_time", "average_road_speed", "bus_trip_count",
                "road_distance", "hour_of_day", "year", "month"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["average_travel_time", "hour_of_day"])

    # ── Temporal features ──────────────────────────────────────────────────
    # day_of_week from Socrata is a string like "Monday" → encode to int
    dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    if "day_of_week" in df.columns:
        df["dow_int"] = df["day_of_week"].map(dow_map).fillna(
            pd.to_numeric(df["day_of_week"], errors="coerce")
        )
    else:
        df["dow_int"] = 0

    df["is_weekend"] = (df["dow_int"] >= 5).astype(int)
    df["is_rush_hour"] = (
        df["hour_of_day"].between(7, 9) | df["hour_of_day"].between(16, 19)
    ).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow_int"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow_int"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Route-level speed stats ────────────────────────────────────────────
    # Baseline travel time per route (median across all hours)
    route_median = df.groupby("route_id")["average_travel_time"].transform("median")
    df["travel_time_vs_median"] = df["average_travel_time"] - route_median

    # Speed ratio: how much slower than that route's fastest hour
    route_max_speed = df.groupby("route_id")["average_road_speed"].transform("max")
    df["speed_ratio"] = df["average_road_speed"] / route_max_speed.replace(0, np.nan)

    # Trip volume normalised per route
    route_max_trips = df.groupby("route_id")["bus_trip_count"].transform("max")
    df["trip_volume_norm"] = df["bus_trip_count"] / route_max_trips.replace(0, np.nan)

    # ── Categoricals ──────────────────────────────────────────────────────
    if "direction" in df.columns:
        df["direction_code"] = df["direction"].astype("category").cat.codes

    if "route_type" in df.columns:
        df["route_type_code"] = df["route_type"].astype("category").cat.codes

    if "borough" in df.columns:
        df["borough_code"] = df["borough"].astype("category").cat.codes

    # Peak period flag derived from hour (used as join key with journey metrics)
    df["period"] = df["hour_of_day"].apply(
        lambda h: "peak" if (7 <= h <= 9 or 16 <= h <= 19) else "off_peak"
    )

    log.info("Speeds features shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Journey Metrics feature engineering
# ---------------------------------------------------------------------------

def engineer_journey(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("additional_travel_time", "additional_bus_stop_time",
                "customer_journey_time", "number_of_customers"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse month → year + month_num for joining
    df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
    df["year"]  = df["month_dt"].dt.year
    df["month_num"] = df["month_dt"].dt.month

    # Normalise period label to match speeds df
    if "period" in df.columns:
        df["period"] = df["period"].str.lower().str.replace(" ", "_")

    # Route-level rolling average delay (last 3 months proxy via rank)
    df = df.sort_values(["route_id", "month_dt"])
    df["route_delay_roll3"] = (
        df.groupby("route_id")["additional_travel_time"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    log.info("Journey features shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Weather feature engineering
# ---------------------------------------------------------------------------

def engineer_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"])

    df["hour_of_day"] = df["datetime_utc"].dt.hour
    df["month_num"]   = df["datetime_utc"].dt.month

    df["is_raining"] = (df["rain_1h_mm"].fillna(0) > 0).astype(int)
    df["is_snowing"] = (df["snow_1h_mm"].fillna(0) > 0).astype(int)

    precip = df["rain_1h_mm"].fillna(0) + df["snow_1h_mm"].fillna(0)
    df["precip_intensity"] = pd.cut(
        precip, bins=[-0.001, 0, 2.5, 7.6, np.inf], labels=[0, 1, 2, 3]
    ).astype(float)

    df["apparent_temp_delta"] = df["feels_like_c"] - df["temp_c"]
    vis_km = df["visibility_m"].fillna(10_000) / 1000
    df["visibility_bucket"] = pd.cut(
        vis_km, bins=[-0.001, 1, 5, np.inf], labels=[0, 1, 2]
    ).astype(float)

    if "weather_main" in df.columns:
        dummies = pd.get_dummies(df["weather_main"].fillna("Unknown"), prefix="wx", dtype=int)
        df = pd.concat([df, dummies], axis=1)

    # Aggregate to hour-of-day + month_num for joining (mean across available days)
    agg_cols = ["hour_of_day", "month_num", "temp_c", "humidity_pct", "pressure_hpa",
                "wind_speed_ms", "clouds_pct", "rain_1h_mm", "snow_1h_mm", "pop",
                "is_raining", "is_snowing", "precip_intensity", "apparent_temp_delta",
                "visibility_bucket"] + [c for c in df.columns if c.startswith("wx_")]

    agg_cols = [c for c in agg_cols if c in df.columns]
    weather_agg = (
        df[agg_cols]
        .groupby(["hour_of_day", "month_num"])
        .mean()
        .reset_index()
    )

    log.info("Weather features shape (aggregated): %s", weather_agg.shape)
    return weather_agg


# ---------------------------------------------------------------------------
# Join + finalise
# ---------------------------------------------------------------------------

def join_and_finalise(
    speeds_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    weather_df: pd.DataFrame | None,
) -> pd.DataFrame:

    # ── Join journey metrics onto speeds (route_id + period + year + month) ─
    journey_cols = ["route_id", "year", "month_num", "period",
                    "additional_travel_time", "additional_bus_stop_time",
                    "customer_journey_time", "route_delay_roll3"]
    journey_cols = [c for c in journey_cols if c in journey_df.columns]

    # Rename month_num in journey to avoid conflict
    journey_slim = journey_df[journey_cols].copy()

    df = speeds_df.merge(
        journey_slim,
        on=[c for c in ["route_id", "year", "month_num", "period"]
            if c in speeds_df.columns and c in journey_slim.columns],
        how="left",
    )
    log.info("After speeds+journey join: %s", df.shape)

    # ── Join weather by hour_of_day + month ──────────────────────────────
    if weather_df is not None and not weather_df.empty:
        df = df.merge(
            weather_df,
            on=[c for c in ["hour_of_day", "month_num"]
                if c in df.columns and c in weather_df.columns],
            how="left",
        )
        log.info("After weather join: %s", df.shape)
    else:
        log.warning("Skipping weather join — no weather data available.")

    # ── Target variable ────────────────────────────────────────────────────
    # PRIMARY target: travel_time_vs_median
    #   = how many minutes a segment's travel time deviates from that route's
    #     typical travel time. This is segment-level (122K unique values,
    #     std ~6 min, range −14 to +74 min) and captures real operational delay.
    #
    # WHY NOT additional_travel_time from journey metrics:
    #   That field is a monthly route-level aggregate → only 52 unique values
    #   compressed into 0–2.6 min → near-zero variance → model always predicts
    #   ~0.6 min regardless of inputs (the bug that caused RMSE=0, R²=1).
    #
    # additional_travel_time is kept as a FEATURE (contextual monthly baseline)
    # but must never be the target.
    if "travel_time_vs_median" in df.columns and df["travel_time_vs_median"].notna().any():
        df["delay_minutes"] = df["travel_time_vs_median"]
        log.info(
            "Target: travel_time_vs_median → delay_minutes  "
            "(%.1f unique values, mean=%.2f, std=%.2f)",
            df["delay_minutes"].nunique(),
            df["delay_minutes"].mean(),
            df["delay_minutes"].std(),
        )
    else:
        raise ValueError(
            "travel_time_vs_median not found. "
            "Check that engineer_speeds() ran successfully."
        )

    # Clip extreme outliers (>30 min deviation = data artefact, not real delay)
    df["delay_minutes"] = df["delay_minutes"].clip(-15, 30)

    # ── Drop leakage + non-predictive columns ─────────────────────────────
    # LEAKAGE RULE: only keep features that a commuter/scheduler would know
    # BEFORE the trip departs. Anything measured DURING the trip leaks the answer.
    #
    # Removed as leakage:
    #   average_travel_time  — this IS the delay signal; target = avg_tt − route_median
    #   road_distance        — segment distance is fixed per route, collinear with avg_tt
    #   speed_ratio          — derived from average_road_speed which also correlates
    #   average_road_speed   — real-time measurement during the trip (not pre-trip)
    #   trip_volume_norm     — also measured during the trip
    drop_cols = [
        # identity / string columns
        "route_id", "day_of_week", "direction", "route_type", "borough",
        "period", "timepoint_stop_name", "next_timepoint_stop_name",
        "timepoint_stop_id", "next_timepoint_stop_id",
        "timepoint_stop_georeference", "next_timepoint_stop_georeference",
        "timepoint_stop_latitude", "timepoint_stop_longitude",
        "next_timepoint_stop_latitude", "next_timepoint_stop_longitude",
        "timestamp", "month", "month_dt",
        # target (already promoted)
        "travel_time_vs_median",
        # leakage — measured during the trip, not known before departure
        "average_travel_time",
        "average_road_speed",
        "road_distance",
        "speed_ratio",
        "trip_volume_norm",
        # monthly aggregate — causes near-zero variance target leakage
        "additional_travel_time",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Keep only numeric columns
    df = df.select_dtypes(include="number")

    # Drop rows without a target
    before = len(df)
    df = df.dropna(subset=["delay_minutes"])
    log.info("Dropped %d rows with null delay_minutes", before - len(df))

    # Fill remaining NaNs with column median
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    log.info("Final feature set: %d rows × %d columns", *df.shape)
    log.info("Columns: %s", df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(
    speeds_path: Path | None = None,
    journey_path: Path | None = None,
    weather_path: Path | None = None,
) -> Path:
    log.info("=== Feature Engineering Start ===")

    speeds_raw  = load_speeds(speeds_path)
    journey_raw = load_journey(journey_path)
    weather_raw = load_weather(weather_path)

    speeds_feat  = engineer_speeds(speeds_raw)
    journey_feat = engineer_journey(journey_raw)
    weather_feat = engineer_weather(weather_raw) if weather_raw is not None else None

    features = join_and_finalise(speeds_feat, journey_feat, weather_feat)

    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)
    log.info("=== Feature Engineering Complete ===")
    return out_path


if __name__ == "__main__":
    run()
