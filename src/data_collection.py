"""
data_collection.py

Pulls three raw datasets and saves them to data/raw/:
  1. MTA Bus Route Segment Speeds  (data.ny.gov · kufs-yh3x)
     → hour_of_day, day_of_week, route, direction, average_travel_time
  2. MTA Bus Customer Journey Metrics  (data.ny.gov · 8mkn-d32t)
     → monthly additional_travel_time + customer_journey_time_performance per route
  3. NYC weather  — OpenWeatherMap (forecast or timemachine depending on plan)

Environment variables consumed:
  OPENWEATHER_API_KEY          — required for weather
  OPENWEATHER_HISTORY_DAYS     — days of hourly history to pull (default 5)
  MTA_APP_TOKEN                — optional Socrata app token (raises rate limits)
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NYC_LAT = 40.7128
NYC_LON = -74.0060

# Verified Socrata endpoints on data.ny.gov
# Route Segment Speeds — hour/dow granularity, average travel time per stop pair
MTA_SPEEDS_URL = "https://data.ny.gov/resource/kufs-yh3x.json"

# Customer Journey Metrics — monthly route-level additional travel time
MTA_JOURNEY_URL = "https://data.ny.gov/resource/8mkn-d32t.json"


# ---------------------------------------------------------------------------
# Shared Socrata helper
# ---------------------------------------------------------------------------

def _socrata_headers() -> dict:
    raw = os.getenv("MTA_APP_TOKEN", "")
    token = raw.split("#")[0].strip()
    return {"X-App-Token": token} if token else {}


def _socrata_fetch(
    url: str,
    where: str | None = None,
    select: str | None = None,
    order: str | None = None,
    limit: int = 50_000,
) -> pd.DataFrame:
    """
    Generic paginated Socrata SODA fetcher.
    Returns a concatenated DataFrame of all pages.
    """
    headers = _socrata_headers()
    all_frames = []
    offset = 0
    page = 1

    while True:
        params: dict = {"$limit": limit, "$offset": offset}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order

        log.info("  [%s] page %d (offset=%d)…", url.split("/")[-1], page, offset)
        resp = requests.get(url, headers=headers, params=params, timeout=60)

        if not resp.ok:
            log.error("  Socrata %s → HTTP %d: %s", url, resp.status_code, resp.text[:300])
            resp.raise_for_status()

        records = resp.json()
        if not records:
            break

        all_frames.append(pd.DataFrame(records))
        log.info("  Got %d records", len(records))

        if len(records) < limit:
            break

        offset += limit
        page += 1
        time.sleep(0.4)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Dataset 1 — MTA Bus Route Segment Speeds  (kufs-yh3x)
# ---------------------------------------------------------------------------

def fetch_mta_segment_speeds(
    routes: list[str] | None = None,
    start_year: int = 2025,
) -> pd.DataFrame:
    """
    Pull MTA Bus Route Segment Speeds from data.ny.gov.

    Key columns returned:
        year, month, day_of_week, hour_of_day,
        route_id, direction, borough, route_type,
        average_travel_time, average_road_speed, bus_trip_count,
        timepoint_stop_name, next_timepoint_stop_name, road_distance

    Parameters
    ----------
    routes      : e.g. ["M15", "B46"] — None pulls all routes.
    start_year  : pull records from this year onwards (dataset starts 2025).
    """
    log.info("Fetching MTA Segment Speeds (kufs-yh3x) — routes=%s", routes)

    where_parts = [f"year >= '{start_year}'"]
    if routes:
        quoted = ", ".join(f"'{r}'" for r in routes)
        where_parts.append(f"route_id IN ({quoted})")

    select_cols = (
        "year,month,day_of_week,hour_of_day,"
        "route_id,direction,borough,route_type,"
        "average_travel_time,average_road_speed,bus_trip_count,"
        "timepoint_stop_name,next_timepoint_stop_name,road_distance"
    )

    df = _socrata_fetch(
        url=MTA_SPEEDS_URL,
        where=" AND ".join(where_parts),
        select=select_cols,
        order="year DESC, month DESC",
    )

    log.info("Segment Speeds pull complete — %d rows", len(df))
    return df


def save_segment_speeds(df: pd.DataFrame) -> Path:
    if df.empty:
        log.warning("Segment Speeds DataFrame is empty — skipping save.")
        return RAW_DIR / "mta_speeds_raw_EMPTY.csv"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"mta_speeds_raw_{ts}.csv"
    df.to_csv(out, index=False)
    log.info("Segment Speeds saved → %s (%d rows)", out, len(df))
    return out


# ---------------------------------------------------------------------------
# Dataset 2 — MTA Customer Journey Metrics  (8mkn-d32t)
# ---------------------------------------------------------------------------

def fetch_mta_journey_metrics(
    routes: list[str] | None = None,
    start_year: int = 2022,
) -> pd.DataFrame:
    """
    Pull MTA Bus Customer Journey-Focused Metrics.

    Key columns:
        month, borough, trip_type, route_id, period,
        additional_travel_time,        ← avg extra minutes beyond schedule
        customer_journey_time_performance,  ← % journeys within 5 min of schedule
        number_of_customers
    """
    log.info("Fetching MTA Customer Journey Metrics (8mkn-d32t) — routes=%s", routes)

    cutoff = f"{start_year}-01-01T00:00:00"
    where_parts = [f"month >= '{cutoff}'"]
    if routes:
        quoted = ", ".join(f"'{r}'" for r in routes)
        where_parts.append(f"route_id IN ({quoted})")

    df = _socrata_fetch(
        url=MTA_JOURNEY_URL,
        where=" AND ".join(where_parts),
        order="month DESC",
    )

    log.info("Journey Metrics pull complete — %d rows", len(df))
    return df


def save_journey_metrics(df: pd.DataFrame) -> Path:
    if df.empty:
        log.warning("Journey Metrics DataFrame is empty — skipping save.")
        return RAW_DIR / "mta_journey_raw_EMPTY.csv"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"mta_journey_raw_{ts}.csv"
    df.to_csv(out, index=False)
    log.info("Journey Metrics saved → %s (%d rows)", out, len(df))
    return out


# ---------------------------------------------------------------------------
# Dataset 3 — OpenWeatherMap weather
# ---------------------------------------------------------------------------

OWM_TIMEMACHINE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def _parse_hourly_block(hourly_data: list[dict]) -> list[dict]:
    rows = []
    for h in hourly_data:
        rows.append({
            "dt": h.get("dt"),
            "datetime_utc": datetime.fromtimestamp(h["dt"], tz=timezone.utc).replace(tzinfo=None).isoformat(),
            "temp_c": h.get("temp"),
            "feels_like_c": h.get("feels_like"),
            "humidity_pct": h.get("humidity"),
            "pressure_hpa": h.get("pressure"),
            "wind_speed_ms": h.get("wind_speed"),
            "wind_deg": h.get("wind_deg"),
            "visibility_m": h.get("visibility"),
            "uvi": h.get("uvi"),
            "clouds_pct": h.get("clouds"),
            "weather_main": h.get("weather", [{}])[0].get("main"),
            "weather_desc": h.get("weather", [{}])[0].get("description"),
            "rain_1h_mm": h.get("rain", {}).get("1h", 0.0),
            "snow_1h_mm": h.get("snow", {}).get("1h", 0.0),
            "pop": h.get("pop", 0.0),
        })
    return rows


def fetch_owm_history(days_back: int | None = None) -> pd.DataFrame:
    """
    Pull hourly weather for NYC.
    Tries OWM One-Call 3.0 timemachine (paid); falls back to /forecast (free).
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENWEATHER_API_KEY is not set.")

    if days_back is None:
        days_back = int(os.getenv("OPENWEATHER_HISTORY_DAYS", 5))

    log.info("Fetching OWM weather — %d days back", days_back)
    all_rows: list[dict] = []

    for day_offset in range(1, days_back + 1):
        target = datetime.now(timezone.utc) - timedelta(days=day_offset)
        log.info("  Weather for %s…", target.strftime("%Y-%m-%d"))

        params = {
            "lat": NYC_LAT, "lon": NYC_LON,
            "dt": int(target.timestamp()),
            "appid": api_key, "units": "metric",
        }
        resp = requests.get(OWM_TIMEMACHINE_URL, params=params, timeout=30)

        if resp.status_code == 401:
            log.warning("  OWM timemachine requires a paid plan — using /forecast fallback.")
            return _fetch_owm_forecast_fallback(api_key)

        if not resp.ok:
            log.warning("  OWM timemachine failed (%d) — using /forecast fallback.", resp.status_code)
            return _fetch_owm_forecast_fallback(api_key)

        payload = resp.json()
        hourly = payload.get("data") or payload.get("hourly", [])
        rows = _parse_hourly_block(hourly)
        all_rows.extend(rows)
        log.info("  Got %d rows.", len(rows))
        time.sleep(0.3)

    if not all_rows:
        log.warning("OWM returned 0 rows — using /forecast fallback.")
        return _fetch_owm_forecast_fallback(api_key)

    df = pd.DataFrame(all_rows).sort_values("dt").reset_index(drop=True)
    log.info("OWM pull complete — %d hourly rows", len(df))
    return df


def _fetch_owm_forecast_fallback(api_key: str) -> pd.DataFrame:
    log.info("  OWM /forecast fallback (5-day / 3-hour, free tier)…")
    resp = requests.get(
        OWM_FORECAST_URL,
        params={"lat": NYC_LAT, "lon": NYC_LON, "appid": api_key, "units": "metric", "cnt": 40},
        timeout=30,
    )
    resp.raise_for_status()

    rows = []
    for item in resp.json().get("list", []):
        rows.append({
            "dt": item["dt"],
            "datetime_utc": datetime.fromtimestamp(item["dt"], tz=timezone.utc).replace(tzinfo=None).isoformat(),
            "temp_c": item["main"]["temp"],
            "feels_like_c": item["main"]["feels_like"],
            "humidity_pct": item["main"]["humidity"],
            "pressure_hpa": item["main"]["pressure"],
            "wind_speed_ms": item["wind"]["speed"],
            "wind_deg": item["wind"].get("deg"),
            "visibility_m": item.get("visibility"),
            "uvi": None,
            "clouds_pct": item["clouds"]["all"],
            "weather_main": item["weather"][0]["main"],
            "weather_desc": item["weather"][0]["description"],
            "rain_1h_mm": item.get("rain", {}).get("3h", 0.0),
            "snow_1h_mm": item.get("snow", {}).get("3h", 0.0),
            "pop": item.get("pop", 0.0),
        })

    df = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
    log.info("  Fallback got %d rows.", len(df))
    return df


def save_weather_data(df: pd.DataFrame) -> Path:
    if df.empty:
        log.warning("Weather DataFrame is empty — skipping save.")
        return RAW_DIR / "weather_raw_EMPTY.csv"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RAW_DIR / f"weather_raw_{ts}.csv"
    df.to_csv(out, index=False)
    log.info("Weather saved → %s (%d rows)", out, len(df))
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(
    routes: list[str] | None = None,
    start_year: int = 2025,
    weather_days_back: int | None = None,
) -> dict[str, Path]:
    """
    Collect all three datasets and return paths to the saved CSVs.

    Parameters
    ----------
    routes          : MTA route IDs, e.g. ["M15", "B46"]. None = all routes.
    start_year      : Pull MTA records from this year onwards.
    weather_days_back : Days of OWM history (default: OPENWEATHER_HISTORY_DAYS).
    """
    log.info("=== Data Collection Start ===")

    speeds_df = fetch_mta_segment_speeds(routes=routes, start_year=start_year)
    speeds_path = save_segment_speeds(speeds_df)

    journey_df = fetch_mta_journey_metrics(routes=routes)
    journey_path = save_journey_metrics(journey_df)

    try:
        weather_df = fetch_owm_history(days_back=weather_days_back)
    except Exception as exc:
        log.warning("Weather fetch failed (%s) — skipping. Re-run once your OWM key is active.", exc)
        weather_df = pd.DataFrame()
    weather_path = save_weather_data(weather_df)

    log.info("=== Data Collection Complete ===")
    log.info("  Segment Speeds  → %s", speeds_path)
    log.info("  Journey Metrics → %s", journey_path)
    log.info("  Weather         → %s", weather_path)

    return {"speeds": speeds_path, "journey": journey_path, "weather": weather_path}


if __name__ == "__main__":
    run(
        routes=["M15", "M15+", "B46", "Q58", "Bx12"],
        start_year=2025,
    )
