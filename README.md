# 🚌 Dynamic Route Delay Predictor + LLM Explainer

Predict how late an NYC MTA bus will run — and get a plain-English explanation of *why* — powered by XGBoost, SHAP, and the Gemini AI.

---

## What this project does

New York City runs thousands of bus trips every day across five boroughs. Some routes run on time. Others fall behind depending on the hour, day, traffic volume, weather, and the route's recent delay history.

This project builds an end-to-end machine learning system that:

1. **Pulls real data** from the NYC MTA open data portal (segment-level travel times + monthly journey metrics) and OpenWeatherMap
2. **Engineers features** — time of day, rush hour flags, route type, borough, weather conditions, rolling delay history, and more
3. **Trains an XGBoost model** to predict how many minutes a route will deviate from its typical travel time
4. **Explains every prediction** using SHAP values so you can see exactly which factors drove the result
5. **Wraps it in a Gemini-powered AI agent** that turns the numbers into a clear, human-readable sentence
6. **Serves everything** through an interactive Streamlit dashboard

---

## How it works — step by step

```
Data Collection  →  Feature Engineering  →  Model Training  →  Dashboard
(MTA + Weather)      (130K+ data points)     (XGBoost + SHAP)   (Streamlit + Gemini)
```

**Data Collection (`data_collection.py`)**
Pulls NYC MTA Bus Time segment speed data and monthly journey metrics from the NY State Open Data portal via the Socrata API. Also fetches current weather for NYC from OpenWeatherMap. Everything lands in `data/raw/` as CSVs.

**Feature Engineering (`feature_engineering.py`)**
Joins the three raw data sources and builds the model-ready feature set. Key features include: hour of day, day of week, rush hour flag, cyclical time encodings, borough, route type, travel direction, active trip count, 3-month rolling delay average, customer journey time, and weather conditions (temperature, rain, wind, cloud cover).

The **target variable** is `travel_time_vs_median` — how many minutes a segment's travel time deviates from that route's historical median. This gives 122,000+ unique values ranging from −14 to +74 minutes, capturing real operational delays rather than smoothed monthly averages.

**Model Training (`train_model.py`)**
Trains an XGBoost regressor on 608K rows using a time-based train/test split (earlier months for training, later months for testing — no data leakage). SHAP values are computed for every prediction, identifying which features pushed the delay up or down and by how much.

| Metric | Value |
|--------|-------|
| MAE    | 3.56 min |
| RMSE   | 5.03 min |
| R²     | 0.31 |
| Training rows | 608K |

**AI Explainer (`explainer_agent.py`)**
Uses the Gemini API to convert the raw numbers (predicted delay + top SHAP features) into a natural language explanation. For example: *"Route B46 is running 2.4 min behind — evening rush combined with high trip volume are the main drivers."*

**Dashboard (`app.py`)**
A four-page Streamlit app:
- **Predict & Explain** — pick a route, time, weather, and conditions → get a delay prediction, SHAP waterfall chart, and Gemini explanation
- **SHAP Feature Impact** — global feature importance across the full dataset
- **Model Performance** — evaluation metrics and prediction distribution
- **Chat with RouteBot** — multi-turn conversation with a Gemini agent about NYC bus delays

---

## Top predictive features (by SHAP importance)

| Feature | What it captures |
|---|---|
| `direction_code` | Inbound vs outbound direction |
| `route_type_code` | Local / SBS / Limited / School |
| `hour_of_day` | Time of day (rush vs off-peak) |
| `bus_trip_count` | How many buses are running right now |
| `route_delay_roll3` | This route's delay trend over the last 3 months |
| `borough_code` | Manhattan / Brooklyn / Queens |
| `customer_journey_time` | End-to-end journey duration |
| Weather features | Rain, temperature, wind speed |

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | NYC MTA Open Data (Socrata API), OpenWeatherMap API |
| ML | XGBoost, Scikit-learn, SHAP |
| AI | Google Gemini API (`gemini-2.0-flash`) |
| Dashboard | Streamlit, Plotly |
| Language | Python 3.12 |

---

## Project structure

```
route-delay-predictor/
├── data/
│   ├── raw/          # MTA + weather CSVs (auto-generated)
│   └── processed/    # Engineered feature parquet
├── models/           # Trained XGBoost model + SHAP artefacts
├── notebooks/        # EDA
├── src/
│   ├── data_collection.py     # MTA + OWM data pull
│   ├── feature_engineering.py # Feature building + target construction
│   ├── train_model.py         # XGBoost training + SHAP + eval
│   ├── explainer_agent.py     # Gemini integration
│   └── app.py                 # Streamlit dashboard
└── requirements.txt
```

---

## Running locally

```bash
# 1. Clone and set up environment
git clone https://github.com/your-username/route-delay-predictor.git
cd route-delay-predictor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys
echo "OPENWEATHER_API_KEY=your_key" >> .env
echo "GEMINI_API_KEY=your_key"      >> .env

# 3. Run the pipeline
python src/data_collection.py
python src/feature_engineering.py
python src/train_model.py

# 4. Launch the dashboard
streamlit run src/app.py
```

---

## Data sources

- **NYC MTA Segment Speeds** — [data.ny.gov](https://data.ny.gov/Transportation/MTA-Segment-Level-Speeds/kv7t-n4uh) (Socrata dataset `kv7t-n4uh`)
- **NYC MTA Journey Metrics** — [data.ny.gov](https://data.ny.gov/Transportation/MTA-Journey-Time-Performance/8mkn-d32t) (Socrata dataset `8mkn-d32t`)
- **OpenWeatherMap** — [openweathermap.org](https://openweathermap.org/api) (free tier, 5-day forecast)
