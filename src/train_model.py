"""
train_model.py

Trains an XGBoost regressor to predict bus delay (minutes), evaluates it,
computes SHAP values, and saves all artefacts to models/.

Input  : data/processed/features.parquet
Outputs: models/xgb_delay_model.json      — trained XGBoost model
         models/shap_values.npy           — SHAP values array
         models/shap_expected_value.npy   — SHAP base value
         models/feature_names.json        — ordered feature list
         models/eval_metrics.json         — test-set metrics

Run standalone:
    python src/train_model.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "delay_minutes"
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading + splitting
# ---------------------------------------------------------------------------

def load_features(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found at {path}. "
            "Run feature_engineering.py first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded features: %d rows × %d cols", *df.shape)
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Return X_train, X_test, y_train, y_test and the feature name list.

    Uses a TIME-BASED split: the most recent 20% of months are held out as
    the test set. This prevents data leakage where the model memorises
    monthly route averages seen during training (which caused RMSE=0 / R²=1
    with a random split, because the same month appears in both train & test).
    """
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            "Check feature_engineering.py output."
        )

    df = df.select_dtypes(include="number").copy()

    # ── Time-based split on month_num + year ──────────────────────────────
    # Build a single sortable period key so we can cut at the 80th percentile
    if "year" in df.columns and "month_num" in df.columns:
        df["_period"] = df["year"] * 100 + df["month_num"]   # e.g. 202601
        cutoff = df["_period"].quantile(TEST_SIZE, interpolation="lower")
        # Resolve ties: take the unique period closest to the 80th pct
        unique_periods = sorted(df["_period"].unique())
        cutoff_period  = unique_periods[int(len(unique_periods) * (1 - TEST_SIZE))]

        train_mask = df["_period"] <  cutoff_period
        test_mask  = df["_period"] >= cutoff_period

        log.info(
            "Time-based split — train periods: <%d | test periods: >=%d",
            cutoff_period, cutoff_period,
        )
        log.info(
            "Train: %d rows | Test: %d rows",
            train_mask.sum(), test_mask.sum(),
        )

        # Guard: fall back to random split if split is degenerate
        if train_mask.sum() < 1000 or test_mask.sum() < 500:
            log.warning(
                "Time-based split produced too few samples — falling back to random split."
            )
            train_mask = None

        df = df.drop(columns=["_period"])

    else:
        log.warning("year/month_num columns missing — using random split.")
        train_mask = None

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = X.columns.tolist()
    log.info("Features used for training: %d", len(feature_names))

    if train_mask is not None:
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",          # fast histogram method
    "eval_metric": "mae",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,               # L1 regularisation
    "reg_lambda": 1.0,              # L2 regularisation
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

HYPERPARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [300, 500, 700],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [3, 5, 10],
    "reg_alpha": [0, 0.1, 0.5],
}


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparams: bool = False,
    n_iter: int = 20,
) -> xgb.XGBRegressor:
    """
    Train XGBoost with early stopping.

    Parameters
    ----------
    tune_hyperparams : run RandomizedSearchCV before final training.
    n_iter           : number of random HP combinations to try.
    """
    if tune_hyperparams:
        log.info("Running RandomizedSearchCV (%d iterations)…", n_iter)
        base = xgb.XGBRegressor(**{**BASE_PARAMS, "n_estimators": 300})
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=HYPERPARAM_GRID,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=3,
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_params = {**BASE_PARAMS, **search.best_params_}
        log.info("Best HP: %s", search.best_params_)
    else:
        best_params = BASE_PARAMS.copy()

    log.info("Training final XGBoost model…")
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    log.info("Training complete.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    preds = model.predict(X_test)
    errors = np.abs(y_test.values - preds)

    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2   = r2_score(y_test, preds)

    # MAPE is undefined / explodes when targets are near zero (this dataset has
    # many segments with travel-time deviation ≈ 0). Replace with intuitive
    # precision-at-threshold metrics instead.
    within_2min = float(np.mean(errors <= 2.0) * 100)   # % predictions within 2 min
    within_5min = float(np.mean(errors <= 5.0) * 100)   # % predictions within 5 min

    # Median Absolute Error — robust to the heavy tails in this distribution
    median_ae = float(np.median(errors))

    metrics = {
        "mae_minutes":      round(mae, 4),
        "rmse_minutes":     round(rmse, 4),
        "r2_score":         round(r2, 4),
        "median_ae_minutes": round(median_ae, 4),
        "within_2min_pct":  round(within_2min, 1),
        "within_5min_pct":  round(within_5min, 1),
        "n_test_samples":   len(y_test),
    }

    log.info("── Evaluation Results ──────────────────────")
    for k, v in metrics.items():
        log.info("  %-22s %s", k, v)
    log.info("────────────────────────────────────────────")
    return metrics


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap(
    model: xgb.XGBRegressor,
    X_sample: pd.DataFrame,
    max_rows: int = 2000,
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    max_rows : cap on sample size to keep memory/time reasonable.

    Returns
    -------
    (shap_values array, expected_value scalar)
    """
    if len(X_sample) > max_rows:
        X_sample = X_sample.sample(max_rows, random_state=RANDOM_STATE)

    log.info("Computing SHAP values on %d samples…", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    expected_value = float(explainer.expected_value)

    log.info("SHAP expected value (base): %.4f minutes", expected_value)

    # Log top-5 most impactful features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:5]
    feat_names = X_sample.columns.tolist()
    log.info("Top-5 features by mean |SHAP|:")
    for rank, idx in enumerate(top_idx, 1):
        log.info("  %d. %-30s %.4f", rank, feat_names[idx], mean_abs_shap[idx])

    return shap_values, expected_value


# ---------------------------------------------------------------------------
# Artefact saving
# ---------------------------------------------------------------------------

def save_artefacts(
    model: xgb.XGBRegressor,
    feature_names: list[str],
    metrics: dict,
    shap_values: np.ndarray,
    expected_value: float,
) -> None:
    model_path = MODELS_DIR / "xgb_delay_model.json"
    model.save_model(model_path)
    log.info("Model saved → %s", model_path)

    np.save(MODELS_DIR / "shap_values.npy", shap_values)
    np.save(MODELS_DIR / "shap_expected_value.npy", np.array([expected_value]))
    log.info("SHAP artefacts saved → models/shap_values.npy")

    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info("Feature names saved → models/feature_names.json")

    with open(MODELS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Eval metrics saved → models/eval_metrics.json")


# ---------------------------------------------------------------------------
# Inference helper (used by app.py + explainer_agent.py)
# ---------------------------------------------------------------------------

def load_model() -> tuple[xgb.XGBRegressor, list[str]]:
    """Load the saved model and feature list from models/."""
    model_path = MODELS_DIR / "xgb_delay_model.json"
    feat_path = MODELS_DIR / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run train_model.py first.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    with open(feat_path) as f:
        feature_names = json.load(f)

    return model, feature_names


def predict_single(
    model: xgb.XGBRegressor,
    feature_names: list[str],
    input_dict: dict,
) -> tuple[float, np.ndarray]:
    """
    Predict delay for a single observation and return (prediction, shap_values).

    Parameters
    ----------
    input_dict : dict mapping feature names → values (missing keys → 0).
    """
    row = {f: input_dict.get(f, 0) for f in feature_names}
    X = pd.DataFrame([row])

    prediction = float(model.predict(X)[0])

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]

    return prediction, sv


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(tune_hyperparams: bool = False) -> None:
    log.info("=== Model Training Start ===")

    df = load_features()
    X_train, X_test, y_train, y_test, feature_names = split_data(df)

    model = train(X_train, y_train, X_test, y_test, tune_hyperparams=tune_hyperparams)
    metrics = evaluate(model, X_test, y_test)
    shap_values, expected_value = compute_shap(model, X_test)

    save_artefacts(model, feature_names, metrics, shap_values, expected_value)

    log.info("=== Model Training Complete ===")
    log.info(
        "MAE: %.2f min | RMSE: %.2f min | R²: %.3f",
        metrics["mae_minutes"],
        metrics["rmse_minutes"],
        metrics["r2_score"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost delay predictor")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV hyperparameter tuning before final fit",
    )
    args = parser.parse_args()
    run(tune_hyperparams=args.tune)
