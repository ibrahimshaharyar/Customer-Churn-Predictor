"""
FastAPI Application for Banking Customer Churn Prediction

Endpoints:
- GET  /               → Serve HTML frontend
- GET  /health         → Health check
- GET  /model/info     → Model metadata
- GET  /dashboard/data → Pre-aggregated EDA stats + model metrics
- POST /predict        → Single customer churn prediction
- POST /predict/batch  → Batch prediction
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List

# Fix import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.predict import ChurnPredictor
from src.utils.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="Banking Customer Churn Prediction API",
    description="API for predicting customer churn using ML models",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Globals
predictor = None
dashboard_cache = None


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global predictor, dashboard_cache
    # Load predictor
    try:
        logger.info("Initializing ChurnPredictor on app startup...")
        predictor = ChurnPredictor(model_name="Gradient_Boosting")
        predictor.initialize()
        logger.info("✅ ChurnPredictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise

    # Pre-compute dashboard data
    try:
        dashboard_cache = compute_dashboard_data()
        logger.info("✅ Dashboard data computed successfully")
    except Exception as e:
        logger.warning(f"Dashboard data computation failed: {str(e)}")
        dashboard_cache = {}


def compute_dashboard_data() -> dict:
    """Read processed CSV and compute aggregated stats for the dashboard."""
    csv_path = project_root / "data" / "processed" / "churn_cleaned.csv"
    metrics_path = project_root / "artifacts" / "metrics" / "model_comparison.csv"

    # ── Read dataset ──────────────────────────────────────────────────────────
    churn_counts = defaultdict(int)          # {0: stayed, 1: churned}
    geo_counts = defaultdict(int)
    gender_counts = defaultdict(int)
    active_counts = defaultdict(int)
    age_buckets = {"18-30": 0, "31-40": 0, "41-50": 0, "51-60": 0, "61+": 0}
    balance_buckets = {"Zero Balance": 0, "1-50k": 0, "50k-100k": 0, "100k-150k": 0, "150k+": 0}
    geo_churn = defaultdict(lambda: {"total": 0, "churned": 0})
    total = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            exited = int(row["Exited"])
            churn_counts[exited] += 1

            geo = row["Geography"]
            geo_counts[geo] += 1
            geo_churn[geo]["total"] += 1
            geo_churn[geo]["churned"] += exited

            gender_counts[row["Gender"]] += 1
            active_counts[int(row["IsActiveMember"])] += 1

            age = int(row["Age"])
            if age <= 30:
                age_buckets["18-30"] += 1
            elif age <= 40:
                age_buckets["31-40"] += 1
            elif age <= 50:
                age_buckets["41-50"] += 1
            elif age <= 60:
                age_buckets["51-60"] += 1
            else:
                age_buckets["61+"] += 1

            bal = float(row["Balance"])
            if bal == 0:
                balance_buckets["Zero Balance"] += 1
            elif bal < 50000:
                balance_buckets["1-50k"] += 1
            elif bal < 100000:
                balance_buckets["50k-100k"] += 1
            elif bal < 150000:
                balance_buckets["100k-150k"] += 1
            else:
                balance_buckets["150k+"] += 1

    churn_rate = round(churn_counts[1] / total * 100, 1)

    # Churn rate by geography
    geo_churn_rate = {
        g: round(v["churned"] / v["total"] * 100, 1)
        for g, v in geo_churn.items()
    }

    # ── Read model metrics ────────────────────────────────────────────────────
    models = []
    try:
        with open(metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append({
                    "name": row["Model"].replace("_", " "),
                    "accuracy": round(float(row["Accuracy"]) * 100, 1),
                    "precision": round(float(row["Precision"]) * 100, 1),
                    "recall": round(float(row["Recall"]) * 100, 1),
                    "f1": round(float(row["F1_Score"]) * 100, 1),
                    "roc_auc": round(float(row["ROC_AUC"]) * 100, 1),
                })
    except Exception:
        models = []

    return {
        "overview": {
            "total_customers": total,
            "churn_rate": churn_rate,
            "stayed": churn_counts[0],
            "churned": churn_counts[1],
            "countries": len(geo_counts),
        },
        "churn_distribution": {
            "labels": ["Stayed", "Churned"],
            "values": [churn_counts[0], churn_counts[1]],
        },
        "geography": {
            "labels": list(geo_counts.keys()),
            "values": list(geo_counts.values()),
            "churn_rates": [geo_churn_rate.get(g, 0) for g in geo_counts.keys()],
        },
        "gender": {
            "labels": list(gender_counts.keys()),
            "values": list(gender_counts.values()),
        },
        "age_distribution": {
            "labels": list(age_buckets.keys()),
            "values": list(age_buckets.values()),
        },
        "balance_distribution": {
            "labels": list(balance_buckets.keys()),
            "values": list(balance_buckets.values()),
        },
        "active_members": {
            "labels": ["Active", "Inactive"],
            "values": [active_counts[1], active_counts[0]],
        },
        "models": models,
    }


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class CustomerData(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    Geography: str = Field(..., pattern="^(France|Germany|Spain)$")
    Gender: str = Field(..., pattern="^(Female|Male)$")
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: Optional[float]
    churn_risk: Optional[str]
    message: str


class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    features: list
    status: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "preprocessors_loaded": (
            predictor.scaler is not None
            and predictor.label_encoder_geo is not None
            and predictor.label_encoder_gender is not None
        ),
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return ModelInfo(
        model_name=predictor.model_name,
        model_type="Gradient Boosting Classifier",
        features=predictor.expected_features,
        status="loaded",
    )


@app.get("/dashboard/data")
async def get_dashboard_data():
    """Return pre-aggregated EDA statistics and model metrics."""
    if dashboard_cache is None:
        raise HTTPException(status_code=503, detail="Dashboard data not ready")
    return dashboard_cache


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        customer_dict = customer.dict()
        logger.info(f"Prediction request: {customer_dict}")
        result = predictor.predict_single(customer_dict)
        message = (
            f"Customer is likely to CHURN. {result['churn_risk']} - take action!"
            if result["prediction"] == 1
            else f"Customer is likely to STAY. {result['churn_risk']} - continue engagement."
        )
        return PredictionResponse(
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            churn_probability=result["churn_probability"],
            churn_risk=result["churn_risk"],
            message=message,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        results = []
        for i, customer in enumerate(customers):
            result = predictor.predict_single(customer.dict())
            results.append({
                "customer_index": i,
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "churn_probability": result["churn_probability"],
                "churn_risk": result["churn_risk"],
            })
        return {"total_customers": len(customers), "predictions": results}
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("🚀 Starting Banking Customer Churn Prediction API")
    print("=" * 80)
    print("\n📍 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("📊 Dashboard: http://localhost:8000")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000)
