# 🚀 Banking Customer Churn Prediction - Quick Start Guide

## ✅ Setup Instructions

### 1. Install Dependencies

You're already in the `churn_mlops` conda environment. Just install the missing packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install fastapi uvicorn pydantic imbalanced-learn
```

---

## 🎯 How to Run

### Option 1: Run Training Pipeline

**Command**:
```bash
python -m src.main_train
```

**Or use the script**:
```bash
./run_training.sh
```

**What it does**:
- Loads raw data
- Preprocesses data
- Engineers features with SMOTE
- Trains 8 models
- Evaluates and selects best model
- Saves all artifacts

---

### Option 2: Start API Server

**Command**:
```bash
python -m uvicorn app.main:app --reload
```

**Or use the script**:
```bash
./run_api.sh
```

**Access**:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

### Option 3: Make Predictions Directly

**Python**:
```python
from src.models.predict import ChurnPredictor

predictor = ChurnPredictor()
predictor.initialize()

result = predictor.predict_single({
    'CreditScore': 650,
    'Geography': 'Germany',
    'Gender': 'Male',
    'Age': 35,
    'Tenure': 5,
    'Balance': 50000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 80000
})

print(result)
```

---

## 📡 API Usage Examples

### Using cURL

```bash
# Start the API first
python -m uvicorn app.main:app --reload

# In another terminal, make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 80000
  }'
```

### Using Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 80000
    }
)

print(response.json())
```

### Using Interactive Docs

1. Start the API: `python -m uvicorn app.main:app --reload`
2. Open browser: http://localhost:8000/docs
3. Click "POST /predict"
4. Click "Try it out"
5. Edit the example JSON
6. Click "Execute"

---

## 🐛 Troubleshooting

### Import Error: "No module named 'src'"

**Fix**: Use module import syntax
```bash
# ❌ Don't do this
python src/main_train.py

# ✅ Do this instead
python -m src.main_train
```

### Import Error: "No module named 'fastapi'" or "No module named 'imblearn'"

**Fix**: Install dependencies
```bash
pip install fastapi uvicorn pydantic imbalanced-learn
```

Or install everything:
```bash
pip install -r requirements.txt
```

### Model Not Found Error

**Fix**: Run training first to create model files
```bash
python -m src.main_train
```

This will create:
- `artifacts/models/*.pkl` (trained models)
- `artifacts/preprocessors/*.pkl` (encoders, scaler)
- `artifacts/metrics/model_comparison.csv` (results)

---

## 📁 Project Structure

```
Banking Customer Churn/
├── src/
│   ├── main_train.py              # 🔥 Main training orchestration
│   ├── data_ingestion/            # Load data
│   ├── data_preprocessing/        # Clean data
│   ├── feature_engineering/       # SMOTE + scaling
│   ├── models/
│   │   ├── train.py              # Train 8 models
│   │   ├── evaluate.py           # Compare models
│   │   └── predict.py            # Make predictions
│   └── utils/                     # Logging + exceptions
│
├── app/
│   └── main.py                    # 🔥 FastAPI application
│
├── data/
│   ├── raw/                       # Original data
│   ├── processed/                 # Cleaned data
│   └── features/                  # Train/test splits
│
├── artifacts/
│   ├── models/                    # Trained models (.pkl)
│   ├── preprocessors/             # Scaler, encoders (.pkl)
│   └── metrics/                   # Model comparison
│
├── logs/                          # Pipeline logs
├── requirements.txt               # Dependencies
├── run_training.sh               # Training script
└── run_api.sh                    # API start script
```

---

## 🎯 Quick Commands Reference

| Task | Command |
|------|---------|
| **Install dependencies** | `pip install -r requirements.txt` |
| **Run training** | `python -m src.main_train` |
| **Start API** | `python -m uvicorn app.main:app --reload` |
| **API docs** | http://localhost:8000/docs |
| **Health check** | http://localhost:8000/health |
| **Make prediction** | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'` |

---

## 📊 What You've Built

✅ **Complete MLOps Pipeline**:
- Data ingestion with validation
- Preprocessing with quality checks
- Feature engineering with SMOTE
- 8 classification models trained
- Best model: Gradient Boosting (F1: 0.589, Recall: 68.3%)

✅ **Production-Ready API**:
- FastAPI with automatic docs
- Input validation
- Single & batch predictions
- Health monitoring
- Comprehensive logging

✅ **Improved Performance**:
- **43% better recall** with SMOTE
- Catching 84 more churned customers
- Better business value

---

## 🚀 Next Steps

1. **Test the API**: Start server and visit http://localhost:8000/docs
2. **Make predictions**: Try predicting churn for sample customers
3. **Review logs**: Check `logs/` for detailed pipeline execution
4. **Deploy**: Ready for containerization and cloud deployment!

Need help? Check:
- `PHASE1_SUMMARY.md` - Logging & exceptions
- `PHASE2_SUMMARY.md` - Data pipeline
- `PHASE3_SUMMARY.md` - Model training
- `PHASE4_SUMMARY.md` - Prediction & serving
- `SMOTE_RESULTS.md` - Performance improvements
