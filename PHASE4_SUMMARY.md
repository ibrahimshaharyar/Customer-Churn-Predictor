# Phase 4: Pipeline Orchestration & Prediction - Summary

## ✅ What Was Implemented

### 1. **Main Training Orchestration** (`src/main_train.py`)

**Purpose**: End-to-end pipeline orchestration script

**Features**:
- Runs complete training pipeline in 5 automated phases:
  1. Data Ingestion
  2. Data Preprocessing
  3. Feature Engineering (with SMOTE)
  4. Model Training
  5. Model Evaluation
- Single function `run_training_pipeline()` executes entire workflow
- Comprehensive logging at each phase
- Returns best model info and all artifact paths
- Custom exception handling throughout

**Usage**:
```bash
python src/main_train.py
```

---

### 2. **Prediction Module** (`src/models/predict.py`)

**Purpose**: Load trained model and make predictions

**Key Features**:
- `ChurnPredictor` class handles all prediction logic
- Automatically loads:
  - Best trained model (Gradient_Boosting.pkl)
  - Preprocessing artifacts (scaler, encoders)
- `predict_single()` - Single customer prediction
- `predict()` - Batch predictions
- Returns:  
  - Prediction (0=Stayed, 1=Churned)
  - Prediction label
  - Churn probability
  - Risk category (Low/Medium/High)

**Example**:
```python
from src.models.predict import ChurnPredictor

predictor = ChurnPredictor()
predictor.initialize()

customer = {
    'CreditScore': 619,
    'Geography': 'France',
    'Gender': 'Female',
    'Age': 42,
    'Tenure': 2,
    'Balance': 0.00,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 101348.88
}

result = predictor.predict_single(customer)
# Output: {'prediction': 1, 'prediction_label': 'Churned', 
#          'churn_probability': 0.5787, 'churn_risk': 'Medium Risk'}
```

---

### 3. **FastAPI Application** (`app/main.py`)

**Purpose**: REST API for serving predictions

**Endpoints**:

#### `GET /`
- Welcome endpoint
- Returns API info

#### `GET /health`
- Health check
- Verifies model and preprocessors loaded

#### `GET /model/info`
- Returns model information
- Model name, type, features, status

#### `POST /predict`
- Single customer prediction
- **Request**: CustomerData (10 features)
- **Response**: Prediction + probability + risk + message

#### `POST /predict/batch`
- Batch predictions
- **Request**: List of CustomerData
- **Response**: List of predictions

**Features**:
- Pydantic models for request/response validation
- Input validation (credit score 300-850, age 18-100, etc.)
- Automatic API documentation at `/docs`
- Model loaded once on startup for efficiency
- Comprehensive error handling

**Start Server**:
```bash
cd app
python main.py
# OR
uvicorn main:app --reload
```

**API Documentation**:
- http://localhost:8000/docs (Interactive Swagger UI)
- http://localhost:8000/redoc (ReDoc)

---

## 📁 Files Created

```
src/
├── main_train.py                 ✨ NEW (Pipeline orchestration)
└── models/
    └── predict.py                ✨ NEW (Prediction module)

app/
├── main.py                       ✨ NEW (FastAPI application)
└── requirements.txt              ✨ NEW (FastAPI dependencies)
```

---

## ✅ Testing Results

### Prediction Module Test
```bash
python src/models/predict.py
```

**Result**:
```
Prediction: Churned
Churn Probability: 57.87%
Risk Category: Medium Risk
```

✅ Model loads successfully  
✅ Preprocessors load successfully  
✅ Prediction works correctly  
✅ Probability calculation working  

---

## 🚀 How to Use

### Method 1: Direct Prediction (Python)

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

### Method 2: API (cURL)

```bash
# Start API
cd app && python main.py

# Make prediction
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

### Method 3: API (Python Requests)

```python
import requests

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

---

## 🎯 Key Design Decisions

### 1. **Predictor Initialization on Startup**
- Model loaded once when FastAPI starts
- Prevents loading model on every request (slow!)
- Much faster response times

### 2. **Pydantic Validation**
- Automatic input validation
- Type checking
- Value range validation (e.g., CreditScore 300-850)
- Clear error messages for invalid inputs

### 3. **Risk Categorization**
- < 30%: Low Risk
- 30-60%: Medium Risk
- \> 60%: High Risk
- Helps business prioritize interventions

### 4. **Batch Predictions**
- Process multiple customers at once
- Efficient for bulk scoring
- Useful for daily churn risk reports

### 5. **Comprehensive Logging**
- Every prediction logged
- Request/response tracked
- Easy debugging and monitoring

---

## 📊 API Response Example

**Request**:
```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.00,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```

**Response**:
```json
{
  "prediction": 1,
  "prediction_label": "Churned",
  "churn_probability": 0.5787,
  "churn_risk": "Medium Risk",
  "message": "Customer is likely to CHURN. Medium Risk - take action!"
}
```

---

## ✅ Benefits

1. **End-to-End Automation**: Single command runs entire pipeline
2. **Easy Predictions**: Simple Python API or REST API
3. **Production-Ready**: FastAPI with validation and error handling
4. **Scalable**: Can handle single or batch predictions
5. **Well-Documented**: Auto-generated API docs
6. **Monitored**: Comprehensive logging throughout

---

## 🚀 Next Steps

Phase 4 complete! Ready for deployment.

**Optional Enhancements**:
- Add authentication to API
- Add rate limiting
- Containerize with Docker
- Deploy to cloud (AWS, Azure, GCP)
- Add model monitoring dashboard
- Implement A/B testing for models
