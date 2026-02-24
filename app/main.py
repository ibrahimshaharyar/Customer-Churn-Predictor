"""
FastAPI Application for Banking Customer Churn Prediction

This API provides endpoints for:
- Health check
- Making churn predictions
- Getting model information

Author: Banking Churn MLOps Team
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
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

# Initialize predictor (loaded once on startup)
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on app startup"""
    global predictor
    try:
        logger.info("Initializing ChurnPredictor on app startup...")
        predictor = ChurnPredictor(model_name="Gradient_Boosting")
        predictor.initialize()
        logger.info("✅ ChurnPredictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise


# Pydantic models for request/response validation
class CustomerData(BaseModel):
    """Customer data for churn prediction"""
    CreditScore: int = Field(..., ge=300, le=850, description="Customer credit score (300-850)")
    Geography: str = Field(..., pattern="^(France|Germany|Spain)$", description="Customer geography: France, Germany, or Spain")
    Gender: str = Field(..., pattern="^(Female|Male)$", description="Customer gender: Female or Male")
    Age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    Tenure: int = Field(..., ge=0, le=10, description="Years with the bank (0-10)")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0=No, 1=Yes)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0=No, 1=Yes)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int
    prediction_label: str
    churn_probability: Optional[float]
    churn_risk: Optional[str]
    message: str


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_type: str
    features: list
    status: str


# API Endpoints
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Banking Customer Churn Prediction API",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "preprocessors_loaded": (
            predictor.scaler is not None and 
            predictor.label_encoder_geo is not None and
            predictor.label_encoder_gender is not None
        )
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return ModelInfo(
        model_name=predictor.model_name,
        model_type="Gradient Boosting Classifier",
        features=predictor.expected_features,
        status="loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict customer churn
    
    Args:
        customer: Customer data
        
    Returns:
        PredictionResponse: Prediction result with probability and risk category
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Convert Pydantic model to dict
        customer_dict = customer.dict()
        
        logger.info(f"Received prediction request for customer: {customer_dict}")
        
        # Make prediction
        result = predictor.predict_single(customer_dict)
        
        # Format response message
        if result['prediction'] == 1:
            message = f"Customer is likely to CHURN. {result['churn_risk']} - take action!"
        else:
            message = f"Customer is likely to STAY. {result['churn_risk']} - continue engagement."
        
        response = PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            churn_probability=result['churn_probability'],
            churn_risk=result['churn_risk'],
            message=message
        )
        
        logger.info(f"Prediction result: {result['prediction_label']} ({result['churn_probability']:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    """
    Predict churn for multiple customers
    
    Args:
        customers: List of customer data
        
    Returns:
        List of predictions
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        results = []
        
        for i, customer in enumerate(customers):
            customer_dict = customer.dict()
            result = predictor.predict_single(customer_dict)
            
            results.append({
                "customer_index": i,
                "prediction": result['prediction'],
                "prediction_label": result['prediction_label'],
                "churn_probability": result['churn_probability'],
                "churn_risk": result['churn_risk']
            })
        
        logger.info(f"Batch prediction completed for {len(customers)} customers")
        
        return {
            "total_customers": len(customers),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Run the app
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 Starting Banking Customer Churn Prediction API")
    print("="*80)
    print("\n📍 API will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("📊 Health check: http://localhost:8000/health")
    print("\n" + "="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
