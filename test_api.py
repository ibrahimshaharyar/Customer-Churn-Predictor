"""
Test the API endpoints with valid data

Run the API first:
    python -m uvicorn app.main:app --reload

Then run this test script:
    python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Model Info")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 3: Single Prediction")
    print("="*80)
    
    # Example customer - likely to churn
    customer_data = {
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
    
    print(f"\nInput Customer Data:")
    print(json.dumps(customer_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nPrediction Result:")
    print(json.dumps(response.json(), indent=2))


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 4: Batch Prediction (3 customers)")
    print("="*80)
    
    customers = [
        {
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 80000
        },
        {
            "CreditScore": 500,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 50,
            "Tenure": 1,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 30000
        },
        {
            "CreditScore": 800,
            "Geography": "Spain",
            "Gender": "Male",
            "Age": 28,
            "Tenure": 8,
            "Balance": 100000,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 120000
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=customers)
    print(f"Status Code: {response.status_code}")
    print(f"\nBatch Prediction Results:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "API TESTING SCRIPT" + " "*35 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("    python -m uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
