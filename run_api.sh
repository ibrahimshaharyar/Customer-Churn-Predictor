#!/bin/bash
# Start the FastAPI server for predictions

echo "=================================="
echo "Starting Churn Prediction API"
echo "=================================="
echo ""

cd "$(dirname "$0")"

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q fastapi uvicorn pydantic 2>/dev/null || true

echo ""
echo "Starting server..."
echo "Access API at: http://localhost:8000"
echo "API Docs at: http://localhost:8000/docs"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
