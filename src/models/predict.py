"""
Prediction Module for Banking Customer Churn Pipeline

This module handles making predictions using the trained model:
- Loads the best trained model
- Loads preprocessing artifacts (encoders, scaler)
- Preprocesses new data
- Makes churn predictions
- Returns prediction with probability

Author: Banking Churn MLOps Team
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Dict

# Fix import path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.utils.exception import CustomException


class ChurnPredictor:
    """
    Handles churn predictions using trained model
    
    This class is responsible for:
    - Loading best trained model
    - Loading preprocessing artifacts
    - Preprocessing new customer data
    - Making predictions
    """
    
    def __init__(self, model_name: str = "Gradient_Boosting"):
        """
        Initialize ChurnPredictor
        
        Args:
            model_name (str): Name of the model to load (default: Gradient_Boosting)
        """
        self.model_name = model_name
        
        # Define paths
        self.models_dir = Path("artifacts/models")
        self.preprocessors_dir = Path("artifacts/preprocessors")
        
        # Initialize placeholders
        self.model = None
        self.scaler = None
        self.label_encoder_geo = None
        self.label_encoder_gender = None
        
        # Expected feature columns (after preprocessing)
        self.expected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary'
        ]
        
        logger.info(f"ChurnPredictor initialized with model: {model_name}")
    
    def load_model(self):
        """Load the trained model from file"""
        try:
            model_path = self.models_dir / f"{self.model_name}.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info(f"✓ Model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}")
            raise CustomException(e, sys)
    
    def load_preprocessors(self):
        """Load preprocessing artifacts (scaler, encoders)"""
        try:
            logger.info("Loading preprocessing artifacts...")
            
            # Load scaler
            scaler_path = self.preprocessors_dir / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            logger.info("✓ Scaler loaded")
            
            # Load Geography encoder
            geo_encoder_path = self.preprocessors_dir / "label_encoder_geo.pkl"
            if not geo_encoder_path.exists():
                raise FileNotFoundError(f"Geography encoder not found: {geo_encoder_path}")
            self.label_encoder_geo = joblib.load(geo_encoder_path)
            logger.info("✓ Geography encoder loaded")
            
            # Load Gender encoder
            gender_encoder_path = self.preprocessors_dir / "label_encoder_gender.pkl"
            if not gender_encoder_path.exists():
                raise FileNotFoundError(f"Gender encoder not found: {gender_encoder_path}")
            self.label_encoder_gender = joblib.load(gender_encoder_path)
            logger.info("✓ Gender encoder loaded")
            
        except Exception as e:
            logger.error("Failed to load preprocessing artifacts")
            raise CustomException(e, sys)
    
    def initialize(self):
        """Initialize predictor by loading model and preprocessors"""
        try:
            logger.info("Initializing ChurnPredictor...")
            
            self.load_model()
            self.load_preprocessors()
            
            logger.info("✅ ChurnPredictor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ChurnPredictor")
            raise CustomException(e, sys)
    
    def preprocess_input(self, input_data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Either a DataFrame or dict with customer features
            
        Returns:
            np.ndarray: Preprocessed features ready for prediction
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Make a copy to avoid modifying original
            df = input_data.copy()
            
            logger.info("Preprocessing input data...")
            logger.info(f"Input shape: {df.shape}")
            
            # Check for required columns
            required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                           'EstimatedSalary']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Encode Geography
            if 'Geography' in df.columns:
                df['Geography'] = self.label_encoder_geo.transform(df['Geography'])
                logger.info("✓ Geography encoded")
            
            # Encode Gender
            if 'Gender' in df.columns:
                df['Gender'] = self.label_encoder_gender.transform(df['Gender'])
                logger.info("✓ Gender encoded")
            
            # Select only the expected features in the correct order
            df = df[self.expected_features]
            
            # Scale features
            df_scaled = self.scaler.transform(df)
            logger.info("✓ Features scaled")
            
            logger.info(f"✅ Preprocessing complete. Output shape: {df_scaled.shape}")
            
            return df_scaled
            
        except Exception as e:
            logger.error("Failed to preprocess input data")
            raise CustomException(e, sys)
    
    def predict(self, input_data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Make churn prediction for customer(s)
        
        Args:
            input_data: Customer data (DataFrame or dict)
            
        Returns:
            dict: Prediction results with probabilities
        """
        try:
            logger.info("Making prediction...")
            
            # Ensure model and preprocessors are loaded
            if self.model is None:
                logger.info("Model not loaded, initializing...")
                self.initialize()
            
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(X)
            
            # Get prediction probabilities (if supported)
            try:
                probabilities = self.model.predict_proba(X)
                prob_churn = probabilities[:, 1]  # Probability of churning
            except:
                logger.warning("Model does not support predict_proba")
                prob_churn = None
            
            # Format results
            results = {
                'prediction': prediction.tolist(),
                'prediction_label': ['Stayed' if p == 0 else 'Churned' for p in prediction],
                'churn_probability': prob_churn.tolist() if prob_churn is not None else None
            }
            
            logger.info(f"✅ Prediction complete")
            logger.info(f"   Predictions: {results['prediction']}")
            if prob_churn is not None:
                logger.info(f"   Churn probabilities: {[f'{p:.2%}' for p in prob_churn]}")
            
            return results
            
        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)
    
    def predict_single(self, customer_data: Dict) -> Dict:
        """
        Make prediction for a single customer
        
        Args:
            customer_data (dict): Single customer's features
            
        Returns:
            dict: Prediction result
        """
        try:
            results = self.predict(customer_data)
            
            # Return single prediction
            return {
                'prediction': results['prediction'][0],
                'prediction_label': results['prediction_label'][0],
                'churn_probability': results['churn_probability'][0] if results['churn_probability'] else None,
                'churn_risk': self._get_risk_category(results['churn_probability'][0]) if results['churn_probability'] else None
            }
            
        except Exception as e:
            logger.error("Single prediction failed")
            raise CustomException(e, sys)
    
    def _get_risk_category(self, probability: float) -> str:
        """
        Categorize churn risk based on probability
        
        Args:
            probability (float): Churn probability (0-1)
            
        Returns:
            str: Risk category
        """
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = ChurnPredictor(model_name="Gradient_Boosting")
    predictor.initialize()
    
    # Example customer data
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
    
    # Make prediction
    result = predictor.predict_single(customer)
    
    print("\n" + "="*80)
    print("CHURN PREDICTION RESULT")
    print("="*80)
    print(f"Prediction: {result['prediction_label']}")
    if result['churn_probability']:
        print(f"Churn Probability: {result['churn_probability']:.2%}")
        print(f"Risk Category: {result['churn_risk']}")
    print("="*80)
