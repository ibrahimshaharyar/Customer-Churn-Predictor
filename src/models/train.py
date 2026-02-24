"""
Model Training Module for Banking Customer Churn Pipeline

This module handles training multiple classification models:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- CatBoost Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- AdaBoost Classifier

Author: Banking Churn MLOps Team
"""

import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
from src.utils import logger, CustomException


class ModelTrainer:
    """
    Handles training multiple classification models
    
    This class is responsible for:
    - Loading feature-engineered data
    - Training multiple classification algorithms
    - Saving trained models to artifacts
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Define paths
        self.features_dir = Path("data/features")
        self.models_dir = Path("artifacts/models")
        
        # Initialize models dictionary
        self.models = self._initialize_models()
        
        # Store trained models
        self.trained_models = {}
        
        logger.info(f"ModelTrainer initialized (random_state={random_state})")
        logger.info(f"Number of models to train: {len(self.models)}")
    
    def _initialize_models(self) -> dict:
        """
        Initialize all classification models
        
        Returns:
            dict: Dictionary of model name -> model instance
        """
        models = {
            "Logistic_Regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            "Random_Forest": RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            "Gradient_Boosting": GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            "XGBoost": XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            "CatBoost": CatBoostClassifier(
                random_state=self.random_state,
                verbose=False
            ),
            "KNeighbors": KNeighborsClassifier(
                n_neighbors=5
            ),
            "Decision_Tree": DecisionTreeClassifier(
                random_state=self.random_state
            ),
            "AdaBoost": AdaBoostClassifier(
                random_state=self.random_state,
                n_estimators=100
            )
        }
        
        return models
    
    def load_feature_data(self) -> tuple:
        """
        Load feature-engineered data from CSV files
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info("Loading feature-engineered data...")
            
            # Define file paths
            X_train_path = self.features_dir / "X_train.csv"
            X_test_path = self.features_dir / "X_test.csv"
            y_train_path = self.features_dir / "y_train.csv"
            y_test_path = self.features_dir / "y_test.csv"
            
            # Check if files exist
            for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")
            
            # Load data
            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path)
            y_train = pd.read_csv(y_train_path).squeeze()  # Convert to Series
            y_test = pd.read_csv(y_test_path).squeeze()
            
            logger.info(f"✓ X_train loaded: {X_train.shape}")
            logger.info(f"✓ X_test loaded: {X_test.shape}")
            logger.info(f"✓ y_train loaded: {y_train.shape}")
            logger.info(f"✓ y_test loaded: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error("Failed to load feature data")
            raise CustomException(e, sys)
    
    def train_single_model(self, name: str, model, X_train, y_train):
        """
        Train a single model
        
        Args:
            name (str): Model name
            model: Model instance
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        try:
            logger.info(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            logger.info(f"✓ {name} training complete")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to train {name}")
            raise CustomException(e, sys)
    
    def save_model(self, name: str, model) -> str:
        """
        Save trained model to file
        
        Args:
            name (str): Model name
            model: Trained model instance
            
        Returns:
            str: Path to saved model
        """
        try:
            # Create models directory
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            # Define save path
            model_path = self.models_dir / f"{name}.pkl"
            
            # Save model
            joblib.dump(model, model_path)
            
            logger.info(f"✓ {name} saved to {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save {name}")
            raise CustomException(e, sys)
    
    def run_training(self) -> dict:
        """
        Run the complete model training pipeline
        
        This is the main method that:
        1. Loads feature data
        2. Trains all models
        3. Saves all trained models
        4. Returns trained models dictionary
        
        Returns:
            dict: Dictionary of model name -> trained model
        """
        try:
            logger.info("="*80)
            logger.info("STARTING MODEL TRAINING PIPELINE")
            logger.info("="*80)
            
            # Step 1: Load feature data
            X_train, X_test, y_train, y_test = self.load_feature_data()
            
            # Step 2: Train all models
            logger.info(f"\nTraining {len(self.models)} classification models...")
            logger.info("-" * 80)
            
            for name, model in self.models.items():
                # Train model
                trained_model = self.train_single_model(name, model, X_train, y_train)
                
                # Store trained model
                self.trained_models[name] = trained_model
                
                # Save model
                self.save_model(name, trained_model)
            
            logger.info("="*80)
            logger.info(f"✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"✅ {len(self.trained_models)} models trained and saved")
            logger.info("="*80)
            
            return self.trained_models
            
        except Exception as e:
            logger.error("Model training pipeline failed")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Create trainer instance
    trainer = ModelTrainer()
    
    # Run training pipeline
    trained_models = trainer.run_training()
    
    print(f"\n✅ Model training complete!")
    print(f"Trained models: {list(trained_models.keys())}")
    print(f"Models saved to: artifacts/models/")
