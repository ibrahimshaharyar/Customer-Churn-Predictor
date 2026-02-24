"""
Model Evaluation Module for Banking Customer Churn Pipeline

This module handles evaluating trained classification models:
- Calculates classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Compares all models
- Identifies best performing model
- Generates confusion matrices
- Saves evaluation results

Author: Banking Churn MLOps Team
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from src.utils import logger, CustomException


class ModelEvaluator:
    """
    Handles evaluation of trained classification models
    
    This class is responsible for:
    - Loading trained models
    - Calculating classification metrics
    - Comparing model performance
    - Saving evaluation results
    """
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        # Define paths
        self.features_dir = Path("data/features")
        self.models_dir = Path("artifacts/models")
        self.metrics_dir = Path("artifacts/metrics")
        
        # Store evaluation results
        self.evaluation_results = {}
        
        logger.info("ModelEvaluator initialized")
    
    def load_test_data(self) -> tuple:
        """
        Load test data for evaluation
        
        Returns:
            tuple: (X_test, y_test)
        """
        try:
            logger.info("Loading test data...")
            
            X_test_path = self.features_dir / "X_test.csv"
            y_test_path = self.features_dir / "y_test.csv"
            
            if not X_test_path.exists() or not y_test_path.exists():
                raise FileNotFoundError("Test data files not found")
            
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).squeeze()
            
            logger.info(f"✓ Test data loaded: X_test {X_test.shape}, y_test {y_test.shape}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error("Failed to load test data")
            raise CustomException(e, sys)
    
    def load_model(self, model_name: str):
        """
        Load a trained model from file
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            Loaded model instance
        """
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            
            logger.info(f"✓ Loaded model: {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}")
            raise CustomException(e, sys)
    
    def evaluate_single_model(self, name: str, model, X_test, y_test) -> dict:
        """
        Evaluate a single model on test data
        
        Args:
            name (str): Model name
            model: Trained model instance
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probability predictions (for ROC-AUC)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None
                logger.warning(f"⚠ {name} does not support predict_proba")
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1_Score': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Add ROC-AUC if available
            if y_pred_proba is not None:
                metrics['ROC_AUC'] = roc_auc_score(y_test, y_pred_proba)
            else:
                metrics['ROC_AUC'] = None
            
            logger.info(f"✓ {name} evaluation complete")
            logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['Precision']:.4f}")
            logger.info(f"  Recall: {metrics['Recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['F1_Score']:.4f}")
            if metrics['ROC_AUC']:
                logger.info(f"  ROC-AUC: {metrics['ROC_AUC']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate {name}")
            raise CustomException(e, sys)
    
    def get_confusion_matrix(self, name: str, model, X_test, y_test) -> np.ndarray:
        """
        Get confusion matrix for a model
        
        Args:
            name (str): Model name
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            np.ndarray: Confusion matrix
        """
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            logger.info(f"✓ Confusion matrix generated for {name}")
            logger.info(f"  True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
            logger.info(f"  False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
            
            return cm
            
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix for {name}")
            raise CustomException(e, sys)
    
    def get_classification_report(self, name: str, model, X_test, y_test) -> str:
        """
        Get detailed classification report
        
        Args:
            name (str): Model name
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            str: Classification report
        """
        try:
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test, y_pred, 
                target_names=['Stayed (0)', 'Churned (1)']
            )
            
            logger.info(f"✓ Classification report generated for {name}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate classification report for {name}")
            raise CustomException(e, sys)
    
    def compare_models(self, results: list) -> pd.DataFrame:
        """
        Compare all models and create comparison DataFrame
        
        Args:
            results (list): List of metric dictionaries
            
        Returns:
            pd.DataFrame: Comparison table sorted by F1 Score
        """
        try:
            logger.info("Creating model comparison table...")
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Sort by F1 Score (descending)
            df = df.sort_values('F1_Score', ascending=False)
            
            # Reset index
            df = df.reset_index(drop=True)
            
            logger.info("✓ Model comparison table created")
            logger.info(f"\n{df.to_string()}")
            
            return df
            
        except Exception as e:
            logger.error("Failed to create comparison table")
            raise CustomException(e, sys)
    
    def get_best_model(self, comparison_df: pd.DataFrame) -> tuple:
        """
        Identify the best performing model based on F1 Score
        
        Args:
            comparison_df (pd.DataFrame): Model comparison table
            
        Returns:
            tuple: (best_model_name, best_f1_score)
        """
        try:
            best_row = comparison_df.iloc[0]
            best_model_name = best_row['Model']
            best_f1_score = best_row['F1_Score']
            
            logger.info(f"✓ Best model identified: {best_model_name}")
            logger.info(f"  F1 Score: {best_f1_score:.4f}")
            
            return best_model_name, best_f1_score
            
        except Exception as e:
            logger.error("Failed to identify best model")
            raise CustomException(e, sys)
    
    def save_comparison_results(self, comparison_df: pd.DataFrame) -> str:
        """
        Save model comparison results to CSV
        
        Args:
            comparison_df (pd.DataFrame): Comparison table
            
        Returns:
            str: Path to saved file
        """
        try:
            logger.info("Saving comparison results...")
            
            # Create metrics directory
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            save_path = self.metrics_dir / "model_comparison.csv"
            comparison_df.to_csv(save_path, index=False)
            
            logger.info(f"✓ Comparison results saved to {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            logger.error("Failed to save comparison results")
            raise CustomException(e, sys)
    
    def run_evaluation(self) -> dict:
        """
        Run the complete model evaluation pipeline
        
        This is the main method that:
        1. Loads test data
        2. Loads all trained models
        3. Evaluates each model
        4. Compares models
        5. Identifies best model
        6. Saves results
        
        Returns:
            dict: Evaluation results including comparison DataFrame and best model
        """
        try:
            logger.info("="*80)
            logger.info("STARTING MODEL EVALUATION PIPELINE")
            logger.info("="*80)
            
            # Step 1: Load test data
            X_test, y_test = self.load_test_data()
            
            # Step 2: Get list of trained models
            model_files = list(self.models_dir.glob("*.pkl"))
            
            if not model_files:
                raise FileNotFoundError(f"No trained models found in {self.models_dir}")
            
            logger.info(f"\nFound {len(model_files)} trained models")
            logger.info("-" * 80)
            
            # Step 3: Evaluate each model
            results = []
            for model_file in model_files:
                model_name = model_file.stem  # Get filename without extension
                
                # Load model
                model = self.load_model(model_name)
                
                # Evaluate model
                metrics = self.evaluate_single_model(model_name, model, X_test, y_test)
                results.append(metrics)
                
                logger.info("-" * 80)
            
            # Step 4: Compare models
            comparison_df = self.compare_models(results)
            
            # Step 5: Identify best model
            best_model_name, best_f1_score = self.get_best_model(comparison_df)
            
            # Step 6: Get detailed report for best model
            best_model = self.load_model(best_model_name)
            best_model_cm = self.get_confusion_matrix(best_model_name, best_model, X_test, y_test)
            best_model_report = self.get_classification_report(best_model_name, best_model, X_test, y_test)
            
            logger.info("\n" + "="*80)
            logger.info(f"BEST MODEL: {best_model_name}")
            logger.info("="*80)
            logger.info(f"\n{best_model_report}")
            
            # Step 7: Save comparison results
            save_path = self.save_comparison_results(comparison_df)
            
            logger.info("="*80)
            logger.info("✅ MODEL EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            return {
                'comparison_df': comparison_df,
                'best_model_name': best_model_name,
                'best_f1_score': best_f1_score,
                'best_model_confusion_matrix': best_model_cm,
                'best_model_report': best_model_report,
                'save_path': save_path
            }
            
        except Exception as e:
            logger.error("Model evaluation pipeline failed")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Create evaluator instance
    evaluator = ModelEvaluator()
    
    # Run evaluation pipeline
    results = evaluator.run_evaluation()
    
    print(f"\n✅ Model evaluation complete!")
    print(f"Best model: {results['best_model_name']}")
    print(f"F1 Score: {results['best_f1_score']:.4f}")
    print(f"\nComparison saved to: {results['save_path']}")
