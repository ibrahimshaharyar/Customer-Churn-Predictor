"""
Main Training Pipeline Orchestration Script

This script runs the complete end-to-end MLOps pipeline:
1. Data Ingestion
2. Data Preprocessing  
3. Feature Engineering (with SMOTE)
4. Model Training
5. Model Evaluation

Author: Banking Churn MLOps Team
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion.ingest import DataIngestion
from src.data_preprocessing.preprocess import DataPreprocessor
from src.feature_engineering.features import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils import logger, CustomException


def run_training_pipeline(use_smote: bool = True):
    """
    Run the complete training pipeline
    
    Args:
        use_smote (bool): Whether to use SMOTE for class balancing
        
    Returns:
        dict: Dictionary containing best model info and paths
    """
    try:
        logger.info("="*80)
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)
        
        # ============================================================
        # PHASE 1: DATA INGESTION
        # ============================================================
        logger.info("\n📥 PHASE 1: Data Ingestion")
        logger.info("-" * 80)
        
        ingestion = DataIngestion()
        df_raw = ingestion.run_ingestion()
        
        logger.info(f"✅ Data ingestion complete: {df_raw.shape}")
        
        # ============================================================
        # PHASE 2: DATA PREPROCESSING
        # ============================================================
        logger.info("\n🧹 PHASE 2: Data Preprocessing")
        logger.info("-" * 80)
        
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.run_preprocessing(df_raw)
        
        logger.info(f"✅ Data preprocessing complete: {df_processed.shape}")
        
        # ============================================================
        # PHASE 3: FEATURE ENGINEERING
        # ============================================================
        logger.info(f"\n⚙️ PHASE 3: Feature Engineering (SMOTE: {use_smote})")
        logger.info("-" * 80)
        
        feature_engineer = FeatureEngineer(use_smote=use_smote)
        feature_results = feature_engineer.run_feature_engineering(df_processed)
        
        logger.info(f"✅ Feature engineering complete")
        logger.info(f"   Training samples: {feature_results['X_train'].shape[0]}")
        logger.info(f"   Test samples: {feature_results['X_test'].shape[0]}")
        
        # ============================================================
        # PHASE 4: MODEL TRAINING
        # ============================================================
        logger.info("\n🤖 PHASE 4: Model Training")
        logger.info("-" * 80)
        
        trainer = ModelTrainer()
        trained_models = trainer.run_training()
        
        logger.info(f"✅ Model training complete: {len(trained_models)} models trained")
        
        # ============================================================
        # PHASE 5: MODEL EVALUATION
        # ============================================================
        logger.info("\n📊 PHASE 5: Model Evaluation")
        logger.info("-" * 80)
        
        evaluator = ModelEvaluator()
        eval_results = evaluator.run_evaluation()
        
        logger.info(f"✅ Model evaluation complete")
        logger.info(f"   Best model: {eval_results['best_model_name']}")
        logger.info(f"   Best F1 Score: {eval_results['best_f1_score']:.4f}")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"\n📈 RESULTS SUMMARY:")
        logger.info(f"   - Best Model: {eval_results['best_model_name']}")
        logger.info(f"   - F1 Score: {eval_results['best_f1_score']:.4f}")
        logger.info(f"   - Models saved: artifacts/models/")
        logger.info(f"   - Metrics saved: artifacts/metrics/")
        logger.info(f"   - Preprocessors saved: artifacts/preprocessors/")
        logger.info("="*80)
        
        return {
            'best_model_name': eval_results['best_model_name'],
            'best_f1_score': eval_results['best_f1_score'],
            'comparison_df': eval_results['comparison_df'],
            'feature_paths': feature_results['data_paths'],
            'artifact_paths': feature_results['artifact_paths']
        }
        
    except Exception as e:
        logger.error("Training pipeline failed!")
        raise CustomException(e, sys)


if __name__ == "__main__":
    """
    Main entry point for training pipeline
    
    Usage:
        python src/main_train.py
    """
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "BANKING CHURN PREDICTION" + " "*35 + "║")
    print("║" + " "*25 + "TRAINING PIPELINE" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    try:
        # Run the complete training pipeline with SMOTE enabled
        results = run_training_pipeline(use_smote=True)
        
        print("\n✅ Training pipeline completed successfully!")
        print(f"\nBest Model: {results['best_model_name']}")
        print(f"F1 Score: {results['best_f1_score']:.4f}")
        print(f"\nModel Comparison:")
        print(results['comparison_df'].to_string(index=False))
        print("\nAll artifacts saved! Ready for predictions.")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed!")
        print(f"Error: {str(e)}")
        print("\nCheck logs/ for detailed error information")
        sys.exit(1)
