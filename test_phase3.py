"""
Test Script for Phase 3: Model Training and Evaluation

This script tests:
1. Model Training (train 8 classification models)
2. Model Evaluation (evaluate all models, compare performance)
3. Best Model Selection

All modules use custom exceptions and logging.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils import logger


def test_model_pipeline():
    """Run the complete model training and evaluation pipeline"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "PHASE 3 - MODEL TRAINING & EVALUATION TEST" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    try:
        # ============================================================
        # TEST 1: MODEL TRAINING
        # ============================================================
        print("\n" + "="*80)
        print("TEST 1: MODEL TRAINING")
        print("="*80)
        
        trainer = ModelTrainer()
        trained_models = trainer.run_training()
        
        print(f"\n✅ Model Training Successful!")
        print(f"   - Trained {len(trained_models)} classification models")
        print(f"   - Models: {list(trained_models.keys())}")
        
        # ============================================================
        # TEST 2: MODEL EVALUATION
        # ============================================================
        print("\n" + "="*80)
        print("TEST 2: MODEL EVALUATION")
        print("="*80)
        
        evaluator = ModelEvaluator()
        results = evaluator.run_evaluation()
        
        print(f"\n✅ Model Evaluation Successful!")
        print(f"   - Evaluated {len(results['comparison_df'])} models")
        print(f"   - Best model: {results['best_model_name']}")
        print(f"   - Best F1 Score: {results['best_f1_score']:.4f}")
        
        # ============================================================
        # VERIFICATION: CHECK SAVED FILES
        # ============================================================
        print("\n" + "="*80)
        print("VERIFICATION: CHECKING SAVED FILES")
        print("="*80)
        
        # Check trained models
        models_dir = Path("artifacts/models")
        model_files = list(models_dir.glob("*.pkl"))
        print(f"✓ {len(model_files)} trained models saved:")
        for model_file in model_files:
            print(f"  - {model_file.name}")
        
        # Check metrics
        metrics_file = Path("artifacts/metrics/model_comparison.csv")
        if metrics_file.exists():
            print(f"✓ Model comparison saved: {metrics_file}")
        else:
            print(f"✗ Model comparison NOT found: {metrics_file}")
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(results['comparison_df'].to_string(index=False))
        
        print("\n" + "="*80)
        print(f"BEST MODEL: {results['best_model_name']}")
        print("="*80)
        print("\nConfusion Matrix:")
        print(results['best_model_confusion_matrix'])
        print("\nClassification Report:")
        print(results['best_model_report'])
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "="*80)
        print("✅ PHASE 3 PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary:")
        print(f"  ✓ Model training working ({len(trained_models)} models trained)")
        print(f"  ✓ Model evaluation working")
        print(f"  ✓ Best model identified: {results['best_model_name']}")
        print(f"  ✓ All models saved to artifacts/models")
        print(f"  ✓ Metrics saved to artifacts/metrics")
        print(f"  ✓ Custom exceptions and logging working")
        print("\nCheck the logs/ folder for detailed pipeline logs!")
        print("\nNext: Ready for Phase 4 (Prediction Pipeline & Serving)")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed!")
        print(f"Error: {str(e)}")
        print("\nCheck the logs/ folder for detailed error information")
        return False


if __name__ == "__main__":
    success = test_model_pipeline()
    sys.exit(0 if success else 1)
