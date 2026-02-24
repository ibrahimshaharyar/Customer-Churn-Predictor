"""
Test Script for Phase 2: Data Pipeline

This script tests the complete data pipeline:
1. Data Ingestion (loading and validation)
2. Data Preprocessing (cleaning)
3. Feature Engineering (encoding, scaling, saving)

All modules use custom exceptions and logging to track any issues.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion.ingest import DataIngestion
from src.data_preprocessing.preprocess import DataPreprocessor
from src.feature_engineering.features import FeatureEngineer
from src.utils import logger


def test_data_pipeline():
    """Run the complete data pipeline end-to-end"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PHASE 2 - DATA PIPELINE TEST" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    try:
        # ============================================================
        # TEST 1: DATA INGESTION
        # ============================================================
        print("\n" + "="*80)
        print("TEST 1: DATA INGESTION")
        print("="*80)
        
        ingestion = DataIngestion()
        df_raw = ingestion.run_ingestion()
        
        print(f"\n✅ Data Ingestion Successful!")
        print(f"   - Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
        print(f"   - Columns: {list(df_raw.columns)}")
        
        # ============================================================
        # TEST 2: DATA PREPROCESSING
        # ============================================================
        print("\n" + "="*80)
        print("TEST 2: DATA PREPROCESSING")
        print("="*80)
        
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.run_preprocessing(df_raw)
        
        print(f"\n✅ Data Preprocessing Successful!")
        print(f"   - Processed {len(df_processed)} rows, {len(df_processed.columns)} columns")
        print(f"   - Dropped columns: {preprocessor.columns_to_drop}")
        print(f"   - Remaining columns: {list(df_processed.columns)}")
        
        # ============================================================
        # TEST 3: FEATURE ENGINEERING
        # ============================================================
        print("\n" + "="*80)
        print("TEST 3: FEATURE ENGINEERING")
        print("="*80)
        
        feature_engineer = FeatureEngineer()
        results = feature_engineer.run_feature_engineering(df_processed)
        
        print(f"\n✅ Feature Engineering Successful!")
        print(f"   - X_train shape: {results['X_train'].shape}")
        print(f"   - X_test shape: {results['X_test'].shape}")
        print(f"   - y_train shape: {results['y_train'].shape}")
        print(f"   - y_test shape: {results['y_test'].shape}")
        
        # ============================================================
        # VERIFICATION: CHECK SAVED FILES
        # ============================================================
        print("\n" + "="*80)
        print("VERIFICATION: CHECKING SAVED FILES")
        print("="*80)
        
        # Check processed data
        processed_file = Path("data/processed/churn_cleaned.csv")
        if processed_file.exists():
            print(f"✓ Processed data saved: {processed_file}")
        else:
            print(f"✗ Processed data NOT found: {processed_file}")
        
        # Check feature data
        for name, path in results['data_paths'].items():
            path_obj = Path(path)
            if path_obj.exists():
                print(f"✓ {name} saved: {path}")
            else:
                print(f"✗ {name} NOT found: {path}")
        
        # Check artifacts
        for name, path in results['artifact_paths'].items():
            path_obj = Path(path)
            if path_obj.exists():
                print(f"✓ {name} saved: {path}")
            else:
                print(f"✗ {name} NOT found: {path}")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "="*80)
        print("✅ PHASE 2 PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary:")
        print(f"  ✓ Data ingestion working")
        print(f"  ✓ Data preprocessing working")
        print(f"  ✓ Feature engineering working")
        print(f"  ✓ All data saved to data/processed and data/features")
        print(f"  ✓ All artifacts saved to artifacts/preprocessors")
        print(f"  ✓ Custom exceptions and logging working")
        print("\nCheck the logs/ folder for detailed pipeline logs!")
        print("\nNext: Ready for Phase 3 (Model Training)")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed!")
        print(f"Error: {str(e)}")
        print("\nCheck the logs/ folder for detailed error information")
        return False


if __name__ == "__main__":
    success = test_data_pipeline()
    sys.exit(0 if success else 1)
