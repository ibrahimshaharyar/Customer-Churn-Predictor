"""
Data Preprocessing Module for Banking Customer Churn Pipeline

This module handles cleaning and preprocessing the raw data:
- Removes unnecessary columns
- Handles missing values (if any)
- Validates processed data
- Saves cleaned data to processed folder

Author: Banking Churn MLOps Team
"""

import sys
import pandas as pd
from pathlib import Path
from src.utils import logger, CustomException


class DataPreprocessor:
    """
    Handles data cleaning and preprocessing operations
    
    This class is responsible for:
    - Dropping unnecessary columns (RowNumber, CustomerId, Surname)
    - Handling missing values
    - Saving processed data
    """
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        logger.info("DataPreprocessor initialized")
        
        # Define columns to drop (not useful for prediction)
        self.columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        
        # Define processed data path
        self.processed_dir = Path("data/processed")
        self.processed_file = self.processed_dir / "churn_cleaned.csv"
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that are not useful for prediction
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: DataFrame without unnecessary columns
        """
        try:
            logger.info(f"Dropping unnecessary columns: {self.columns_to_drop}")
            
            # Check if columns exist before dropping
            existing_cols = [col for col in self.columns_to_drop if col in df.columns]
            missing_cols = [col for col in self.columns_to_drop if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"⚠ Columns not found (skipping): {missing_cols}")
            
            # Drop columns
            df_clean = df.drop(columns=existing_cols)
            
            logger.info(f"✓ Dropped {len(existing_cols)} columns")
            logger.info(f"✓ Remaining columns: {list(df_clean.columns)}")
            logger.info(f"✓ New shape: {df_clean.shape}")
            
            return df_clean
            
        except Exception as e:
            logger.error("Failed to drop columns")
            raise CustomException(e, sys)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Strategy:
        - Check for missing values
        - Log if any are found
        - For this dataset, we expect no missing values
        
        Args:
            df (pd.DataFrame): DataFrame to check
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        try:
            logger.info("Checking for missing values...")
            
            # Get missing value counts
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            
            if total_missing > 0:
                logger.warning(f"⚠ Found {total_missing} missing values")
                logger.warning(f"Missing value breakdown:\n{missing_counts[missing_counts > 0]}")
                
                # For now, we'll drop rows with missing values
                # In production, you might want different strategies per column
                df_clean = df.dropna()
                logger.info(f"✓ Dropped rows with missing values")
                logger.info(f"✓ Rows before: {len(df)}, after: {len(df_clean)}")
                
                return df_clean
            else:
                logger.info("✓ No missing values found")
                return df
                
        except Exception as e:
            logger.error("Failed to handle missing values")
            raise CustomException(e, sys)
    
    def validate_processed_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the processed data meets requirements
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            bool: True if validation passes
        """
        try:
            logger.info("Validating processed data...")
            
            # Check 1: DataFrame is not empty
            if len(df) == 0:
                raise ValueError("Processed DataFrame is empty!")
            
            # Check 2: Required columns present
            required_cols = [
                'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Exited'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Required columns missing: {missing_cols}")
            
            # Check 3: Target column has correct values (0 or 1)
            if 'Exited' in df.columns:
                unique_values = df['Exited'].unique()
                if not set(unique_values).issubset({0, 1}):
                    raise ValueError(f"Exited column should only contain 0 or 1, found: {unique_values}")
            
            logger.info("✓ All validation checks passed")
            return True
            
        except Exception as e:
            logger.error("Processed data validation failed")
            raise CustomException(e, sys)
    
    def save_processed_data(self, df: pd.DataFrame) -> str:
        """
        Save processed data to CSV file
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            str: Path to saved file
        """
        try:
            logger.info(f"Saving processed data to: {self.processed_file}")
            
            # Create directory if it doesn't exist
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(self.processed_file, index=False)
            
            logger.info(f"✓ Processed data saved successfully")
            logger.info(f"✓ Saved {len(df)} rows, {len(df.columns)} columns")
            
            return str(self.processed_file)
            
        except Exception as e:
            logger.error(f"Failed to save processed data to {self.processed_file}")
            raise CustomException(e, sys)
    
    def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline
        
        This is the main method that:
        1. Drops unnecessary columns
        2. Handles missing values
        3. Validates processed data
        4. Saves processed data to file
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            logger.info("="*80)
            logger.info("STARTING DATA PREPROCESSING PIPELINE")
            logger.info("="*80)
            
            # Step 1: Drop unnecessary columns
            df_clean = self.drop_unnecessary_columns(df)
            
            # Step 2: Handle missing values
            df_clean = self.handle_missing_values(df_clean)
            
            # Step 3: Validate processed data
            self.validate_processed_data(df_clean)
            
            # Step 4: Save processed data
            save_path = self.save_processed_data(df_clean)
            
            logger.info("="*80)
            logger.info("✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info(f"✅ Processed data saved to: {save_path}")
            logger.info("="*80)
            
            return df_clean
            
        except Exception as e:
            logger.error("Data preprocessing pipeline failed")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # First, load the data
    from src.data_ingestion.ingest import DataIngestion
    
    ingestion = DataIngestion()
    df_raw = ingestion.run_ingestion()
    
    # Then preprocess it
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.run_preprocessing(df_raw)
    
    print(f"\n✅ Data preprocessing complete!")
    print(f"Processed DataFrame shape: {df_processed.shape}")
    print(f"\nRemaining columns: {list(df_processed.columns)}")
