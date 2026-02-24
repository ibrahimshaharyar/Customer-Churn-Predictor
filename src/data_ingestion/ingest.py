"""
Data Ingestion Module for Banking Customer Churn Pipeline

This module handles loading raw data from CSV files and validating the schema.
It uses logging and custom exceptions to track any issues during data loading.

Author: Banking Churn MLOps Team
"""

import sys
import pandas as pd
from pathlib import Path
from src.utils import logger, CustomException


class DataIngestion:
    """
    Handles loading and validating raw customer churn data
    
    This class is responsible for:
    - Loading raw CSV data
    - Validating data schema (required columns)
    - Checking basic data quality
    """
    
    def __init__(self, data_path: str = "data/raw/Churn_Modelling.csv"):
        """
        Initialize DataIngestion
        
        Args:
            data_path (str): Path to the raw CSV file
        """
        self.data_path = data_path
        logger.info(f"DataIngestion initialized with path: {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        
        Returns:
            pd.DataFrame: Loaded raw data
            
        Raises:
            CustomException: If file not found or cannot be read
        """
        try:
            logger.info(f"Loading data from: {self.data_path}")
            
            # Check if file exists
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found at: {self.data_path}")
            
            # Load CSV file
            df = pd.read_csv(self.data_path)
            
            logger.info(f"✓ Data loaded successfully. Shape: {df.shape}")
            logger.info(f"✓ Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}")
            raise CustomException(e, sys)
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns are present
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if schema is valid
            
        Raises:
            CustomException: If required columns are missing
        """
        try:
            logger.info("Validating data schema...")
            
            # Define required columns
            required_columns = [
                'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 
                'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                'EstimatedSalary', 'Exited'
            ]
            
            # Get actual columns
            actual_columns = list(df.columns)
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in actual_columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info("✓ Schema validation passed - all required columns present")
            return True
            
        except Exception as e:
            logger.error("Schema validation failed")
            raise CustomException(e, sys)
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Perform basic data quality checks
        
        Args:
            df (pd.DataFrame): DataFrame to check
            
        Returns:
            dict: Data quality report
        """
        try:
            logger.info("Performing data quality checks...")
            
            quality_report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Log quality metrics
            logger.info(f"✓ Total rows: {quality_report['total_rows']}")
            logger.info(f"✓ Total columns: {quality_report['total_columns']}")
            logger.info(f"✓ Duplicate rows: {quality_report['duplicate_rows']}")
            
            # Check for missing values
            missing_count = sum(quality_report['missing_values'].values())
            if missing_count > 0:
                logger.warning(f"⚠ Found {missing_count} missing values")
            else:
                logger.info("✓ No missing values found")
            
            return quality_report
            
        except Exception as e:
            logger.error("Data quality check failed")
            raise CustomException(e, sys)
    
    def run_ingestion(self) -> pd.DataFrame:
        """
        Run the complete data ingestion pipeline
        
        This is the main method that:
        1. Loads the data
        2. Validates the schema
        3. Checks data quality
        4. Returns the validated DataFrame
        
        Returns:
            pd.DataFrame: Validated raw data
        """
        try:
            logger.info("="*80)
            logger.info("STARTING DATA INGESTION PIPELINE")
            logger.info("="*80)
            
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Validate schema
            self.validate_schema(df)
            
            # Step 3: Check data quality
            quality_report = self.check_data_quality(df)
            
            logger.info("="*80)
            logger.info("✅ DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            return df
            
        except Exception as e:
            logger.error("Data ingestion pipeline failed")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Create ingestion instance
    ingestion = DataIngestion()
    
    # Run ingestion pipeline
    df = ingestion.run_ingestion()
    
    print(f"\n✅ Data ingestion complete!")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
