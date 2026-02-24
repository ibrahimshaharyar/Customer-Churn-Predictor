"""
Feature Engineering Module for Banking Customer Churn Pipeline

This module handles feature engineering operations:
- Encoding categorical variables (Geography, Gender)
- Splitting features and target
- Train/test split
- SMOTE for handling class imbalance (NEW!)
- Feature scaling
- Saving feature artifacts

Author: Banking Churn MLOps Team
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils import logger, CustomException


class FeatureEngineer:
    """
    Handles feature engineering and transformation operations
    
    This class is responsible for:
    - Encoding categorical variables (Geography, Gender)
    - Splitting features (X) and target (y)
    - Creating train/test splits
    - Applying SMOTE for class balance (optional)
    - Scaling features
    - Saving all artifacts for reproducibility
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, use_smote: bool = True):
        """
        Initialize FeatureEngineer
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            use_smote (bool): Whether to use SMOTE for balancing training data (default: True)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.use_smote = use_smote
        
        # Initialize encoders and scaler (will be fit during processing)
        self.label_encoder_geo = LabelEncoder()
        self.label_encoder_gender = LabelEncoder()
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=random_state) if use_smote else None
        
        # Define paths
        self.features_dir = Path("data/features")
        self.artifacts_dir = Path("artifacts/preprocessors")
        
        logger.info(f"FeatureEngineer initialized (test_size={test_size}, random_state={random_state}, use_smote={use_smote})")
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder
        
        Categorical columns:
        - Geography: France, Germany, Spain → 0, 1, 2
        - Gender: Female, Male → 0, 1
        
        Args:
            df (pd.DataFrame): DataFrame with categorical columns
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables
        """
        try:
            logger.info("Encoding categorical variables...")
            
            # Make a copy to avoid modifying original
            df_encoded = df.copy()
            
            # Encode Geography
            if 'Geography' in df.columns:
                df_encoded['Geography'] = self.label_encoder_geo.fit_transform(df['Geography'])
                geo_mapping = dict(zip(
                    self.label_encoder_geo.classes_, 
                    self.label_encoder_geo.transform(self.label_encoder_geo.classes_)
                ))
                logger.info(f"✓ Geography encoded: {geo_mapping}")
            else:
                raise ValueError("Geography column not found in DataFrame")
            
            # Encode Gender
            if 'Gender' in df.columns:
                df_encoded['Gender'] = self.label_encoder_gender.fit_transform(df['Gender'])
                gender_mapping = dict(zip(
                    self.label_encoder_gender.classes_,
                    self.label_encoder_gender.transform(self.label_encoder_gender.classes_)
                ))
                logger.info(f"✓ Gender encoded: {gender_mapping}")
            else:
                raise ValueError("Gender column not found in DataFrame")
            
            logger.info("✓ Categorical encoding complete")
            return df_encoded
            
        except Exception as e:
            logger.error("Failed to encode categorical variables")
            raise CustomException(e, sys)
    
    def split_features_target(self, df: pd.DataFrame) -> tuple:
        """
        Split DataFrame into features (X) and target (y)
        
        Args:
            df (pd.DataFrame): Full dataframe with target column
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        try:
            logger.info("Splitting features and target...")
            
            # Check if Exited column exists
            if 'Exited' not in df.columns:
                raise ValueError("Target column 'Exited' not found in DataFrame")
            
            # Split features and target
            X = df.drop('Exited', axis=1)
            y = df['Exited']
            
            logger.info(f"✓ Features (X) shape: {X.shape}")
            logger.info(f"✓ Target (y) shape: {y.shape}")
            logger.info(f"✓ Feature columns: {list(X.columns)}")
            
            # Check target distribution
            target_dist = y.value_counts()
            logger.info(f"✓ Target distribution:\n{target_dist}")
            
            return X, y
            
        except Exception as e:
            logger.error("Failed to split features and target")
            raise CustomException(e, sys)
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Create train and test splits
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info(f"Creating train/test split (test_size={self.test_size})...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y  # Maintain class distribution
            )
            
            logger.info(f"✓ Training set: {X_train.shape[0]} samples")
            logger.info(f"✓ Test set: {X_test.shape[0]} samples")
            
            # Log target distribution in train and test
            logger.info(f"✓ Train target distribution:\n{y_train.value_counts()}")
            logger.info(f"✓ Test target distribution:\n{y_test.value_counts()}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error("Failed to create train/test split")
            raise CustomException(e, sys)
    
    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """
        Apply SMOTE to balance training data
        
        Important: Only apply to training data, NEVER to test data!
        This creates synthetic samples of the minority class (churned customers)
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            tuple: (X_train_balanced, y_train_balanced)
        """
        try:
            if not self.use_smote:
                logger.info("SMOTE disabled, returning original training data")
                return X_train, y_train
            
            logger.info("Applying SMOTE to balance training data...")
            logger.info(f"Original training set distribution:\n{y_train.value_counts()}")
            
            # Apply SMOTE
            X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
            
            # Convert back to DataFrame/Series to preserve column names
            X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
            y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)
            
            logger.info(f"✓ SMOTE applied successfully")
            logger.info(f"✓ Balanced training set: {X_train_balanced.shape[0]} samples")
            logger.info(f"✓ New distribution:\n{y_train_balanced.value_counts()}")
            
            # Calculate increase
            original_samples = len(y_train)
            new_samples = len(y_train_balanced)
            increase = new_samples - original_samples
            logger.info(f"✓ Added {increase} synthetic samples ({increase/original_samples*100:.1f}% increase)")
            
            return X_train_balanced, y_train_balanced
            
        except Exception as e:
            logger.error("Failed to apply SMOTE")
            raise CustomException(e, sys)
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Scale features using StandardScaler
        
        Important: Fit scaler on training data only, then transform both train and test
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) as numpy arrays
        """
        try:
            logger.info("Scaling features with StandardScaler...")
            
            # Fit scaler on training data only
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Transform test data using fitted scaler
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"✓ Training set scaled: {X_train_scaled.shape}")
            logger.info(f"✓ Test set scaled: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logger.error("Failed to scale features")
            raise CustomException(e, sys)
    
    def save_feature_data(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Save feature data to CSV files
        
        Args:
            X_train: Training features (can be DataFrame or array)
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            dict: Paths to saved files
        """
        try:
            logger.info("Saving feature data...")
            
            # Create features directory
            self.features_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert arrays to DataFrames if needed for better readability
            if not isinstance(X_train, pd.DataFrame):
                # If it's a numpy array, we need column names
                logger.info("Converting numpy arrays to DataFrames for saving...")
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)
            
            # Save files
            paths = {
                'X_train': self.features_dir / 'X_train.csv',
                'X_test': self.features_dir / 'X_test.csv',
                'y_train': self.features_dir / 'y_train.csv',
                'y_test': self.features_dir / 'y_test.csv'
            }
            
            X_train.to_csv(paths['X_train'], index=False)
            X_test.to_csv(paths['X_test'], index=False)
            y_train.to_csv(paths['y_train'], index=False)
            y_test.to_csv(paths['y_test'], index=False)
            
            logger.info(f"✓ Saved X_train to {paths['X_train']}")
            logger.info(f"✓ Saved X_test to {paths['X_test']}")
            logger.info(f"✓ Saved y_train to {paths['y_train']}")
            logger.info(f"✓ Saved y_test to {paths['y_test']}")
            
            return {k: str(v) for k, v in paths.items()}
            
        except Exception as e:
            logger.error("Failed to save feature data")
            raise CustomException(e, sys)
    
    def save_artifacts(self) -> dict:
        """
        Save preprocessing artifacts (encoders and scaler) for future use
        
        Returns:
            dict: Paths to saved artifacts
        """
        try:
            logger.info("Saving preprocessing artifacts...")
            
            # Create artifacts directory
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Define artifact paths
            paths = {
                'scaler': self.artifacts_dir / 'scaler.pkl',
                'label_encoder_geo': self.artifacts_dir / 'label_encoder_geo.pkl',
                'label_encoder_gender': self.artifacts_dir / 'label_encoder_gender.pkl'
            }
            
            # Save artifacts
            joblib.dump(self.scaler, paths['scaler'])
            joblib.dump(self.label_encoder_geo, paths['label_encoder_geo'])
            joblib.dump(self.label_encoder_gender, paths['label_encoder_gender'])
            
            logger.info(f"✓ Saved scaler to {paths['scaler']}")
            logger.info(f"✓ Saved Geography encoder to {paths['label_encoder_geo']}")
            logger.info(f"✓ Saved Gender encoder to {paths['label_encoder_gender']}")
            
            return {k: str(v) for k, v in paths.items()}
            
        except Exception as e:
            logger.error("Failed to save preprocessing artifacts")
            raise CustomException(e, sys)
    
    def run_feature_engineering(self, df: pd.DataFrame) -> dict:
        """
        Run the complete feature engineering pipeline
        
        This is the main method that:
        1. Encodes categorical variables
        2. Splits features and target
        3. Creates train/test split
        4. Applies SMOTE for class balance (if enabled)
        5. Scales features
        6. Saves all data and artifacts
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            dict: Contains all processed data and file paths
        """
        try:
            logger.info("="*80)
            logger.info("STARTING FEATURE ENGINEERING PIPELINE")
            logger.info("="*80)
            
            # Step 1: Encode categorical variables
            df_encoded = self.encode_categorical_variables(df)
            
            # Step 2: Split features and target
            X, y = self.split_features_target(df_encoded)
            
            # Step 3: Create train/test split
            X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)
            
            # Step 4: Apply SMOTE to balance training data (if enabled)
            X_train, y_train = self.apply_smote(X_train, y_train)
            
            # Step 5: Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            # Step 6: Save feature data (save the scaled versions)
            # Convert scaled arrays back to DataFrames with original column names
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            data_paths = self.save_feature_data(
                X_train_scaled_df, X_test_scaled_df, y_train, y_test
            )
            
            # Step 7: Save preprocessing artifacts
            artifact_paths = self.save_artifacts()
            
            logger.info("="*80)
            logger.info("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            # Return everything
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'data_paths': data_paths,
                'artifact_paths': artifact_paths
            }
            
        except Exception as e:
            logger.error("Feature engineering pipeline failed")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Load processed data
    from src.data_ingestion.ingest import DataIngestion
    from src.data_preprocessing.preprocess import DataPreprocessor
    
    # Step 1: Load and preprocess data
    ingestion = DataIngestion()
    df_raw = ingestion.run_ingestion()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.run_preprocessing(df_raw)
    
    # Step 2: Feature engineering
    feature_engineer = FeatureEngineer()
    results = feature_engineer.run_feature_engineering(df_processed)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"X_train shape: {results['X_train'].shape}")
    print(f"X_test shape: {results['X_test'].shape}")
    print(f"\nData saved to: {results['data_paths']}")
    print(f"Artifacts saved to: {results['artifact_paths']}")
