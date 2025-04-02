"""
Feature engineering for financial transaction data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

class TransactionFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for extracting transaction-based features.
    """
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.models_dir = self.project_root / "src" / "models"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Make a copy to avoid modifying the original DataFrame
        X_transformed = X.copy()
        
        # Print columns for debugging
        print(f"Columns in TransactionFeatureExtractor: {X_transformed.columns.tolist()}")
        
        # First, convert date columns to datetime if they exist
        date_columns = ['TransactionDate', 'PreviousTransactionDate']
        existing_date_columns = []
        
        for col in date_columns:
            if col in X_transformed.columns:
                try:
                    # Try to convert to datetime
                    X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
                    existing_date_columns.append(col)
                    print(f"Successfully converted {col} to datetime")
                except Exception as e:
                    print(f"Error converting {col} to datetime: {e}")
                    # If conversion fails, drop the column
                    X_transformed = X_transformed.drop(columns=[col], errors='ignore')
        
        # Extract time-based features if TransactionDate exists
        if 'TransactionDate' in existing_date_columns:
            # Check if there are any non-null values
            if not X_transformed['TransactionDate'].isna().all():
                X_transformed['TransactionHour'] = X_transformed['TransactionDate'].dt.hour
                X_transformed['TransactionDayOfWeek'] = X_transformed['TransactionDate'].dt.dayofweek
                X_transformed['TransactionMonth'] = X_transformed['TransactionDate'].dt.month
                X_transformed['IsWeekend'] = X_transformed['TransactionDayOfWeek'].isin([5, 6]).astype(int)
                
                # Time periods (morning, afternoon, evening, night)
                X_transformed['TimePeriod'] = pd.cut(
                    X_transformed['TransactionHour'],
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night', 'Morning', 'Afternoon', 'Evening']
                )
                print("Added time-based features")
        
        # Calculate time since previous transaction (in hours) if both date columns exist
        if all(col in existing_date_columns for col in ['TransactionDate', 'PreviousTransactionDate']):
            # Check if there are any non-null values in both columns
            if not (X_transformed['TransactionDate'].isna().all() or X_transformed['PreviousTransactionDate'].isna().all()):
                try:
                    X_transformed['HoursSincePreviousTransaction'] = (
                        X_transformed['TransactionDate'] - X_transformed['PreviousTransactionDate']
                    ).dt.total_seconds() / 3600
                    
                    # Replace negative values (due to data errors) with median
                    median_hours = X_transformed['HoursSincePreviousTransaction'].median()
                    X_transformed['HoursSincePreviousTransaction'] = X_transformed['HoursSincePreviousTransaction'].clip(lower=0, upper=None)
                    X_transformed['HoursSincePreviousTransaction'] = X_transformed['HoursSincePreviousTransaction'].fillna(median_hours)
                    
                    # Flag for unusual transaction timing
                    hours_std = X_transformed['HoursSincePreviousTransaction'].std()
                    hours_mean = X_transformed['HoursSincePreviousTransaction'].mean()
                    X_transformed['UnusualTransactionTiming'] = (
                        X_transformed['HoursSincePreviousTransaction'] > (hours_mean + 2 * hours_std)
                    ).astype(int)
                    print("Added time difference features")
                except Exception as e:
                    print(f"Error calculating time difference: {e}")
        
        # Transaction amount features
        if 'TransactionAmount' in X_transformed.columns and 'AccountBalance' in X_transformed.columns:
            # Ratio of transaction amount to account balance
            X_transformed['TransactionToBalanceRatio'] = X_transformed['TransactionAmount'] / X_transformed['AccountBalance'].replace(0, np.nan)
            X_transformed['TransactionToBalanceRatio'] = X_transformed['TransactionToBalanceRatio'].fillna(0).replace([np.inf, -np.inf], 0)
            
            # Flag for high-value transactions (top 10%)
            high_value_threshold = X_transformed['TransactionAmount'].quantile(0.9)
            X_transformed['HighValueTransaction'] = (X_transformed['TransactionAmount'] > high_value_threshold).astype(int)
            print("Added transaction amount features")
        
        # Login attempts feature
        if 'LoginAttempts' in X_transformed.columns:
            X_transformed['HighLoginAttempts'] = (X_transformed['LoginAttempts'] > 1).astype(int)
            print("Added login attempts feature")
        
        # Drop original date columns as they can't be used directly by models
        X_transformed = X_transformed.drop(columns=existing_date_columns, errors='ignore')
        
        # Drop non-numeric columns that won't be used by the model
        cols_to_drop = ['s.no', 's_no', 'TransactionID', 'AccountID', 'DeviceID', 'IP_Address', 'MerchantID']
        X_transformed = X_transformed.drop(columns=[col for col in cols_to_drop if col in X_transformed.columns], errors='ignore')
        
        # Print final columns for debugging
        print(f"Final columns after feature extraction: {X_transformed.columns.tolist()}")
        
        return X_transformed

class FeatureEngineeringPipeline:
    """
    Pipeline for feature engineering on financial transaction data.
    """
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.models_dir = self.project_root / "src" / "models"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.pipeline = None
        
    def fit(self, X_train, y_train=None):
        """
        Fit the feature engineering pipeline to the training data.
        
        Args:
            X_train: Training features.
            y_train: Training target (optional).
            
        Returns:
            self: The fitted pipeline.
        """
        # Create the preprocessing pipeline based on the actual columns in X_train
        self.pipeline = self.create_preprocessing_pipeline(X_train)
        
        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using the fitted pipeline.
        
        Args:
            X: Data to transform.
            
        Returns:
            numpy.ndarray: Transformed features.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet. Call fit() first.")
        
        return self.pipeline.transform(X)
    
    def fit_transform(self, X_train, y_train=None):
        """
        Fit the pipeline and transform the training data.
        
        Args:
            X_train: Training features.
            y_train: Training target (optional).
            
        Returns:
            numpy.ndarray: Transformed training features.
        """
        return self.fit(X_train, y_train).transform(X_train)
    
    def create_preprocessing_pipeline(self, X):
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            X: Sample DataFrame to determine column types.
            
        Returns:
            sklearn.pipeline.Pipeline: Preprocessing pipeline.
        """
        print(f"Creating preprocessing pipeline with columns: {X.columns.tolist()}")
        
        # Identify numeric and categorical columns from the actual data
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove any columns we don't want to include
        cols_to_exclude = ['s.no', 's_no', 'TransactionID', 'AccountID', 'DeviceID', 'IP_Address', 'MerchantID', 'Fraud', 'TransactionDate', 'PreviousTransactionDate']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude and col in X.columns]
        categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude and col in X.columns]
        
        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        # Create transformers for each column type
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Only create categorical transformer if there are categorical columns
        transformers = []
        if numeric_cols:
            transformers.append(('num', numeric_transformer, numeric_cols))
        
        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        # If no transformers, create a passthrough transformer
        if not transformers:
            print("No columns to transform, using passthrough")
            preprocessor = 'passthrough'
        else:
            # Create the column transformer
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'  # Drop any columns not specified
            )
        
        # Create the complete pipeline with custom feature extraction
        pipeline = Pipeline(steps=[
            ('feature_extractor', TransactionFeatureExtractor()),
            ('preprocessor', preprocessor)
        ])
        
        return pipeline

if __name__ == "__main__":
    # This is a demonstration of how to use the feature engineering pipeline
    from pathlib import Path
    import pandas as pd
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Load the data
    data_dir = project_root / "src" / "data" / "processed"
    
    try:
        # Try to load the latest version
        with open(data_dir / "latest_version.txt", "r") as f:
            timestamp = f.read().strip()
            
        X_train = pd.read_csv(data_dir / f"X_train_{timestamp}.csv")
        y_train = pd.read_csv(data_dir / f"y_train_{timestamp}.csv").iloc[:, 0]
        
        # Create and fit the feature engineering pipeline
        pipeline = FeatureEngineeringPipeline()
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        
        print(f"Original shape: {X_train.shape}")
        print(f"Transformed shape: {X_train_transformed.shape}")
        print("Feature engineering completed successfully!")
        
    except FileNotFoundError:
        print("Processed data not found. Run the data ingestion pipeline first.")
