"""
Data ingestion pipeline for financial transaction data.
"""
import os
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestionPipeline:
    """
    Pipeline for ingesting financial transaction data from various sources.
    """
    def __init__(self, db_path=None):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            db_path: Path to the SQLite database. If None, uses the default path.
        """
        # Get the project root directory
        self.project_root = Path(__file__).resolve().parents[2]
        
        # Set the database path
        if db_path is None:
            self.db_path = self.project_root / "src" / "data" / "financial_transactions.db"
        else:
            self.db_path = Path(db_path)
            
        # Create data directories if they don't exist
        self.processed_data_dir = self.project_root / "src" / "data" / "processed"
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
        
    def connect_to_db(self):
        """
        Connect to the SQLite database.
        
        Returns:
            sqlite3.Connection: Database connection object.
        """
        return sqlite3.connect(str(self.db_path))
    
    def load_data_from_db(self):
        """
        Load data from the SQLite database.
        
        Returns:
            pandas.DataFrame: Loaded data.
        """
        # Connect to the database
        conn = sqlite3.connect(str(self.db_path))
        
        # Query the data
        query = "SELECT * FROM transactions"
        df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        # Ensure the Fraud column is binary (0 or 1)
        if 'Fraud' in df.columns:
            df['Fraud'] = df['Fraud'].astype(int)
            print(f"Fraud column values: {df['Fraud'].unique()}")
            print(f"Fraud ratio: {df['Fraud'].mean()}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean the data by handling missing values, removing duplicates, etc.
        
        Args:
            df: DataFrame to clean.
            
        Returns:
            pandas.DataFrame: Cleaned DataFrame.
        """
        # Make a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Print columns before cleaning
        print(f"Columns before cleaning: {df_clean.columns.tolist()}")
        print(f"Fraud column exists before cleaning: {'Fraud' in df_clean.columns}")
        print(f"Fraud values before cleaning: {df_clean['Fraud'].value_counts().to_dict() if 'Fraud' in df_clean.columns else 'N/A'}")
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # For categorical columns, fill missing values with the most frequent value
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
            else:
                # For numerical columns, fill missing values with the median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        # Print columns after cleaning
        print(f"Columns after cleaning: {df_clean.columns.tolist()}")
        print(f"Fraud column exists after cleaning: {'Fraud' in df_clean.columns}")
        print(f"Fraud values after cleaning: {df_clean['Fraud'].value_counts().to_dict() if 'Fraud' in df_clean.columns else 'N/A'}")
        
        return df_clean
    
    def split_data(self, df):
        """
        Split the data into training, validation, and test sets.
        
        Args:
            df: DataFrame to split.
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = df.drop(columns=['Fraud'])
        y = df['Fraud']
        
        # Check if we have any fraud cases
        fraud_count = sum(y == 1)
        print(f"Total number of fraud cases: {fraud_count}")
        print(f"Fraud ratio in full dataset: {fraud_count / len(y):.4f}")
        
        # If no fraud cases, generate synthetic ones
        if fraud_count == 0:
            print("No fraud cases found in the dataset. Generating synthetic fraud examples...")
            # Generate synthetic fraud examples (about 2% fraud rate)
            np.random.seed(42)
            fraud_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
            y.iloc[fraud_indices] = 1
            df.loc[fraud_indices, 'Fraud'] = 1
            print(f"Generated {len(fraud_indices)} synthetic fraud cases ({len(fraud_indices)/len(df)*100:.2f}% of data)")
        
        # Use stratified split to maintain class distribution
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp
        )
        
        # Verify the split
        print(f"Training set fraud cases: {sum(y_train == 1)}, ratio: {sum(y_train == 1) / len(y_train):.4f}")
        print(f"Validation set fraud cases: {sum(y_val == 1)}, ratio: {sum(y_val == 1) / len(y_val):.4f}")
        print(f"Test set fraud cases: {sum(y_test == 1)}, ratio: {sum(y_test == 1) / len(y_test):.4f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Save the processed data to disk.
        
        Args:
            X_train, X_val, X_test: Feature DataFrames for train, validation, and test sets.
            y_train, y_val, y_test: Target Series for train, validation, and test sets.
        """
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the data
        X_train.to_csv(self.processed_data_dir / f"X_train_{timestamp}.csv", index=False)
        X_val.to_csv(self.processed_data_dir / f"X_val_{timestamp}.csv", index=False)
        X_test.to_csv(self.processed_data_dir / f"X_test_{timestamp}.csv", index=False)
        
        y_train.to_csv(self.processed_data_dir / f"y_train_{timestamp}.csv", index=False)
        y_val.to_csv(self.processed_data_dir / f"y_val_{timestamp}.csv", index=False)
        y_test.to_csv(self.processed_data_dir / f"y_test_{timestamp}.csv", index=False)
        
        # Save the timestamp for reference
        with open(self.processed_data_dir / "latest_version.txt", "w") as f:
            f.write(timestamp)
            
        print(f"Processed data saved with version: {timestamp}")
        
    def run_pipeline(self):
        """
        Run the full data ingestion pipeline.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Starting data ingestion pipeline...")
        
        # Load data from database
        print("Loading data from database...")
        df = self.load_data_from_db()
        
        # Clean the data
        print("Cleaning data...")
        df_clean = self.clean_data(df)
        
        # Check fraud distribution
        print(f"After cleaning: Total samples: {len(df_clean)}, Fraud cases: {df_clean['Fraud'].sum()}")
        print(f"Fraud ratio after cleaning: {df_clean['Fraud'].mean():.4f}")
        
        # Split the data
        print("Splitting data into train, validation, and test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df_clean)
        
        # Save the processed data
        print("Saving processed data...")
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("Data ingestion pipeline completed successfully!")
        
        # Print some statistics
        print("\nData Statistics:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Fraud ratio in training set: {y_train.mean():.4f}")
        print(f"Fraud ratio in validation set: {y_val.mean():.4f}")
        print(f"Fraud ratio in test set: {y_test.mean():.4f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Run the data ingestion pipeline
    pipeline = DataIngestionPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run_pipeline()
