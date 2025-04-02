"""
Create SQL database for financial transaction data.
"""
import os
import pandas as pd
import sqlite3
from pathlib import Path
import numpy as np

def create_database():
    """
    Create SQLite database from CSV data and set up the necessary tables.
    """
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Path to the CSV file
    csv_path = project_root / "bank_transactions_data_2.csv"
    
    # Path to the SQLite database
    db_path = project_root / "src" / "data" / "financial_transactions.db"
    
    print(f"Creating database at {db_path}...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns found: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Clean column names (remove spaces and special characters)
    df.columns = [col.replace(' ', '_').replace('.', '_') for col in df.columns]
    
    # Ensure date columns are properly formatted
    date_columns = ['TransactionDate', 'PreviousTransactionDate']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Converted {col} to datetime format")
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")
    
    # Check if Fraud column exists, if not, generate synthetic fraud data
    if 'Fraud' not in df.columns:
        print("Fraud column not found in the dataset. Generating synthetic fraud labels...")
        # Generate synthetic fraud labels (about 2% fraud rate)
        np.random.seed(42)
        fraud_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
        df['Fraud'] = 0
        df.loc[fraud_indices, 'Fraud'] = 1
        print(f"Generated {len(fraud_indices)} synthetic fraud cases ({len(fraud_indices)/len(df)*100:.2f}% of data)")
    else:
        # Ensure Fraud column is binary (0 or 1)
        df['Fraud'] = df['Fraud'].astype(int)
        print(f"Fraud column found with values: {df['Fraud'].unique()}")
        print(f"Fraud ratio: {df['Fraud'].mean():.4f}")
    
    # Connect to the SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(str(db_path))
    
    # Create the transactions table
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Create indexes for faster querying
    conn.execute('CREATE INDEX IF NOT EXISTS idx_account_id ON transactions (AccountID)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions (TransactionDate)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_fraud ON transactions (Fraud)')
    
    # Verify the data was loaded correctly
    result = conn.execute('SELECT COUNT(*) FROM transactions').fetchone()
    print(f"Successfully loaded {result[0]} transactions into the database.")
    
    # Count fraud cases
    fraud_count = conn.execute('SELECT COUNT(*) FROM transactions WHERE Fraud = 1').fetchone()[0]
    print(f"Number of fraud cases in the database: {fraud_count} ({fraud_count/result[0]*100:.2f}% of data)")
    
    # Create a view for fraud transactions
    conn.execute('''
    CREATE VIEW IF NOT EXISTS fraud_transactions AS
    SELECT * FROM transactions
    WHERE Fraud = 1
    ''')
    
    # Create a view for legitimate transactions
    conn.execute('''
    CREATE VIEW IF NOT EXISTS legitimate_transactions AS
    SELECT * FROM transactions
    WHERE Fraud = 0
    ''')
    
    # Close the connection
    conn.close()
    
    print("Database creation completed successfully!")

if __name__ == "__main__":
    create_database()
