# Financial Fraud Detection System

A comprehensive anomaly detection system for financial fraud detection using a combination of supervised (XGBoost, Random Forest) and unsupervised (Isolation Forest) techniques.

## Project Structure

```
├── src/
│   ├── data/           # Data ingestion, processing, and database operations
│   ├── features/       # Feature engineering and transformation pipelines
│   ├── models/         # Model training, evaluation, and prediction
│   ├── utils/          # Utility functions and helpers
│   ├── api/            # API for real-time fraud detection
│   ├── monitoring/     # Model monitoring and performance tracking
│   ├── config/         # Configuration files
│   └── notebooks/      # Jupyter notebooks for exploration and analysis
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Initialize the SQL database:
   ```
   python src/data/create_database.py
   ```

3. Run the data ingestion pipeline:
   ```
   python src/data/data_ingestion.py
   ```

4. Train the models:
   ```
   python src/models/train_models.py
   ```

5. Start the real-time fraud detection API:
   ```
   python src/api/app.py
   ```

## Key Components

1. **Data Ingestion & Preprocessing**: Load and clean transaction data from SQL databases and real-time streams.
2. **Feature Engineering**: Create time-based and category-based features for improved model sensitivity.
3. **Model Training**: Combine supervised (XGBoost, Random Forest) and unsupervised (Isolation Forest) techniques.
4. **Ensemble Scoring**: Implement a score fusion mechanism for real-time detection.
5. **Model Monitoring**: Track model performance and detect data drift.
6. **A/B Testing**: Compare new integrated approach against legacy systems.
7. **Deployment**: Real-time data ingestion and model inference in production.

## Dashboard

Access the fraud detection dashboard at `http://localhost:8050/` after starting the application.
