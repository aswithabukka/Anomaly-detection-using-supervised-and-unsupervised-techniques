"""
Real-time fraud detection API.
"""
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import custom modules
from src.features.feature_engineering import FeatureEngineeringPipeline
from src.models.ensemble import FraudDetectionEnsemble

# Create the FastAPI app
app = FastAPI(
    title="Financial Fraud Detection API",
    description="API for real-time fraud detection in financial transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the transaction model
class Transaction(BaseModel):
    TransactionID: str
    AccountID: str
    TransactionAmount: float
    TransactionDate: str
    TransactionType: str
    Location: str
    DeviceID: str
    IP_Address: Optional[str] = None
    MerchantID: Optional[str] = None
    Channel: str
    CustomerAge: Optional[int] = None
    CustomerOccupation: Optional[str] = None
    TransactionDuration: Optional[int] = None
    LoginAttempts: Optional[int] = None
    AccountBalance: Optional[float] = None
    PreviousTransactionDate: Optional[str] = None

class TransactionBatch(BaseModel):
    transactions: List[Transaction]

class FeedbackModel(BaseModel):
    TransactionID: str
    IsFraud: bool
    Feedback: Optional[str] = None

# Global variables for models
ensemble_model = None
feature_pipeline = None
feedback_data = []

def load_models():
    """
    Load the ensemble model and feature pipeline.
    """
    global ensemble_model, feature_pipeline
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "src" / "models"
    
    try:
        # Find the latest ensemble config
        ensemble_configs = sorted(models_dir.glob("ensemble_config_*.joblib"))
        if not ensemble_configs:
            raise FileNotFoundError("No ensemble config found")
        
        latest_config = ensemble_configs[-1]
        ensemble_config = joblib.load(latest_config)
        
        # Load the models
        models = {}
        for name, model_path in ensemble_config['models'].items():
            model_files = sorted(models_dir.glob(f"{name}_*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No {name} model found")
            models[name] = joblib.load(model_files[-1])
        
        # Load the feature pipeline
        try:
            with open(models_dir / "latest_feature_pipeline.txt", "r") as f:
                timestamp = f.read().strip()
            pipeline_path = models_dir / f"feature_pipeline_{timestamp}.joblib"
            feature_pipeline = joblib.load(pipeline_path)
        except FileNotFoundError:
            raise FileNotFoundError("Feature pipeline not found")
        
        # Create the ensemble model
        ensemble_model = FraudDetectionEnsemble(
            models=models,
            weights=ensemble_config['weights'],
            threshold=ensemble_config['threshold']
        )
        ensemble_model.feature_pipeline = feature_pipeline
        
        print("Models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """
    Load models on startup.
    """
    load_models()

@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": "Financial Fraud Detection API"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    if ensemble_model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {"status": "healthy"}

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction.
    """
    if ensemble_model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Make prediction
        result = ensemble_model.predict(None, df)
        
        # Format response
        response = {
            "TransactionID": transaction.TransactionID,
            "FraudPrediction": bool(result['prediction'][0]),
            "FraudScore": float(result['fraud_score'][0]),
            "ModelScores": {name: float(scores[0]) for name, scores in result['model_scores'].items()},
            "Timestamp": datetime.now().isoformat()
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_fraud_batch(batch: TransactionBatch):
    """
    Predict fraud for a batch of transactions.
    """
    if ensemble_model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert transactions to DataFrame
        transactions_dict = [t.dict() for t in batch.transactions]
        df = pd.DataFrame(transactions_dict)
        
        # Make predictions
        result = ensemble_model.predict(None, df)
        
        # Format response
        response = []
        for i, transaction in enumerate(batch.transactions):
            response.append({
                "TransactionID": transaction.TransactionID,
                "FraudPrediction": bool(result['prediction'][i]),
                "FraudScore": float(result['fraud_score'][i]),
                "ModelScores": {name: float(scores[i]) for name, scores in result['model_scores'].items()},
                "Timestamp": datetime.now().isoformat()
            })
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackModel, background_tasks: BackgroundTasks):
    """
    Submit feedback on a prediction.
    """
    global feedback_data
    
    # Add feedback to the list
    feedback_data.append({
        "TransactionID": feedback.TransactionID,
        "IsFraud": feedback.IsFraud,
        "Feedback": feedback.Feedback,
        "Timestamp": datetime.now().isoformat()
    })
    
    # Save feedback to file in the background
    background_tasks.add_task(save_feedback)
    
    return {"message": "Feedback received", "TransactionID": feedback.TransactionID}

def save_feedback():
    """
    Save feedback data to a file.
    """
    global feedback_data
    
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "src" / "data"
        feedback_file = data_dir / "feedback.json"
        
        # Create the directory if it doesn't exist
        data_dir.mkdir(exist_ok=True)
        
        # Load existing feedback if the file exists
        existing_feedback = []
        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                existing_feedback = json.load(f)
        
        # Append new feedback
        all_feedback = existing_feedback + feedback_data
        
        # Save to file
        with open(feedback_file, "w") as f:
            json.dump(all_feedback, f, indent=2)
        
        # Clear the feedback data
        feedback_data = []
        
        print(f"Feedback saved to {feedback_file}")
        
    except Exception as e:
        print(f"Error saving feedback: {e}")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded models.
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get model information
        model_info = {
            "Models": list(ensemble_model.models.keys()),
            "Weights": ensemble_model.weights,
            "Threshold": ensemble_model.threshold,
            "FeaturePipeline": str(ensemble_model.feature_pipeline)
        }
        
        return model_info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/reload-models")
async def reload_models():
    """
    Reload the models.
    """
    try:
        load_models()
        return {"message": "Models reloaded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
