"""
Ensemble scoring mechanism for financial fraud detection.
"""
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.pyfunc
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Import custom modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import FeatureEngineeringPipeline

class FraudDetectionEnsemble(mlflow.pyfunc.PythonModel):
    """
    Ensemble model for fraud detection that combines supervised and unsupervised models.
    """
    def __init__(self, models=None, weights=None, threshold=0.5):
        """
        Initialize the ensemble model.
        
        Args:
            models: Dictionary of trained models.
            weights: Dictionary of model weights.
            threshold: Threshold for binary classification.
        """
        self.models = models or {}
        self.weights = weights or {}
        self.threshold = threshold
        self.feature_pipeline = None
    
    def load_context(self, context):
        """
        Load the model artifacts when the model is loaded from MLflow.
        
        Args:
            context: MLflow model context.
        """
        # Load the models
        self.models = {
            'random_forest': joblib.load(context.artifacts['random_forest']),
            'xgboost': joblib.load(context.artifacts['xgboost']),
            'isolation_forest': joblib.load(context.artifacts['isolation_forest'])
        }
        
        # Load the weights
        self.weights = joblib.load(context.artifacts['weights'])
        
        # Load the feature pipeline
        self.feature_pipeline = joblib.load(context.artifacts['feature_pipeline'])
        
        # Load the threshold
        self.threshold = float(context.artifacts['threshold'])
    
    def predict(self, context, model_input):
        """
        Generate predictions using the ensemble model.
        
        Args:
            context: MLflow model context.
            model_input: Input data for prediction.
            
        Returns:
            dict: Dictionary containing predictions and scores.
        """
        # Convert to DataFrame if not already
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        
        # Preprocess the data
        if self.feature_pipeline is not None:
            X_processed = self.feature_pipeline.transform(model_input)
        else:
            # If no feature pipeline is provided, assume the input is already processed
            X_processed = model_input
        
        # Get predictions from each model
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                # For Isolation Forest, convert decision_function to probabilities
                scores = -model.decision_function(X_processed)
                # Normalize to [0, 1]
                predictions[name] = (scores - scores.min()) / (scores.max() - scores.min()) if len(scores) > 0 else scores
            else:
                # For supervised models
                predictions[name] = model.predict_proba(X_processed)[:, 1]
        
        # Create ensemble scores (weighted average)
        ensemble_scores = np.zeros(len(model_input))
        for name, preds in predictions.items():
            if name in self.weights:
                ensemble_scores += self.weights[name] * preds
        
        # Convert to binary predictions using the threshold
        ensemble_preds = (ensemble_scores > self.threshold).astype(int)
        
        # Return predictions and scores
        return {
            'prediction': ensemble_preds,
            'fraud_score': ensemble_scores,
            'model_scores': {name: preds for name, preds in predictions.items()}
        }

class EnsembleBuilder:
    """
    Builder for creating and optimizing ensemble models.
    """
    def __init__(self):
        """
        Initialize the ensemble builder.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.models_dir = self.project_root / "src" / "models"
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def load_latest_models(self):
        """
        Load the latest trained models.
        
        Returns:
            dict: Dictionary of loaded models.
        """
        # Find the latest model files
        model_files = {
            'random_forest': sorted(self.models_dir.glob('random_forest_*.joblib'))[-1],
            'xgboost': sorted(self.models_dir.glob('xgboost_*.joblib'))[-1],
            'isolation_forest': sorted(self.models_dir.glob('isolation_forest_*.joblib'))[-1]
        }
        
        # Load the models
        models = {name: joblib.load(str(path)) for name, path in model_files.items()}
        
        return models
    
    def load_latest_feature_pipeline(self):
        """
        Load the latest feature engineering pipeline.
        
        Returns:
            FeatureEngineeringPipeline: Loaded feature pipeline.
        """
        # Find the latest feature pipeline
        try:
            with open(self.models_dir / "latest_feature_pipeline.txt", "r") as f:
                timestamp = f.read().strip()
            pipeline_path = self.models_dir / f"feature_pipeline_{timestamp}.joblib"
            
            # Load the pipeline
            return joblib.load(pipeline_path)
        except FileNotFoundError:
            # If no pipeline is found, create a new one
            return FeatureEngineeringPipeline()
    
    def optimize_weights(self, models, X_val, y_val):
        """
        Optimize the weights for the ensemble model using validation data.
        
        Args:
            models: Dictionary of trained models.
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            dict: Optimized weights.
        """
        # Get predictions from each model
        predictions = {}
        
        for name, model in models.items():
            if name == 'isolation_forest':
                # For Isolation Forest, convert decision_function to probabilities
                scores = -model.decision_function(X_val)
                # Normalize to [0, 1]
                predictions[name] = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                # For supervised models
                predictions[name] = model.predict_proba(X_val)[:, 1]
        
        # Define a grid of weights to try
        weight_grid = []
        for rf_weight in np.linspace(0.1, 0.5, 5):
            for xgb_weight in np.linspace(0.3, 0.7, 5):
                iso_weight = 1 - rf_weight - xgb_weight
                if 0 <= iso_weight <= 0.5:  # Ensure weights are valid
                    weight_grid.append({
                        'random_forest': rf_weight,
                        'xgboost': xgb_weight,
                        'isolation_forest': iso_weight
                    })
        
        # Evaluate each combination of weights
        best_auc = 0
        best_weights = None
        
        for weights in weight_grid:
            # Calculate ensemble scores
            ensemble_scores = np.zeros(len(y_val))
            for name, preds in predictions.items():
                ensemble_scores += weights[name] * preds
            
            # Calculate AUC
            auc = roc_auc_score(y_val, ensemble_scores)
            
            # Update best weights if better AUC is found
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
        
        print(f"Optimized weights: {best_weights}")
        print(f"Best AUC: {best_auc:.4f}")
        
        return best_weights
    
    def find_optimal_threshold(self, models, weights, X_val, y_val):
        """
        Find the optimal threshold for binary classification.
        
        Args:
            models: Dictionary of trained models.
            weights: Dictionary of model weights.
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            float: Optimal threshold.
        """
        # Get predictions from each model
        predictions = {}
        
        for name, model in models.items():
            if name == 'isolation_forest':
                # For Isolation Forest, convert decision_function to probabilities
                scores = -model.decision_function(X_val)
                # Normalize to [0, 1]
                predictions[name] = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                # For supervised models
                predictions[name] = model.predict_proba(X_val)[:, 1]
        
        # Calculate ensemble scores
        ensemble_scores = np.zeros(len(y_val))
        for name, preds in predictions.items():
            ensemble_scores += weights[name] * preds
        
        # Find the optimal threshold using precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val, ensemble_scores)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find the threshold with the highest F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"Best F1 score: {f1_scores[best_idx]:.4f}")
        
        return best_threshold
    
    def build_ensemble(self, models=None, X_val=None, y_val=None, weights=None, threshold=None):
        """
        Build an ensemble model.
        
        Args:
            models: Dictionary of trained models. If None, loads the latest models.
            X_val, y_val: Validation data for optimizing weights and threshold.
            weights: Dictionary of model weights. If None, uses default weights.
            threshold: Threshold for binary classification. If None, uses default threshold.
            
        Returns:
            FraudDetectionEnsemble: Ensemble model.
        """
        # Load models if not provided
        if models is None:
            models = self.load_latest_models()
        
        # Use default weights if not provided and validation data is not available
        if weights is None and (X_val is None or y_val is None):
            weights = {
                'random_forest': 0.3,
                'xgboost': 0.5,
                'isolation_forest': 0.2
            }
        
        # Optimize weights if validation data is provided
        if weights is None and X_val is not None and y_val is not None:
            weights = self.optimize_weights(models, X_val, y_val)
        
        # Use default threshold if not provided and validation data is not available
        if threshold is None and (X_val is None or y_val is None):
            threshold = 0.5
        
        # Find optimal threshold if validation data is provided
        if threshold is None and X_val is not None and y_val is not None:
            threshold = self.find_optimal_threshold(models, weights, X_val, y_val)
        
        # Create the ensemble model
        ensemble = FraudDetectionEnsemble(models, weights, threshold)
        
        # Load the feature pipeline
        ensemble.feature_pipeline = self.load_latest_feature_pipeline()
        
        # Save the ensemble
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the weights
        weights_path = self.models_dir / f"ensemble_weights_{timestamp}.joblib"
        joblib.dump(weights, weights_path)
        
        # Save the threshold
        with open(self.models_dir / f"ensemble_threshold_{timestamp}.txt", "w") as f:
            f.write(str(threshold))
        
        # Save the ensemble configuration
        ensemble_config = {
            'timestamp': timestamp,
            'weights': weights,
            'threshold': threshold,
            'models': {name: str(model) for name, model in models.items()}
        }
        
        joblib.dump(ensemble_config, self.models_dir / f"ensemble_config_{timestamp}.joblib")
        
        # Log the ensemble with MLflow
        with mlflow.start_run(run_name="ensemble_model"):
            # Log parameters
            for name, weight in weights.items():
                mlflow.log_param(f"weight_{name}", weight)
            mlflow.log_param("threshold", threshold)
            
            # Log the model
            artifacts = {
                'random_forest': str(self.models_dir / model_files['random_forest'].name),
                'xgboost': str(self.models_dir / model_files['xgboost'].name),
                'isolation_forest': str(self.models_dir / model_files['isolation_forest'].name),
                'weights': str(weights_path),
                'threshold': str(threshold),
                'feature_pipeline': str(self.models_dir / f"feature_pipeline_{pipeline_timestamp}.joblib")
            }
            
            mlflow.pyfunc.log_model(
                artifact_path="ensemble_model",
                python_model=ensemble,
                artifacts=artifacts
            )
        
        print(f"Ensemble model created and saved with timestamp: {timestamp}")
        
        return ensemble

if __name__ == "__main__":
    # This is a demonstration of how to build an ensemble model
    import sys
    from pathlib import Path
    
    # Add the project root to the Python path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    
    from src.data.data_ingestion import DataIngestionPipeline
    from src.features.feature_engineering import FeatureEngineeringPipeline
    
    # Load the data
    data_pipeline = DataIngestionPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline.load_data()
    
    # Preprocess the data
    feature_pipeline = FeatureEngineeringPipeline()
    X_val_processed = feature_pipeline.transform(X_val)
    
    # Build the ensemble
    builder = EnsembleBuilder()
    models = builder.load_latest_models()
    ensemble = builder.build_ensemble(models, X_val_processed, y_val)
    
    # Make predictions
    predictions = ensemble.predict(None, X_val)
    
    # Print some statistics
    print(f"\nEnsemble Predictions:")
    print(f"Fraud rate: {predictions['prediction'].mean():.4f}")
    print(f"Average fraud score: {predictions['fraud_score'].mean():.4f}")
    
    # Evaluate the ensemble
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_val, predictions['prediction'])
    precision = precision_score(y_val, predictions['prediction'])
    recall = recall_score(y_val, predictions['prediction'])
    f1 = f1_score(y_val, predictions['prediction'])
    roc_auc = roc_auc_score(y_val, predictions['fraud_score'])
    
    print(f"\nEnsemble Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
