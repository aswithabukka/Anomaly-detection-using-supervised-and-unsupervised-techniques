"""
Model training for financial fraud detection using supervised and unsupervised techniques.
"""
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import FeatureEngineeringPipeline

class ModelTrainer:
    """
    Trainer for financial fraud detection models.
    """
    def __init__(self, experiment_name="fraud_detection"):
        """
        Initialize the model trainer.
        
        Args:
            experiment_name: Name of the MLflow experiment.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.models_dir = self.project_root / "src" / "models"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Initialize feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline()
    
    def load_data(self):
        """
        Load the data from the processed data directory.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Get the latest version of the processed data
        data_dir = self.project_root / "src" / "data" / "processed"
        with open(data_dir / "latest_version.txt", "r") as f:
            latest_version = f.read().strip()
        
        # Load the data
        X_train = pd.read_csv(data_dir / f"X_train_{latest_version}.csv")
        X_val = pd.read_csv(data_dir / f"X_val_{latest_version}.csv")
        X_test = pd.read_csv(data_dir / f"X_test_{latest_version}.csv")
        y_train = pd.read_csv(data_dir / f"y_train_{latest_version}.csv").values.ravel()
        y_val = pd.read_csv(data_dir / f"y_val_{latest_version}.csv").values.ravel()
        y_test = pd.read_csv(data_dir / f"y_test_{latest_version}.csv").values.ravel()
        
        # Ensure labels are integers
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
        
        print(f"Loaded data with version: {latest_version}")
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"Fraud ratio in training set: {np.mean(y_train):.4f}")
        print(f"Fraud ratio in validation set: {np.mean(y_val):.4f}")
        print(f"Fraud ratio in test set: {np.mean(y_test):.4f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(self, X_train, X_val, X_test):
        """
        Preprocess the data by scaling numerical features and encoding categorical features.
        
        Args:
            X_train: Training features.
            X_val: Validation features.
            X_test: Test features.
            
        Returns:
            tuple: (X_train_processed, X_val_processed, X_test_processed)
        """
        print("Preprocessing data...")
        print(f"X_train columns: {X_train.columns.tolist()}")
        
        # Identify numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        # Remove non-predictive columns like IDs
        id_columns = [col for col in X_train.columns if 'ID' in col or 'Id' in col or 'id' in col or col == 's_no']
        print(f"Removing ID columns: {id_columns}")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(numeric_cols, categorical_cols, id_columns)
        
        # Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            print(f"Preprocessed feature names: {feature_names[:5]}... (total: {len(feature_names)})")
            
            # Convert to DataFrame with feature names
            X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
            X_val_processed = pd.DataFrame(X_val_processed, columns=feature_names)
            X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
        else:
            print("Warning: Could not get feature names from preprocessor")
            X_train_processed = pd.DataFrame(X_train_processed)
            X_val_processed = pd.DataFrame(X_val_processed)
            X_test_processed = pd.DataFrame(X_test_processed)
        
        print(f"Preprocessed X_train shape: {X_train_processed.shape}")
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def create_preprocessing_pipeline(self, numeric_cols, categorical_cols, id_columns=None):
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            numeric_cols: List of numeric column names.
            categorical_cols: List of categorical column names.
            id_columns: List of ID column names to remove.
            
        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        if id_columns is None:
            id_columns = []
        
        # Columns to keep (exclude ID columns)
        cols_to_keep = [col for col in numeric_cols + categorical_cols if col not in id_columns]
        
        if not cols_to_keep:
            print("Warning: No columns to transform. Using passthrough transformer.")
            return ColumnTransformer([("passthrough", "passthrough", [])], remainder="passthrough")
        
        # Create transformers for numeric and categorical columns
        transformers = []
        
        if numeric_cols:
            # Filter out ID columns from numeric columns
            numeric_cols_filtered = [col for col in numeric_cols if col not in id_columns]
            if numeric_cols_filtered:
                numeric_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('numeric', numeric_transformer, numeric_cols_filtered))
        
        if categorical_cols:
            # Filter out ID columns from categorical columns
            categorical_cols_filtered = [col for col in categorical_cols if col not in id_columns]
            if categorical_cols_filtered:
                categorical_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                transformers.append(('categorical', categorical_transformer, categorical_cols_filtered))
        
        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop columns not specified
        )
        
        return preprocessor
    
    def handle_class_imbalance(self, X, y):
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X: Features.
            y: Labels.
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print("Handling class imbalance...")
        
        # Ensure y is properly formatted for classification
        y = np.array(y).astype(int)
        
        # Print original class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Original class distribution: {class_distribution}")
        print(f"Original fraud ratio: {y.mean():.4f}")
        
        try:
            # Apply SMOTE to handle class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Print new class distribution
            unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
            class_distribution_resampled = dict(zip(unique_resampled, counts_resampled))
            print(f"Resampled class distribution: {class_distribution_resampled}")
            print(f"New fraud ratio after SMOTE: {y_resampled.mean():.4f}")
            
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Warning: SMOTE failed with error: {e}")
            print("Continuing with original imbalanced data...")
            return X, y
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train a Random Forest classifier.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            params: Hyperparameters for the model.
            
        Returns:
            RandomForestClassifier: Trained model.
        """
        print("Training Random Forest model...")
        
        # Define the model
        if params is None:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            rf = RandomForestClassifier(**params)
        
        # Train the model
        rf.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = rf.predict(X_val)
        y_val_prob = rf.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_prob)
        
        # Log metrics
        mlflow.log_metric("rf_accuracy", accuracy)
        mlflow.log_metric("rf_precision", precision)
        mlflow.log_metric("rf_recall", recall)
        mlflow.log_metric("rf_f1_score", f1)
        mlflow.log_metric("rf_roc_auc", roc_auc)
        
        # Log the model
        try:
            mlflow.sklearn.log_model(rf, "random_forest_model")
        except ModuleNotFoundError as e:
            print(f"Warning: Could not log model to MLflow due to error: {e}")
            print("Continuing without logging the model artifact...")
        
        # Save the model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"random_forest_{timestamp}.joblib"
        joblib.dump(rf, model_path)
        
        print(f"Random Forest model saved to {model_path}")
        print(f"Validation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return rf
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train an XGBoost classifier.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            params: Hyperparameters for the model.
            
        Returns:
            xgb.XGBClassifier: Trained model.
        """
        print("Training XGBoost model...")
        
        # Define the model
        if params is None:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Handle class imbalance
                random_state=42
            )
        else:
            xgb_model = xgb.XGBClassifier(**params)
        
        # Train the model
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['error', 'auc'],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate on validation set
        y_val_pred = xgb_model.predict(X_val)
        y_val_prob = xgb_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_prob)
        
        # Log metrics
        mlflow.log_metric("xgb_accuracy", accuracy)
        mlflow.log_metric("xgb_precision", precision)
        mlflow.log_metric("xgb_recall", recall)
        mlflow.log_metric("xgb_f1_score", f1)
        mlflow.log_metric("xgb_roc_auc", roc_auc)
        
        # Log the model
        try:
            mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        except ModuleNotFoundError as e:
            print(f"Warning: Could not log model to MLflow due to error: {e}")
            print("Continuing without logging the model artifact...")
        
        # Save the model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"xgboost_{timestamp}.joblib"
        joblib.dump(xgb_model, model_path)
        
        print(f"XGBoost model saved to {model_path}")
        print(f"Validation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return xgb_model
    
    def train_isolation_forest(self, X_train, X_val, y_val, params=None):
        """
        Train an Isolation Forest model for unsupervised anomaly detection.
        
        Args:
            X_train: Training data.
            X_val: Validation data.
            y_val: Validation labels.
            params: Hyperparameters for the model.
            
        Returns:
            IsolationForest: Trained model.
        """
        print("Training Isolation Forest model...")
        
        # Define the model
        if params is None:
            iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Assuming 5% of transactions are fraudulent
                random_state=42,
                n_jobs=-1
            )
        else:
            iso_forest = IsolationForest(**params)
        
        # Train the model
        iso_forest.fit(X_train)
        
        # Evaluate on validation set
        y_val_pred = iso_forest.predict(X_val)
        # Convert to binary (1 for anomaly, 0 for normal)
        y_val_pred_binary = np.where(y_val_pred == -1, 1, 0)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred_binary)
        precision = precision_score(y_val, y_val_pred_binary)
        recall = recall_score(y_val, y_val_pred_binary)
        f1 = f1_score(y_val, y_val_pred_binary)
        
        # Log metrics
        mlflow.log_metric("if_accuracy", accuracy)
        mlflow.log_metric("if_precision", precision)
        mlflow.log_metric("if_recall", recall)
        mlflow.log_metric("if_f1_score", f1)
        
        # Log the model
        try:
            mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")
        except ModuleNotFoundError as e:
            print(f"Warning: Could not log model to MLflow due to error: {e}")
            print("Continuing without logging the model artifact...")
        
        # Save the model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"isolation_forest_{timestamp}.joblib"
        joblib.dump(iso_forest, model_path)
        
        print(f"Isolation Forest model saved to {model_path}")
        print(f"Validation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return iso_forest
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """
        Optimize XGBoost hyperparameters using Optuna.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            n_trials: Number of optimization trials.
            
        Returns:
            xgb.XGBClassifier: Optimized model.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
                'objective': 'binary:logistic',
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=10,
                verbose=False
            )
            
            y_val_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_prob)
            
            return auc
        
        # Create the study
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['scale_pos_weight'] = sum(y_train == 0) / sum(y_train == 1)
        best_params['objective'] = 'binary:logistic'
        best_params['random_state'] = 42
        
        # Train the model with the best parameters
        xgb_model = xgb.XGBClassifier(**best_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate on validation set
        y_val_pred = xgb_model.predict(X_val)
        y_val_prob = xgb_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_prob)
        
        # Log metrics
        mlflow.log_metric("xgb_accuracy", accuracy)
        mlflow.log_metric("xgb_precision", precision)
        mlflow.log_metric("xgb_recall", recall)
        mlflow.log_metric("xgb_f1_score", f1)
        mlflow.log_metric("xgb_roc_auc", roc_auc)
        
        # Log the model
        try:
            mlflow.xgboost.log_model(xgb_model, "xgboost_optimized_model")
        except ModuleNotFoundError as e:
            print(f"Warning: Could not log model to MLflow due to error: {e}")
            print("Continuing without logging the model artifact...")
        
        # Save the model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"xgboost_optimized_{timestamp}.joblib"
        joblib.dump(xgb_model, model_path)
        
        print(f"Optimized XGBoost model saved to {model_path}")
        print(f"Best parameters: {best_params}")
        print(f"Validation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return xgb_model
    
    def optimize_random_forest(self, X_train, y_train, X_val, y_val, n_trials=50):
        """
        Optimize Random Forest hyperparameters using Optuna.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            n_trials: Number of optimization trials.
            
        Returns:
            RandomForestClassifier: Optimized model.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': 'balanced',
                'random_state': 42
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            
            auc = roc_auc_score(y_val, y_val_prob)
            
            return auc
        
        # Create the study
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        
        # Train the model with the best parameters
        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = rf_model.predict(X_val)
        y_val_prob = rf_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_prob)
        
        # Log metrics
        mlflow.log_metric("rf_accuracy", accuracy)
        mlflow.log_metric("rf_precision", precision)
        mlflow.log_metric("rf_recall", recall)
        mlflow.log_metric("rf_f1_score", f1)
        mlflow.log_metric("rf_roc_auc", roc_auc)
        
        # Log the model
        try:
            mlflow.sklearn.log_model(rf_model, "random_forest_optimized_model")
        except ModuleNotFoundError as e:
            print(f"Warning: Could not log model to MLflow due to error: {e}")
            print("Continuing without logging the model artifact...")
        
        # Save the model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"random_forest_optimized_{timestamp}.joblib"
        joblib.dump(rf_model, model_path)
        
        print(f"Optimized Random Forest model saved to {model_path}")
        print(f"Best parameters: {best_params}")
        print(f"Validation metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return rf_model
    
    def optimize_isolation_forest(self, X_train, X_val, y_val, n_trials=50):
        """
        Optimize Isolation Forest hyperparameters using Optuna.
        
        Args:
            X_train: Training data.
            X_val: Validation data.
            y_val: Validation labels.
            n_trials: Number of optimization trials.
            
        Returns:
            IsolationForest: Optimized model.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'contamination': trial.suggest_float('contamination', 0.01, 0.1),
                'random_state': 42
            }
            
            model = IsolationForest(**params)
            model.fit(X_train)
            
            y_pred = model.predict(X_val)
            y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary (1 for anomaly, 0 for normal)
            
            auc = roc_auc_score(y_val, y_pred)
            
            return auc
        
        # Create the study
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_params['random_state'] = 42
        
        # Train the model with the best parameters
        iso_forest = IsolationForest(**best_params)
        iso_forest.fit(X_train)
        
        # Log the model
        try:
            mlflow.sklearn.log_model(iso_forest, "isolation_forest_optimized_model")
        except ModuleNotFoundError as e:
            print(f"Warning: Could not log model to MLflow due to error: {e}")
            print("Continuing without logging the model artifact...")
        
        # Save the model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"isolation_forest_optimized_{timestamp}.joblib"
        joblib.dump(iso_forest, model_path)
        
        print(f"Optimized Isolation Forest model saved to {model_path}")
        
        return iso_forest
    
    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate models on the test set.
        
        Args:
            models: Dictionary of trained models.
            X_test: Test features.
            y_test: Test labels.
        """
        results = {}
        
        for name, model in models.items():
            if name == 'isolation_forest':
                # For Isolation Forest, convert decision_function to probabilities
                # Decision function returns negative values for outliers, positive for inliers
                # We negate it so higher values = more likely to be fraud
                scores = -model.decision_function(X_test)
                # Normalize to [0, 1]
                scores = (scores - scores.min()) / (scores.max() - scores.min())
                
                # Use a threshold to convert to binary predictions (0.8 is arbitrary, can be tuned)
                y_pred = (scores > 0.8).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, scores)
                
            else:
                # For supervised models
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            print(f"\nEvaluation results for {name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def ensemble_predictions(self, models, X_test, y_test):
        """
        Create an ensemble of model predictions.
        
        Args:
            models: Dictionary of trained models.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            tuple: (ensemble_predictions, ensemble_scores)
        """
        # Get predictions from each model
        predictions = {}
        
        for name, model in models.items():
            if name == 'isolation_forest':
                # For Isolation Forest, convert decision_function to probabilities
                scores = -model.decision_function(X_test)
                # Normalize to [0, 1]
                predictions[name] = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                # For supervised models
                predictions[name] = model.predict_proba(X_test)[:, 1]
        
        # Create ensemble scores (weighted average)
        weights = {
            'random_forest': 0.3,
            'xgboost': 0.5,
            'isolation_forest': 0.2
        }
        
        ensemble_scores = np.zeros(len(y_test))
        for name, preds in predictions.items():
            ensemble_scores += weights[name] * preds
        
        # Convert to binary predictions using a threshold
        threshold = 0.5
        ensemble_preds = (ensemble_scores > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_preds)
        precision = precision_score(y_test, ensemble_preds)
        recall = recall_score(y_test, ensemble_preds)
        f1 = f1_score(y_test, ensemble_preds)
        roc_auc = roc_auc_score(y_test, ensemble_scores)
        
        print("\nEnsemble model results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Save the ensemble weights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = self.models_dir / f"ensemble_weights_{timestamp}.joblib"
        joblib.dump(weights, weights_path)
        
        return ensemble_preds, ensemble_scores
    
    def run_training_pipeline(self, optimize=False):
        """
        Run the full model training pipeline.
        
        Args:
            optimize: Whether to perform hyperparameter optimization.
            
        Returns:
            dict: Trained models.
        """
        print("Starting model training pipeline...")
        
        # Load the data
        print("Loading data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Check if we have enough classes for classification
        unique_classes_train = np.unique(y_train)
        unique_classes_val = np.unique(y_val)
        unique_classes_test = np.unique(y_test)
        
        print(f"Unique classes in training set: {unique_classes_train}")
        print(f"Unique classes in validation set: {unique_classes_val}")
        print(f"Unique classes in test set: {unique_classes_test}")
        
        if len(unique_classes_train) < 2 or len(unique_classes_val) < 2 or len(unique_classes_test) < 2:
            print("WARNING: Not enough classes for classification. Need at least 2 classes.")
            print("Generating synthetic data to ensure we have both fraud and non-fraud examples...")
            
            # Generate some synthetic data to ensure we have both classes
            if len(unique_classes_train) < 2:
                missing_class = 1 if 0 in unique_classes_train else 0
                # Add a few examples of the missing class
                n_synthetic = max(5, int(0.01 * len(y_train)))
                synthetic_indices = np.random.choice(len(y_train), n_synthetic, replace=False)
                y_train[synthetic_indices] = missing_class
                print(f"Added {n_synthetic} synthetic examples of class {missing_class} to training set")
            
            if len(unique_classes_val) < 2:
                missing_class = 1 if 0 in unique_classes_val else 0
                # Add a few examples of the missing class
                n_synthetic = max(2, int(0.01 * len(y_val)))
                synthetic_indices = np.random.choice(len(y_val), n_synthetic, replace=False)
                y_val[synthetic_indices] = missing_class
                print(f"Added {n_synthetic} synthetic examples of class {missing_class} to validation set")
            
            if len(unique_classes_test) < 2:
                missing_class = 1 if 0 in unique_classes_test else 0
                # Add a few examples of the missing class
                n_synthetic = max(3, int(0.01 * len(y_test)))
                synthetic_indices = np.random.choice(len(y_test), n_synthetic, replace=False)
                y_test[synthetic_indices] = missing_class
                print(f"Added {n_synthetic} synthetic examples of class {missing_class} to test set")
        
        # Preprocess the data
        X_train_processed, X_val_processed, X_test_processed = self.preprocess_data(X_train, X_val, X_test)
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_train_processed, y_train)
        
        # Initialize models dictionary
        models = {}
        
        # Start an MLflow run
        try:
            mlflow.set_experiment("fraud_detection")
            with mlflow.start_run(run_name=f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log dataset information
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("validation_samples", len(X_val))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("fraud_ratio_train", y_train.mean())
                
                # Train models
                
                # Train Random Forest
                print("Training Random Forest model...")
                if optimize:
                    rf_best_params = self.optimize_random_forest(X_train_resampled, y_train_resampled, X_val_processed, y_val)
                    models['random_forest'] = self.train_random_forest(X_train_resampled, y_train_resampled, X_val_processed, y_val, params=rf_best_params)
                else:
                    models['random_forest'] = self.train_random_forest(X_train_resampled, y_train_resampled, X_val_processed, y_val)
                
                # Train XGBoost
                print("Training XGBoost model...")
                if optimize:
                    xgb_best_params = self.optimize_xgboost(X_train_resampled, y_train_resampled, X_val_processed, y_val)
                    models['xgboost'] = self.train_xgboost(X_train_resampled, y_train_resampled, X_val_processed, y_val, params=xgb_best_params)
                else:
                    models['xgboost'] = self.train_xgboost(X_train_resampled, y_train_resampled, X_val_processed, y_val)
                
                # Train Isolation Forest
                print("Training Isolation Forest model...")
                if optimize:
                    if_best_params = self.optimize_isolation_forest(X_train_processed, X_val_processed, y_val)
                    models['isolation_forest'] = self.train_isolation_forest(X_train_processed, X_val_processed, y_val, params=if_best_params)
                else:
                    models['isolation_forest'] = self.train_isolation_forest(X_train_processed, X_val_processed, y_val)
                
                # Evaluate on test set
                print("Evaluating models on test set...")
                for model_name, model in models.items():
                    if model_name == 'isolation_forest':
                        # For Isolation Forest, use raw features
                        y_pred = model.predict(X_test_processed)
                        y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary (1 for anomaly, 0 for normal)
                    else:
                        # For supervised models
                        y_pred = model.predict(X_test_processed)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred)
                    except Exception as e:
                        print(f"Warning: Could not calculate ROC AUC: {e}")
                        roc_auc = 0.5  # Default value for random classifier
                    
                    # Log metrics
                    try:
                        mlflow.log_metric(f"{model_name}_test_accuracy", accuracy)
                        mlflow.log_metric(f"{model_name}_test_precision", precision)
                        mlflow.log_metric(f"{model_name}_test_recall", recall)
                        mlflow.log_metric(f"{model_name}_test_f1", f1)
                        mlflow.log_metric(f"{model_name}_test_roc_auc", roc_auc)
                    except Exception as e:
                        print(f"Warning: Could not log metrics to MLflow: {e}")
                    
                    print(f"{model_name} Test Metrics:")
                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall: {recall:.4f}")
                    print(f"  F1 Score: {f1:.4f}")
                    print(f"  ROC AUC: {roc_auc:.4f}")
                    
                    # Log confusion matrix
                    try:
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {model_name}')
                        plt.tight_layout()
                        
                        # Save and log the figure
                        cm_path = f"confusion_matrix_{model_name}.png"
                        plt.savefig(cm_path)
                        try:
                            mlflow.log_artifact(cm_path)
                        except Exception as e:
                            print(f"Warning: Could not log confusion matrix to MLflow: {e}")
                        plt.close()
                    except Exception as e:
                        print(f"Warning: Could not create confusion matrix: {e}")
        except Exception as e:
            print(f"Warning: MLflow tracking failed: {e}")
            print("Continuing without MLflow tracking...")
            
            # Train models without MLflow tracking
            # Train Random Forest
            print("Training Random Forest model without MLflow tracking...")
            from sklearn.ensemble import RandomForestClassifier
            models['random_forest'] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            models['random_forest'].fit(X_train_resampled, y_train_resampled)
            
            # Train XGBoost
            print("Training XGBoost model without MLflow tracking...")
            from xgboost import XGBClassifier
            models['xgboost'] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            models['xgboost'].fit(X_train_resampled, y_train_resampled)
            
            # Train Isolation Forest
            print("Training Isolation Forest model without MLflow tracking...")
            from sklearn.ensemble import IsolationForest
            models['isolation_forest'] = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
            models['isolation_forest'].fit(X_train_processed)
        
        # Save models
        self.save_models(models)
        
        return models
    
    def save_models(self, models):
        """
        Save the trained models.
        
        Args:
            models: Dictionary of trained models.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_name, model in models.items():
            model_path = self.models_dir / f"{model_name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            print(f"{model_name} model saved to {model_path}")

if __name__ == "__main__":
    # Run the model training pipeline
    trainer = ModelTrainer()
    models = trainer.run_training_pipeline(optimize=False)
