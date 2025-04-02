"""
Model monitoring for financial fraud detection.
"""
import os
import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
from scipy.stats import ks_2samp

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

class ModelMonitor:
    """
    Monitor for tracking model performance and detecting data drift.
    """
    def __init__(self):
        """
        Initialize the model monitor.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.models_dir = self.project_root / "src" / "models"
        self.data_dir = self.project_root / "src" / "data"
        self.monitoring_dir = self.project_root / "src" / "monitoring" / "reports"
        self.monitoring_dir.mkdir(exist_ok=True, parents=True)
        
        # Connect to the database
        self.db_path = self.data_dir / "financial_transactions.db"
        
    def connect_to_db(self):
        """
        Connect to the SQLite database.
        
        Returns:
            sqlite3.Connection: Database connection object.
        """
        return sqlite3.connect(str(self.db_path))
    
    def load_predictions(self, days=7):
        """
        Load recent predictions from the database.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            pd.DataFrame: DataFrame containing recent predictions.
        """
        conn = self.connect_to_db()
        
        # Calculate the date threshold
        date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Query to get recent predictions
        query = f"""
        SELECT p.*, t.Fraud
        FROM predictions p
        JOIN transactions t ON p.TransactionID = t.TransactionID
        WHERE p.Timestamp >= '{date_threshold}'
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except:
            # If the predictions table doesn't exist yet, return an empty DataFrame
            conn.close()
            return pd.DataFrame()
    
    def load_feedback(self):
        """
        Load feedback data.
        
        Returns:
            pd.DataFrame: DataFrame containing feedback data.
        """
        feedback_file = self.data_dir / "feedback.json"
        
        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                feedback_data = json.load(f)
            
            return pd.DataFrame(feedback_data)
        else:
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, predictions_df):
        """
        Calculate performance metrics based on recent predictions.
        
        Args:
            predictions_df: DataFrame containing predictions.
            
        Returns:
            dict: Dictionary of performance metrics.
        """
        if predictions_df.empty or 'FraudPrediction' not in predictions_df.columns or 'Fraud' not in predictions_df.columns:
            return {}
        
        # Convert to binary values
        y_true = predictions_df['Fraud'].astype(int)
        y_pred = predictions_df['FraudPrediction'].astype(int)
        y_score = predictions_df['FraudScore']
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_score),
            'fraud_rate': y_true.mean(),
            'alert_rate': y_pred.mean(),
            'sample_size': len(y_true)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]
        
        return metrics
    
    def detect_data_drift(self, reference_data, current_data, threshold=0.1):
        """
        Detect data drift using the Kolmogorov-Smirnov test.
        
        Args:
            reference_data: Reference data (e.g., training data).
            current_data: Current data to check for drift.
            threshold: Threshold for the KS statistic to consider drift.
            
        Returns:
            dict: Dictionary containing drift detection results.
        """
        if reference_data.empty or current_data.empty:
            return {}
        
        # Select only numeric columns
        numeric_cols = reference_data.select_dtypes(include=['number']).columns.intersection(
            current_data.select_dtypes(include=['number']).columns
        )
        
        drift_results = {}
        
        for col in numeric_cols:
            # Skip columns with all identical values
            if reference_data[col].nunique() <= 1 or current_data[col].nunique() <= 1:
                continue
            
            # Perform KS test
            ks_stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            
            drift_results[col] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': ks_stat > threshold
            }
        
        # Calculate overall drift score (percentage of features with drift)
        if drift_results:
            drift_score = sum(1 for col in drift_results if drift_results[col]['drift_detected']) / len(drift_results)
            drift_results['overall_drift_score'] = drift_score
            drift_results['overall_drift_detected'] = drift_score > 0.3  # If more than 30% of features have drift
        
        return drift_results
    
    def plot_metrics_over_time(self, metrics_history):
        """
        Plot performance metrics over time.
        
        Args:
            metrics_history: List of dictionaries containing metrics over time.
            
        Returns:
            str: Path to the saved plot.
        """
        if not metrics_history:
            return None
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_history)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        plt.subplot(2, 2, 1)
        plt.plot(metrics_df['date'], metrics_df['precision'], label='Precision')
        plt.plot(metrics_df['date'], metrics_df['recall'], label='Recall')
        plt.plot(metrics_df['date'], metrics_df['f1_score'], label='F1 Score')
        plt.title('Precision, Recall, and F1 Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.plot(metrics_df['date'], metrics_df['roc_auc'], label='ROC AUC')
        plt.title('ROC AUC Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.plot(metrics_df['date'], metrics_df['fraud_rate'], label='Fraud Rate')
        plt.plot(metrics_df['date'], metrics_df['alert_rate'], label='Alert Rate')
        plt.title('Fraud Rate and Alert Rate Over Time')
        plt.xlabel('Date')
        plt.ylabel('Rate')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.plot(metrics_df['date'], metrics_df['sample_size'], label='Sample Size')
        plt.title('Sample Size Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.monitoring_dir / f"metrics_over_time_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def plot_confusion_matrix(self, metrics):
        """
        Plot confusion matrix.
        
        Args:
            metrics: Dictionary containing confusion matrix values.
            
        Returns:
            str: Path to the saved plot.
        """
        if not metrics or 'true_positives' not in metrics:
            return None
        
        # Create confusion matrix
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.monitoring_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def plot_drift_metrics(self, drift_results):
        """
        Plot drift metrics.
        
        Args:
            drift_results: Dictionary containing drift detection results.
            
        Returns:
            str: Path to the saved plot.
        """
        if not drift_results or 'overall_drift_score' not in drift_results:
            return None
        
        # Extract KS statistics for each feature
        features = []
        ks_stats = []
        drift_detected = []
        
        for feature, result in drift_results.items():
            if feature != 'overall_drift_score' and feature != 'overall_drift_detected':
                features.append(feature)
                ks_stats.append(result['ks_statistic'])
                drift_detected.append(result['drift_detected'])
        
        # Create DataFrame
        drift_df = pd.DataFrame({
            'Feature': features,
            'KS Statistic': ks_stats,
            'Drift Detected': drift_detected
        })
        
        # Sort by KS statistic
        drift_df = drift_df.sort_values('KS Statistic', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot KS statistics
        plt.subplot(1, 1, 1)
        bars = plt.bar(drift_df['Feature'], drift_df['KS Statistic'], color=['red' if d else 'green' for d in drift_df['Drift Detected']])
        plt.title('Feature Drift (Kolmogorov-Smirnov Statistic)')
        plt.xlabel('Feature')
        plt.ylabel('KS Statistic')
        plt.axhline(y=0.1, color='r', linestyle='--', label='Drift Threshold (0.1)')
        plt.legend()
        plt.xticks(rotation=90)
        
        # Add overall drift score
        plt.text(0.02, 0.95, f"Overall Drift Score: {drift_results['overall_drift_score']:.2f}",
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.monitoring_dir / f"drift_metrics_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def generate_monitoring_report(self, days=7):
        """
        Generate a monitoring report.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            dict: Dictionary containing the report.
        """
        # Load recent predictions
        predictions_df = self.load_predictions(days)
        
        # Load feedback data
        feedback_df = self.load_feedback()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(predictions_df)
        
        # Load reference data (training data)
        try:
            # Load the latest version
            with open(self.data_dir / "processed" / "latest_version.txt", "r") as f:
                timestamp = f.read().strip()
            
            reference_data = pd.read_csv(self.data_dir / "processed" / f"X_train_{timestamp}.csv")
            
            # Detect data drift
            if not predictions_df.empty:
                current_data = predictions_df.drop(columns=['FraudPrediction', 'FraudScore', 'Timestamp', 'Fraud'], errors='ignore')
                drift_results = self.detect_data_drift(reference_data, current_data)
            else:
                drift_results = {}
                
        except FileNotFoundError:
            drift_results = {}
        
        # Create plots
        metrics_plot = None
        confusion_matrix_plot = None
        drift_plot = None
        
        if metrics:
            # For metrics over time, we would need historical data
            # Here we just use the current metrics
            metrics_history = [
                {**metrics, 'date': datetime.now().strftime("%Y-%m-%d")}
            ]
            metrics_plot = self.plot_metrics_over_time(metrics_history)
            confusion_matrix_plot = self.plot_confusion_matrix(metrics)
        
        if drift_results:
            drift_plot = self.plot_drift_metrics(drift_results)
        
        # Create the report
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': f"Last {days} days",
            'metrics': metrics,
            'drift_results': drift_results,
            'plots': {
                'metrics_plot': metrics_plot,
                'confusion_matrix_plot': confusion_matrix_plot,
                'drift_plot': drift_plot
            },
            'feedback_count': len(feedback_df),
            'predictions_count': len(predictions_df)
        }
        
        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.monitoring_dir / f"monitoring_report_{timestamp}.json"
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Monitoring report generated and saved to {report_path}")
        
        return report
    
    def check_model_health(self, report=None, days=7):
        """
        Check model health and trigger alerts if necessary.
        
        Args:
            report: Monitoring report. If None, generates a new report.
            days: Number of days to look back.
            
        Returns:
            dict: Dictionary containing health check results.
        """
        if report is None:
            report = self.generate_monitoring_report(days)
        
        health_check = {
            'timestamp': datetime.now().isoformat(),
            'alerts': [],
            'status': 'healthy'
        }
        
        # Check performance metrics
        if 'metrics' in report and report['metrics']:
            metrics = report['metrics']
            
            # Check precision
            if metrics.get('precision', 1.0) < 0.7:
                health_check['alerts'].append({
                    'type': 'performance',
                    'metric': 'precision',
                    'value': metrics['precision'],
                    'threshold': 0.7,
                    'message': f"Precision is below threshold: {metrics['precision']:.2f} < 0.7"
                })
                health_check['status'] = 'degraded'
            
            # Check recall
            if metrics.get('recall', 1.0) < 0.7:
                health_check['alerts'].append({
                    'type': 'performance',
                    'metric': 'recall',
                    'value': metrics['recall'],
                    'threshold': 0.7,
                    'message': f"Recall is below threshold: {metrics['recall']:.2f} < 0.7"
                })
                health_check['status'] = 'degraded'
            
            # Check F1 score
            if metrics.get('f1_score', 1.0) < 0.7:
                health_check['alerts'].append({
                    'type': 'performance',
                    'metric': 'f1_score',
                    'value': metrics['f1_score'],
                    'threshold': 0.7,
                    'message': f"F1 score is below threshold: {metrics['f1_score']:.2f} < 0.7"
                })
                health_check['status'] = 'degraded'
            
            # Check ROC AUC
            if metrics.get('roc_auc', 1.0) < 0.8:
                health_check['alerts'].append({
                    'type': 'performance',
                    'metric': 'roc_auc',
                    'value': metrics['roc_auc'],
                    'threshold': 0.8,
                    'message': f"ROC AUC is below threshold: {metrics['roc_auc']:.2f} < 0.8"
                })
                health_check['status'] = 'degraded'
        
        # Check data drift
        if 'drift_results' in report and report['drift_results'] and 'overall_drift_detected' in report['drift_results']:
            if report['drift_results']['overall_drift_detected']:
                health_check['alerts'].append({
                    'type': 'data_drift',
                    'value': report['drift_results']['overall_drift_score'],
                    'threshold': 0.3,
                    'message': f"Data drift detected: {report['drift_results']['overall_drift_score']:.2f} > 0.3"
                })
                health_check['status'] = 'degraded'
        
        # Save health check results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        health_check_path = self.monitoring_dir / f"health_check_{timestamp}.json"
        
        with open(health_check_path, "w") as f:
            json.dump(health_check, f, indent=2)
        
        print(f"Health check completed and saved to {health_check_path}")
        
        return health_check

if __name__ == "__main__":
    # Run the model monitoring
    monitor = ModelMonitor()
    report = monitor.generate_monitoring_report()
    health_check = monitor.check_model_health(report)
    
    # Print summary
    print("\nMonitoring Report Summary:")
    if 'metrics' in report and report['metrics']:
        print(f"Performance Metrics:")
        for metric, value in report['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    print("\nHealth Check Summary:")
    print(f"Status: {health_check['status']}")
    if health_check['alerts']:
        print("Alerts:")
        for alert in health_check['alerts']:
            print(f"  {alert['message']}")
