"""
Dashboard for visualizing fraud detection metrics and anomaly scores.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import custom modules
from src.monitoring.model_monitor import ModelMonitor

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Financial Fraud Detection Dashboard"

# Initialize the model monitor
monitor = ModelMonitor()

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1("Financial Fraud Detection Dashboard", className="text-center my-4"),
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Dashboard Controls"),
                            dbc.CardBody(
                                [
                                    html.P("Select time period:"),
                                    dcc.Dropdown(
                                        id="time-period-dropdown",
                                        options=[
                                            {"label": "Last 24 hours", "value": "1"},
                                            {"label": "Last 7 days", "value": "7"},
                                            {"label": "Last 30 days", "value": "30"},
                                            {"label": "Last 90 days", "value": "90"},
                                        ],
                                        value="7",
                                        clearable=False,
                                    ),
                                    html.Div(className="my-3"),
                                    dbc.Button(
                                        "Refresh Data",
                                        id="refresh-button",
                                        color="primary",
                                        className="mr-2",
                                    ),
                                    html.Div(className="my-3"),
                                    dbc.Button(
                                        "Generate Report",
                                        id="report-button",
                                        color="success",
                                    ),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Fraud Rate"),
                                            dbc.CardBody(
                                                [
                                                    html.H2(
                                                        id="fraud-rate-value",
                                                        className="card-title text-center",
                                                    ),
                                                    html.P(
                                                        "Percentage of fraudulent transactions",
                                                        className="card-text text-center",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    width=4,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Alert Rate"),
                                            dbc.CardBody(
                                                [
                                                    html.H2(
                                                        id="alert-rate-value",
                                                        className="card-title text-center",
                                                    ),
                                                    html.P(
                                                        "Percentage of transactions flagged as fraud",
                                                        className="card-text text-center",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    width=4,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Model Health"),
                                            dbc.CardBody(
                                                [
                                                    html.H2(
                                                        id="model-health-value",
                                                        className="card-title text-center",
                                                    ),
                                                    html.P(
                                                        "Current model health status",
                                                        className="card-text text-center",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    width=4,
                                ),
                            ]
                        ),
                        html.Div(className="my-4"),
                        dbc.Card(
                            [
                                dbc.CardHeader("Performance Metrics"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            id="performance-metrics-table",
                                                        )
                                                    ],
                                                    width=12,
                                                )
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
        html.Div(className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Fraud Score Distribution"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="fraud-score-histogram",
                                        config={"displayModeBar": False},
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Confusion Matrix"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="confusion-matrix-heatmap",
                                        config={"displayModeBar": False},
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=6,
                ),
            ]
        ),
        html.Div(className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Metrics Over Time"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="metrics-over-time",
                                        config={"displayModeBar": False},
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=12,
                ),
            ]
        ),
        html.Div(className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Data Drift Detection"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="data-drift-chart",
                                        config={"displayModeBar": False},
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=12,
                ),
            ]
        ),
        html.Div(className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Recent High-Risk Transactions"),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="high-risk-transactions-table",
                                    )
                                ]
                            ),
                        ]
                    ),
                    width=12,
                ),
            ]
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Monitoring Report"),
                dbc.ModalBody(id="report-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-report-modal", className="ml-auto")
                ),
            ],
            id="report-modal",
            size="lg",
        ),
        # Store for intermediate data
        dcc.Store(id="predictions-store"),
        dcc.Store(id="metrics-store"),
        dcc.Store(id="health-store"),
        dcc.Interval(
            id="interval-component",
            interval=300 * 1000,  # Update every 5 minutes
            n_intervals=0,
        ),
    ],
    fluid=True,
)

# Define callback to load data
@app.callback(
    [
        Output("predictions-store", "data"),
        Output("metrics-store", "data"),
        Output("health-store", "data"),
    ],
    [Input("refresh-button", "n_clicks"), Input("interval-component", "n_intervals")],
    [State("time-period-dropdown", "value")],
)
def load_data(n_clicks, n_intervals, days):
    """
    Load data and calculate metrics.
    """
    days = int(days) if days else 7
    
    try:
        # Load predictions
        predictions_df = monitor.load_predictions(days)
        
        # Calculate metrics
        metrics = monitor.calculate_performance_metrics(predictions_df)
        
        # Check model health
        health_check = monitor.check_model_health(days=days)
        
        # Convert DataFrames to dictionaries for JSON serialization
        if not predictions_df.empty:
            predictions_data = predictions_df.to_dict("records")
        else:
            predictions_data = []
        
        return predictions_data, metrics, health_check
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], {}, {"status": "unknown", "alerts": []}

# Define callback to update KPIs
@app.callback(
    [
        Output("fraud-rate-value", "children"),
        Output("alert-rate-value", "children"),
        Output("model-health-value", "children"),
        Output("model-health-value", "className"),
    ],
    [Input("metrics-store", "data"), Input("health-store", "data")],
)
def update_kpis(metrics, health):
    """
    Update KPI values.
    """
    if not metrics:
        return "N/A", "N/A", "Unknown", "card-title text-center text-secondary"
    
    # Format fraud rate
    fraud_rate = metrics.get("fraud_rate", 0) * 100
    fraud_rate_text = f"{fraud_rate:.2f}%"
    
    # Format alert rate
    alert_rate = metrics.get("alert_rate", 0) * 100
    alert_rate_text = f"{alert_rate:.2f}%"
    
    # Format model health
    if not health:
        health_status = "Unknown"
        health_class = "card-title text-center text-secondary"
    else:
        health_status = health.get("status", "unknown").capitalize()
        if health_status == "Healthy":
            health_class = "card-title text-center text-success"
        elif health_status == "Degraded":
            health_class = "card-title text-center text-warning"
        else:
            health_class = "card-title text-center text-secondary"
    
    return fraud_rate_text, alert_rate_text, health_status, health_class

# Define callback to update performance metrics table
@app.callback(
    Output("performance-metrics-table", "children"),
    [Input("metrics-store", "data")],
)
def update_performance_metrics_table(metrics):
    """
    Update the performance metrics table.
    """
    if not metrics:
        return html.P("No data available")
    
    # Create a DataFrame for the metrics
    metrics_to_display = [
        {"Metric": "Accuracy", "Value": metrics.get("accuracy", "N/A")},
        {"Metric": "Precision", "Value": metrics.get("precision", "N/A")},
        {"Metric": "Recall", "Value": metrics.get("recall", "N/A")},
        {"Metric": "F1 Score", "Value": metrics.get("f1_score", "N/A")},
        {"Metric": "ROC AUC", "Value": metrics.get("roc_auc", "N/A")},
        {"Metric": "Sample Size", "Value": metrics.get("sample_size", "N/A")},
    ]
    
    # Format the values
    for item in metrics_to_display:
        if isinstance(item["Value"], float):
            item["Value"] = f"{item['Value']:.4f}"
    
    # Create the table
    table = dash_table.DataTable(
        id="metrics-table",
        columns=[
            {"name": "Metric", "id": "Metric"},
            {"name": "Value", "id": "Value"},
        ],
        data=metrics_to_display,
        style_cell={"textAlign": "left", "padding": "15px"},
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(248, 248, 248)",
            }
        ],
    )
    
    return table

# Define callback to update fraud score histogram
@app.callback(
    Output("fraud-score-histogram", "figure"),
    [Input("predictions-store", "data")],
)
def update_fraud_score_histogram(predictions):
    """
    Update the fraud score histogram.
    """
    if not predictions:
        return go.Figure().update_layout(
            title="No data available",
            xaxis_title="Fraud Score",
            yaxis_title="Count",
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    if "FraudScore" not in df.columns:
        return go.Figure().update_layout(
            title="No fraud score data available",
            xaxis_title="Fraud Score",
            yaxis_title="Count",
        )
    
    # Create the histogram
    fig = px.histogram(
        df,
        x="FraudScore",
        color="Fraud",
        nbins=20,
        labels={"FraudScore": "Fraud Score", "Fraud": "Actual Fraud"},
        color_discrete_map={0: "green", 1: "red"},
        opacity=0.7,
    )
    
    # Add a vertical line for the threshold (assuming 0.5)
    fig.add_shape(
        type="line",
        x0=0.5,
        y0=0,
        x1=0.5,
        y1=1,
        yref="paper",
        line=dict(color="black", width=2, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Fraud Scores",
        xaxis_title="Fraud Score",
        yaxis_title="Count",
        legend_title="Actual Fraud",
        barmode="overlay",
    )
    
    return fig

# Define callback to update confusion matrix
@app.callback(
    Output("confusion-matrix-heatmap", "figure"),
    [Input("metrics-store", "data")],
)
def update_confusion_matrix(metrics):
    """
    Update the confusion matrix heatmap.
    """
    if not metrics or "true_positives" not in metrics:
        return go.Figure().update_layout(
            title="No data available",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
    
    # Create the confusion matrix
    cm = [
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ]
    
    # Calculate percentages for annotations
    total = sum(sum(row) for row in cm)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": j,
                    "y": i,
                    "text": f"{value}<br>({value/total:.1%})",
                    "font": {"color": "white" if value > total / 10 else "black"},
                    "showarrow": False,
                }
            )
    
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Not Fraud", "Fraud"],
            y=["Not Fraud", "Fraud"],
            colorscale="Blues",
            showscale=False,
        )
    )
    
    # Add annotations
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
    )
    
    return fig

# Define callback to update metrics over time
@app.callback(
    Output("metrics-over-time", "figure"),
    [Input("predictions-store", "data")],
    [State("time-period-dropdown", "value")],
)
def update_metrics_over_time(predictions, days):
    """
    Update the metrics over time chart.
    """
    if not predictions:
        return go.Figure().update_layout(
            title="No data available",
            xaxis_title="Date",
            yaxis_title="Value",
        )
    
    days = int(days) if days else 7
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    if "Timestamp" not in df.columns:
        return go.Figure().update_layout(
            title="No timestamp data available",
            xaxis_title="Date",
            yaxis_title="Value",
        )
    
    # Convert timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    # Group by day
    df["Date"] = df["Timestamp"].dt.date
    
    # Calculate daily metrics
    daily_metrics = []
    
    for date, group in df.groupby("Date"):
        if "Fraud" in group.columns and "FraudPrediction" in group.columns:
            metrics = monitor.calculate_performance_metrics(group)
            daily_metrics.append(
                {
                    "Date": date,
                    "Accuracy": metrics.get("accuracy", None),
                    "Precision": metrics.get("precision", None),
                    "Recall": metrics.get("recall", None),
                    "F1": metrics.get("f1_score", None),
                    "ROC AUC": metrics.get("roc_auc", None),
                    "Fraud Rate": metrics.get("fraud_rate", None),
                    "Alert Rate": metrics.get("alert_rate", None),
                }
            )
    
    if not daily_metrics:
        return go.Figure().update_layout(
            title="No daily metrics available",
            xaxis_title="Date",
            yaxis_title="Value",
        )
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(daily_metrics)
    
    # Create the figure
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Performance Metrics", "Fraud and Alert Rates"),
        vertical_spacing=0.1,
    )
    
    # Add performance metrics
    for metric in ["Precision", "Recall", "F1", "ROC AUC"]:
        if metric in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df["Date"],
                    y=metrics_df[metric],
                    mode="lines+markers",
                    name=metric,
                ),
                row=1,
                col=1,
            )
    
    # Add fraud and alert rates
    for metric in ["Fraud Rate", "Alert Rate"]:
        if metric in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df["Date"],
                    y=metrics_df[metric],
                    mode="lines+markers",
                    name=metric,
                ),
                row=2,
                col=1,
            )
    
    # Update layout
    fig.update_layout(
        title="Metrics Over Time",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Rate", row=2, col=1)
    
    return fig

# Define callback to update data drift chart
@app.callback(
    Output("data-drift-chart", "figure"),
    [Input("predictions-store", "data")],
)
def update_data_drift_chart(predictions):
    """
    Update the data drift chart.
    """
    if not predictions:
        return go.Figure().update_layout(
            title="No data available",
            xaxis_title="Feature",
            yaxis_title="KS Statistic",
        )
    
    try:
        # Load reference data (training data)
        data_dir = monitor.data_dir / "processed"
        
        try:
            # Load the latest version
            with open(data_dir / "latest_version.txt", "r") as f:
                timestamp = f.read().strip()
            
            reference_data = pd.read_csv(data_dir / f"X_train_{timestamp}.csv")
            
            # Convert predictions to DataFrame
            current_data = pd.DataFrame(predictions)
            
            # Remove non-feature columns
            current_data = current_data.drop(
                columns=["FraudPrediction", "FraudScore", "Timestamp", "Fraud"],
                errors="ignore",
            )
            
            # Detect data drift
            drift_results = monitor.detect_data_drift(reference_data, current_data)
            
            if not drift_results or "overall_drift_score" not in drift_results:
                return go.Figure().update_layout(
                    title="No drift data available",
                    xaxis_title="Feature",
                    yaxis_title="KS Statistic",
                )
            
            # Extract drift metrics
            features = []
            ks_stats = []
            drift_detected = []
            
            for feature, result in drift_results.items():
                if feature != "overall_drift_score" and feature != "overall_drift_detected":
                    features.append(feature)
                    ks_stats.append(result["ks_statistic"])
                    drift_detected.append(result["drift_detected"])
            
            # Create DataFrame
            drift_df = pd.DataFrame(
                {
                    "Feature": features,
                    "KS Statistic": ks_stats,
                    "Drift Detected": drift_detected,
                }
            )
            
            # Sort by KS statistic
            drift_df = drift_df.sort_values("KS Statistic", ascending=False)
            
            # Create the figure
            fig = px.bar(
                drift_df,
                x="Feature",
                y="KS Statistic",
                color="Drift Detected",
                color_discrete_map={True: "red", False: "green"},
                labels={"KS Statistic": "KS Statistic", "Drift Detected": "Drift Detected"},
            )
            
            # Add a horizontal line for the threshold
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=0.1,
                x1=len(features) - 0.5,
                y1=0.1,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add annotation for overall drift score
            fig.add_annotation(
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"Overall Drift Score: {drift_results['overall_drift_score']:.2f}",
                showarrow=False,
                font=dict(size=14),
                align="left",
            )
            
            # Update layout
            fig.update_layout(
                title="Feature Drift Detection",
                xaxis_title="Feature",
                yaxis_title="KS Statistic",
                xaxis=dict(tickangle=45),
            )
            
            return fig
            
        except FileNotFoundError:
            return go.Figure().update_layout(
                title="Reference data not found",
                xaxis_title="Feature",
                yaxis_title="KS Statistic",
            )
            
    except Exception as e:
        print(f"Error updating data drift chart: {e}")
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            xaxis_title="Feature",
            yaxis_title="KS Statistic",
        )

# Define callback to update high-risk transactions table
@app.callback(
    Output("high-risk-transactions-table", "children"),
    [Input("predictions-store", "data")],
)
def update_high_risk_transactions_table(predictions):
    """
    Update the high-risk transactions table.
    """
    if not predictions:
        return html.P("No data available")
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    if "FraudScore" not in df.columns:
        return html.P("No fraud score data available")
    
    # Sort by fraud score (descending) and take top 10
    high_risk_df = df.sort_values("FraudScore", ascending=False).head(10)
    
    # Select columns to display
    display_cols = [
        "TransactionID",
        "AccountID",
        "TransactionAmount",
        "TransactionDate",
        "Location",
        "Channel",
        "FraudScore",
        "FraudPrediction",
        "Fraud",
    ]
    
    display_cols = [col for col in display_cols if col in high_risk_df.columns]
    
    if not display_cols:
        return html.P("No columns to display")
    
    # Create the table
    table = dash_table.DataTable(
        id="high-risk-table",
        columns=[{"name": col, "id": col} for col in display_cols],
        data=high_risk_df[display_cols].to_dict("records"),
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(248, 248, 248)",
            },
            {
                "if": {"filter_query": "{FraudPrediction} = 1", "column_id": "FraudPrediction"},
                "backgroundColor": "rgba(255, 0, 0, 0.2)",
                "color": "red",
            },
            {
                "if": {"filter_query": "{Fraud} = 1", "column_id": "Fraud"},
                "backgroundColor": "rgba(255, 0, 0, 0.2)",
                "color": "red",
            },
        ],
    )
    
    return table

# Define callback to generate and display report
@app.callback(
    [Output("report-modal", "is_open"), Output("report-modal-body", "children")],
    [Input("report-button", "n_clicks"), Input("close-report-modal", "n_clicks")],
    [State("report-modal", "is_open"), State("time-period-dropdown", "value")],
)
def toggle_report_modal(n_report, n_close, is_open, days):
    """
    Toggle the report modal and generate a report.
    """
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return is_open, None
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "report-button" and n_report:
        days = int(days) if days else 7
        report = monitor.generate_monitoring_report(days)
        
        # Create report content
        report_content = [
            html.H5(f"Report for the last {days} days"),
            html.P(f"Generated on: {report['timestamp']}"),
            html.Hr(),
            
            html.H6("Performance Metrics"),
            html.Div(
                dash_table.DataTable(
                    columns=[
                        {"name": "Metric", "id": "metric"},
                        {"name": "Value", "id": "value"},
                    ],
                    data=[
                        {"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else v}
                        for k, v in report.get("metrics", {}).items()
                    ],
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold",
                    },
                )
            ),
            html.Hr(),
            
            html.H6("Data Drift"),
            html.P(
                f"Overall Drift Score: {report.get('drift_results', {}).get('overall_drift_score', 'N/A')}"
            ),
            html.P(
                f"Drift Detected: {report.get('drift_results', {}).get('overall_drift_detected', 'N/A')}"
            ),
            html.Hr(),
            
            html.H6("Statistics"),
            html.P(f"Predictions Count: {report.get('predictions_count', 0)}"),
            html.P(f"Feedback Count: {report.get('feedback_count', 0)}"),
        ]
        
        return True, report_content
    
    elif button_id == "close-report-modal" and n_close:
        return False, None
    
    return is_open, None

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
