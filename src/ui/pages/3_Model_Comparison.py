import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px
import os

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "house_rent_experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("📊 Model Comparison Dashboard")

# Load MLflow experiment
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if not exp:
    st.error("❌ No experiment found. Train the model first.")
    st.stop()

# Load runs
runs = mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"])

if runs.empty:
    st.warning("⚠ No MLflow runs found.")
    st.stop()

# Clean table
df = runs[["run_id", "start_time", 
           "metrics.rmse", "metrics.mae", "metrics.r2"]].copy()

df.columns = ["Run ID", "Start Time", "RMSE", "MAE", "R²"]

st.subheader("📋 MLflow Runs Overview")
st.dataframe(df, use_container_width=True)

st.divider()

# -----------------------------
# Best Run Summary
# -----------------------------
best_run = df.sort_values("RMSE").iloc[0]

st.subheader("🏆 Best Model Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Best RMSE", f"{best_run['RMSE']:.2f}")
col2.metric("Best MAE", f"{best_run['MAE']:.2f}")
col3.metric("Best R²", f"{best_run['R²']:.4f}")

st.divider()

# -----------------------------
# Line charts for trends
# -----------------------------
st.subheader("📈 Performance Trend Over Time")

trend_plot = px.line(
    df.sort_values("Start Time"),
    x="Start Time",
    y=["RMSE", "MAE", "R²"],
    markers=True,
    title="Metrics Over Training Runs"
)
st.plotly_chart(trend_plot, use_container_width=True)

st.divider()

# -----------------------------
# Bar chart for runs
# -----------------------------
st.subheader("📊 Metrics Comparison Across Runs")

bar_plot = px.bar(
    df,
    x="Run ID",
    y=["RMSE", "MAE", "R²"],
    barmode="group",
    title="Comparison of RMSE, MAE, and R² Across Runs"
)
st.plotly_chart(bar_plot, use_container_width=True)

st.success("Comparison loaded successfully ✔")
