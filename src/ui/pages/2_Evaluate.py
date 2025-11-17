import streamlit as st
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import os
import joblib

# -----------------------------
# MLflow Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "house_rent_experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -----------------------------
# Page Title
# -----------------------------
st.title("📊 Model Evaluation Dashboard")

# -----------------------------
# Load Experiment FIRST
# -----------------------------
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

st.markdown(f"**Tracking URI:** `{MLFLOW_TRACKING_URI}`")

if exp:
    st.markdown(f"**Experiment Loaded:** `{exp.name}`  (ID: {exp.experiment_id})")
else:
    st.error("❌ MLflow experiment not found. Train a model first.")
    st.stop()

# -----------------------------
# Load Latest Runs
# -----------------------------
runs = mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"])

if runs.empty:
    st.warning("⚠ No runs found in MLflow. Train the model first.")
    st.stop()

st.write("### ✅ Latest Training Run Metrics")

latest = runs.iloc[0]  # Best/latest run

# Show metrics cleanly
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{latest['metrics.mae']:.2f}")
col2.metric("RMSE", f"{latest['metrics.rmse']:.2f}")
col3.metric("R² Score", f"{latest['metrics.r2']:.4f}")

st.divider()

# -----------------------------
# Evaluation on Test Data
# -----------------------------
st.write("### 📈 Actual vs Predicted (Test Data)")

test_path = "data/processed/test.csv"

if not os.path.exists(test_path):
    st.error("❌ Test dataset not found. Train the model first.")
    st.stop()

df = pd.read_csv(test_path)

y_test = df["target"]
X_test = df.drop(columns=["target"])

# Load model & pipeline
pipeline_path = "models/preprocessing_pipeline.joblib"
model_path = "models/linear_model.joblib"

if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
    st.error("❌ Model or preprocessing pipeline missing. Train again.")
    st.stop()

pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)

# Make predictions
predictions = model.predict(X_test)

# ----- Scatter Plot -----
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Rent")
ax.set_ylabel("Predicted Rent")
ax.set_title("Actual vs Predicted Rent")
st.pyplot(fig)

# ----- Residual Plot -----
st.write("### 🔥 Residuals Distribution")

residuals = y_test - predictions
st.line_chart(residuals)

# Comparison Table
st.write("### 📋 Actual vs Predicted Table")
comparison_df = pd.DataFrame({
    "Actual Rent": y_test,
    "Predicted Rent": predictions,
    "Error": residuals
})
st.dataframe(comparison_df)
