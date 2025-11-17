import streamlit as st
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import os
import requests

import os
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house_rent_experiment")
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "house_rent_experiment"



mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


DATA_PATH = "data/raw/houseprices.csv"

try:
    df = pd.read_csv(DATA_PATH)

    # Clean locality column
    locality_options = (
        df["locality"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    locality_options = sorted(locality_options)

except Exception as e:
    st.error(f"❌ Could not load locality options from dataset: {e}")
    locality_options = []

# ------------------- USER INPUT FORM -------------------
# ------------------- FIXED OPTIONS FROM USER -------------------

locality_options = [
    "Attibele",
    "BTM Layout",
    "Electronic City",
    "Indiranagar",
    "Jayanagar",
    "K R Puram",
    "Malleshwaram",
    "Marathahalli",
    "Yalahanka"
]

facing_options = [
    "East",
    "North",
    "West",
    "South",
    "North-East",
    "North-West",
    "South-East"
]

parking_options = [
    "Bike and Car",
    "Car",
    "Bike"
]

# ------------------- USER INPUT FORM -------------------
with st.form("input_form"):
    st.subheader("📝 House Details")

    locality = st.selectbox("Locality", options=locality_options)

    facing = st.selectbox("Facing Direction", options=facing_options)

    parking = st.selectbox("Parking", options=parking_options)

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (Sq Ft)", min_value=200, max_value=10000, step=10, value=1200)
        BHK = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1, value=2)

    with col2:
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
        price_per_sqft = st.number_input("Price Per Sq Ft", min_value=1, max_value=500, step=1, value=40)

    submitted = st.form_submit_button("Predict Rent")

# ------------------- PREDICTION -------------------
if submitted:
    st.subheader("🔍 Prediction Result")

    input_data = {
        "locality": locality,
        "area": area,
        "price_per_sqft": price_per_sqft,
        "facing": facing,
        "BHK": BHK,
        "bathrooms": bathrooms,
        "parking": parking
    }

    try:
        response = requests.post(
            "http://localhost:8000/predict",  
            json={"data": input_data}
        )
        if response.ok:
            predicted_rent = response.json()["prediction"]
            st.success(f"🏠 **Estimated Rent: ₹ {predicted_rent:,.0f} / month**")
        else:
            st.error("Prediction failed. Check your API server.")
    except Exception as e:
        st.error(f"Error connecting to FastAPI server: {e}")

# ------------------- MLflow Experiment Insights -------------------
st.subheader("📊 Latest model runs (MLflow)")

try:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp:
        runs = mlflow.search_runs(exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=5)
        st.dataframe(
            runs[["run_id", "metrics.rmse", "metrics.mae", "metrics.r2"]],
            hide_index=True
        )
    else:
        st.info("No MLflow experiment found yet!")
except Exception as e:
    st.warning(f"Could not fetch MLflow data: {e}")
