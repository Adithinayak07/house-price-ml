import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = (mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = r2_score(y_true, y_pred)

    df_compare = pd.DataFrame({
        "Actual Rent": y_true,
        "Predicted Rent": y_pred,
        "Error": y_true - y_pred
    })

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "comparison": df_compare
    }
