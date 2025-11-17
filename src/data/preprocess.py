import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from ..config import PATHS, RANDOM_STATE, TEST_SIZE


def load_raw(path: str) -> pd.DataFrame:
    """Loads raw CSV file."""
    return pd.read_csv(path)


def build_preprocessing_pipeline():
    """Creates preprocessing pipeline for numeric + categorical features."""

    numeric_features = ["area", "price_per_sqft", "BHK", "bathrooms"]
    categorical_features = ["locality", "facing", "parking"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def preprocess_and_split(df: pd.DataFrame, target_col: str):
    """Preprocesses data and returns train/test splits + saves artifacts."""

    df = df.copy()

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Build pipeline
    preprocessor, numeric_features, categorical_features = build_preprocessing_pipeline()

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit only on training
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save processed datasets
    train_df = pd.DataFrame(X_train_processed)
    train_df["target"] = y_train.reset_index(drop=True)
    train_df.to_csv(PATHS.processed_train, index=False)

    test_df = pd.DataFrame(X_test_processed)
    test_df["target"] = y_test.reset_index(drop=True)
    test_df.to_csv(PATHS.processed_test, index=False)

    # Save preprocessing pipeline
    joblib.dump(preprocessor, PATHS.pipeline_artifact)

    return (
        X_train_processed,
        X_test_processed,
        y_train.values,
        y_test.values,
        preprocessor
    )
