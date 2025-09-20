# model_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import numpy as np
# ----------------------------
# Data container
# ----------------------------
@dataclass
class DataBundle:
    # splits
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    # feature groups
    numeric: List[str]
    categorical: List[str]
    boolean: List[str]
    # ready-to-use preprocessor built during preparation
    preprocessor: ColumnTransformer


# ----------------------------
# Helpers
# ----------------------------
def _is_bool_series(s: pd.Series) -> bool:
    if s.dtype == bool:
        return True
    vals = set(map(str, s.dropna().unique()))
    return vals.issubset({"True", "False", "true", "false", "0", "1"})


def _build_preprocessor(numeric: List[str], categorical: List[str], boolean: List[str]) -> ColumnTransformer:
    """
    Build a preprocessor that:
      - imputes numerics + scales (optional but fine to keep)
      - imputes categoricals/booleans + label-encodes (OrdinalEncoder)
    OrdinalEncoder is the multi-column 'LabelEncoder' and avoids exploding columns like OHE.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),   # fine with tree/linear; remove if you prefer
    ])

    # OrdinalEncoder maps categories to integers [0..K-1].
    # We set unknown and missing to -1 so inference on unseen values does not crash.
    cat_ord = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", cat_ord),
    ])

    bool_ord = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    bool_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", bool_ord),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  num_pipe,  numeric),
            ("bool", bool_pipe, boolean),
            ("cat",  cat_pipe,  categorical),
        ],
        remainder="drop",
        # output is dense but compact (no OHE explosion)
    )
    return preprocessor


# ----------------------------
# 1) prepare_data()  (NOW builds the preprocessor)
# ----------------------------
def prepare_data(
    csv_path: str,
    target: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    build_churn_from_playtime_2w: bool = True,
) -> DataBundle:
    """
    Load CSV, (optionally) build churn target from playtime_2weeks, CLEAN FEATURES,
    INFER feature groups, BUILD PREPROCESSOR, and split train/test.

    Churn (no leakage):
      churn = 1 if playtime_2weeks > 0 else 0
    Then drop playtime_2weeks and median_playtime_2weeks from features.
    """
    df = pd.read_csv(csv_path)

    # Create churn from playtime_2weeks if needed
    if build_churn_from_playtime_2w:
        if "playtime_2weeks" not in df.columns:
            raise ValueError("Column 'playtime_2weeks' missing for churn construction.")
        df[target] = (pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0) > 0).astype(int)

    assert target in df.columns, f"Target column '{target}' not found."

    # Columns never used as features (IDs + leakage)
    always_drop = {
        "game_id", "name", "release_date",
        "playtime_2weeks", "median_playtime_2weeks", target,
    }
    if drop_cols:
        always_drop |= set(drop_cols)

    y = df[target]
    X = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")

    # Remove all-NaN / constant columns
    to_drop = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
    if to_drop:
        X = X.drop(columns=to_drop)

    # Infer feature groups
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    boolean = [c for c in X.columns if c not in numeric and _is_bool_series(X[c])]
    categorical = [c for c in X.columns if c not in numeric and c not in boolean]

    # Build preprocessor HERE during preparation
    preprocessor = _build_preprocessor(numeric, categorical, boolean)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return DataBundle(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric=numeric, categorical=categorical, boolean=boolean,
        preprocessor=preprocessor,
    )


# ----------------------------
# 2) train_model()  (uses preprocessor built in preparation)
# ----------------------------
def train_model(
    data: DataBundle,
    model: Optional[object] = None,
) -> Pipeline:
    """
    Fit a pipeline = (preprocessor from prepare_data) + classifier.
    """
    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    pipe = Pipeline(steps=[("prep", data.preprocessor), ("model", model)])
    pipe.fit(data.X_train, data.y_train)
    return pipe


# ----------------------------
# 3) evaluate_model()
# ----------------------------
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred, digits=4),
    }


# ----------------------------
# 4) save_model() / 5) load_model()
# ----------------------------
def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)