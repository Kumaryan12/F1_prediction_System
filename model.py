from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn. ensemble import RandomForestRegressor
from typing import Tuple

FEATS = [
    "grid_pos",
    "sc_prob", "vsc_prob", "pit_loss",
    "drv_form3", "team_form3", 
    "team", "driver"
]

def _prep_fe_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Work on a copy; coerce numeric/categorical dtypes we expect
    df = df.copy()

    # Numeric features we might have
    num_maybe = ["grid_pos", "drv_form3", "team_form3", "sc_prob", "vsc_prob", "pit_loss"]
    for c in num_maybe:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categorical features we might have
    for c in ("team", "driver"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Build the exact feature list; create missing ones as NaN so the imputer can handle them
    present = [c for c in FEATS if c in df.columns]
    missing = [c for c in FEATS if c not in df.columns]
    for m in missing:
        df[m] = np.nan

    feat_list = present + missing
    return df[feat_list], feat_list

    

def train_model(train_df: pd.DataFrame):
    # Drop rows without targets; keep a clean copy
    df = train_df.copy()
    df = df.dropna(subset=["finish_pos"])
    # (optional) also drop rows with completely missing grid if you want
    # df = df.dropna(subset=["grid_pos"])

    X, feat_list = _prep_fe_matrix(df)
    y = df["finish_pos"].astype(float)

    cat_cols = [c for c in feat_list if c in ("team", "driver")]
    num_cols = [c for c in feat_list if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline([
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1))
    ])
    model.fit(X, y)
    return model

def predict_event(model, features_df: pd.DataFrame) -> pd.DataFrame:
    X, _ = _prep_fe_matrix(features_df.copy())
    preds = model.predict(X)
    out=features_df[["driver", "team", "grid_pos"]].copy()
    out["pred_finish"]= preds
    out = out.sort_values("pred_finish").reset_index(drop = True)
    out["pred_rank"] = np.arange(1, len(out) + 1)
    return out
