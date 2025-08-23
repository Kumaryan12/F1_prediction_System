import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

FEATS = [
    "grid_pos",
    "sc_prob", "vsc_prob", "pit_loss",
    "drv_form3", "team_form3",
    "team", "driver"
]

def _prep_fe_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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

    rf = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,           # <-- enable OOB generalization estimate
    )

    model = Pipeline([
        ("prep", pre),
        ("rf", rf)
    ])
    model.fit(X, y)
    return model

def predict_event(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """(kept for compatibility) mean prediction only."""
    X, _ = _prep_fe_matrix(features_df.copy())
    preds = model.predict(X)
    out = features_df[["driver", "team", "grid_pos"]].copy()
    out["pred_finish"] = preds
    out = out.sort_values("pred_finish").reset_index(drop=True)
    out["pred_rank"] = np.arange(1, len(out) + 1)
    return out

def predict_event_with_uncertainty(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """Mean prediction + 1σ spread from RF ensemble (≈68% CI)."""
    X, _ = _prep_fe_matrix(features_df.copy())
    prep = model.named_steps["prep"]
    rf   = model.named_steps["rf"]

    Xmat = prep.transform(X)
    # Tree estimators expect dense; guard if the transformer is sparse
    if hasattr(Xmat, "toarray"):
        Xmat = Xmat.toarray()

    tree_preds = np.column_stack([est.predict(Xmat) for est in rf.estimators_])  # (n_samples, n_trees)
    mean = tree_preds.mean(axis=1)
    std  = tree_preds.std(axis=1)

    ci_lo = np.clip(mean - std, 1, 20)
    ci_hi = np.clip(mean + std, 1, 20)

    out = features_df[["driver", "team", "grid_pos"]].copy()
    out["pred_finish"] = mean
    out["pred_std"] = std
    out["ci68_low"] = ci_lo
    out["ci68_high"] = ci_hi
    # Heuristic confidence: lower std → higher confidence
    out["confidence_%"] = (1.0 / (1.0 + out["pred_std"])) * 100.0

    out = out.sort_values("pred_finish").reset_index(drop=True)
    out["pred_rank"] = np.arange(1, len(out) + 1)
    return out

def oob_errors(model, train_df: pd.DataFrame):
    """Return OOB metrics if available: R2, MAE, RMSE (bagging-based)."""
    rf = model.named_steps["rf"]
    if not hasattr(rf, "oob_prediction_") or rf.oob_prediction_ is None:
        return None

    y_true = train_df.dropna(subset=["finish_pos"])["finish_pos"].astype(float).to_numpy()
    y_oob  = rf.oob_prediction_
    if len(y_oob) != len(y_true):
        return None

    mae  = float(np.mean(np.abs(y_true - y_oob)))
    rmse = float(np.sqrt(np.mean((y_true - y_oob) ** 2)))
    r2   = float(getattr(rf, "oob_score_", np.nan))
    return {"oob_r2": r2, "oob_mae": mae, "oob_rmse": rmse}
