# main.py  (for package name: F1_prediction_system)
from __future__ import annotations
import argparse
import pandas as pd

from .config import HIST_YEARS
from .data import build_training_until as build_until_data, get_target_drivers
from .features import (
    add_circuit_context_df, add_driver_team_form, merge_latest_forms, add_quali_proxy
)
from .model import train_model, predict_event


def build_training_frame(target_year: int, target_gp: str) -> pd.DataFrame:
    """Wrapper that pulls history + season-to-date, then adds forms + circuit context."""
    train_df = build_until_data(target_year, target_gp, hist_years=HIST_YEARS)
    train_df = add_driver_team_form(train_df)
    train_df = add_circuit_context_df(train_df)
    return train_df


def build_predict_frame(target_year: int, target_gp: str, train_df_with_forms: pd.DataFrame) -> pd.DataFrame:
    pred_df = get_target_drivers(target_year, target_gp)       # Q if available; else FP1
    pred_df = add_circuit_context_df(pred_df)
    pred_df = merge_latest_forms(pred_df, train_df_with_forms)

    # If grid is unknown (pre-Q), use quali proxy from recent races
    if pred_df["grid_pos"].isna().any():
        proxy_base = train_df_with_forms[["driver", "date", "grid_pos"]].dropna()
        pred_df = add_quali_proxy(pred_df, proxy_base, window=3)
    return pred_df


def main():
    parser = argparse.ArgumentParser(description="F1 Race Predictor (RF baseline)")
    parser.add_argument("--year", type=int, default=2025, help="Target season, e.g., 2025")
    parser.add_argument("--gp", type=str, default="Dutch Grand Prix", help="Target GP name")
    parser.add_argument("--preq", action="store_true", help="Force pre-qualifying mode (always use proxy)")
    args = parser.parse_args()

    target_year, target_gp = args.year, args.gp

    print(f"[INFO] Building training frame up to {target_gp} {target_year}…")
    train_df = build_training_frame(target_year, target_gp)
    print(f"[INFO] Training rows: {train_df.shape[0]}")

    print("[INFO] Training model…")
    model = train_model(train_df)

    print(f"[INFO] Building prediction frame for {target_gp} {target_year}…")
    pred_df = build_predict_frame(target_year, target_gp, train_df)

    if args.preq:
        proxy_base = train_df[["driver", "date", "grid_pos"]].dropna()
        pred_df = add_quali_proxy(pred_df, proxy_base, window=3)

    print("[INFO] Predicting order…")
    out = predict_event(model, pred_df)

    print("\nPredicted Top 10:")
    print(out.head(10).to_string(index=False))

    out.to_csv("predicted_order.csv", index=False)
    print("\n[INFO] Saved full predictions to predicted_order.csv")


if __name__ == "__main__":
    main()
