# main.py  (package: F1_prediction_system)
from __future__ import annotations
import argparse
import pandas as pd

from .config import HIST_YEARS
from .data import build_training_until as build_until_data, get_target_drivers
from .features import (
    add_circuit_context_df, add_driver_team_form, merge_latest_forms, add_quali_proxy
)
from .model import train_model, predict_event_with_uncertainty, oob_errors


def build_training_frame(target_year: int, target_gp: str) -> pd.DataFrame:
    """Pull history + season-to-date, then add forms + circuit context."""
    train_df = build_until_data(target_year, target_gp, hist_years=HIST_YEARS)
    train_df = add_driver_team_form(train_df)
    train_df = add_circuit_context_df(train_df)
    return train_df


def build_predict_frame(target_year: int, target_gp: str, train_df_with_forms: pd.DataFrame) -> pd.DataFrame:
    """Get target drivers (Q if available; else FP1/fallback), add context + latest forms."""
    pred_df = get_target_drivers(target_year, target_gp)
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
    parser.add_argument("--preq", action="store_true",
                        help="Force pre-qualifying mode (ignore Q and use quali proxy)")
    parser.add_argument("--proxy_window", type=int, default=3,
                        help="Window for quali proxy rolling mean (when --preq or grid unknown)")
    parser.add_argument("--mc", type=int, default=500,
                        help="Monte-Carlo samples for rank probabilities (0 disables)")
    parser.add_argument("--interval", type=int, choices=(68, 95), default=68,
                        help="Confidence interval width to display in console (68 or 95)")
    args = parser.parse_args()

    target_year, target_gp = args.year, args.gp

    # 1) Build training set
    print(f"[INFO] Building training frame up to {target_gp} {target_year}…")
    train_df = build_training_frame(target_year, target_gp)
    print(f"[INFO] Training rows: {train_df.shape[0]}")

    # 2) Train model
    print("[INFO] Training model…")
    model = train_model(train_df)

    # Optional: Out-of-bag diagnostics
    errs = oob_errors(model, train_df)
    if errs:
        print(f"[OOB] R2={errs['oob_r2']:.3f} | MAE={errs['oob_mae']:.2f} | RMSE={errs['oob_rmse']:.2f}")
    else:
        print("[OOB] Not available")

    # 3) Build prediction frame
    print(f"[INFO] Building prediction frame for {target_gp} {target_year}…")
    pred_df = build_predict_frame(target_year, target_gp, train_df)

    # If user explicitly wants pre-Q behavior, overwrite grid and compute proxy
    if args.preq:
        proxy_base = train_df[["driver", "date", "grid_pos"]].dropna()
        pred_df.loc[:, "grid_pos"] = pd.NA  # force missing so proxy applies to all
        pred_df = add_quali_proxy(pred_df, proxy_base, window=args.proxy_window)

    # Sanity checks
    if pred_df.empty:
        raise RuntimeError("Prediction frame is empty; no driver list available.")
    for col in ("driver", "team", "grid_pos"):
        if col not in pred_df.columns:
            raise RuntimeError(f"Prediction frame missing required column: {col}")

    # 4) Predict with uncertainty (adds std, 68/95% intervals; MC adds p_top10/p_podium/etc.)
    print("[INFO] Predicting order…")
    out = predict_event_with_uncertainty(
        model,
        pred_df,
        
    )

    # 5) Print Top-10 with chosen interval
    # Map which interval columns to show
    if args.interval == 95:
        lo_col, hi_col = "pi95_low", "pi95_high"
    else:
        lo_col, hi_col = "pi68_low", "pi68_high"

    cols_to_print = [c for c in (
        "driver", "team", "grid_pos",
        "pred_finish", "pred_rank", "pred_std",
        lo_col, hi_col,
        "p_top10", "p_podium", "p_rank_pm1"
    ) if c in out.columns]

    print("\nPredicted Top 10:")
    print(out[cols_to_print].head(10).to_string(index=False))

    # 6) Save full table
    out.to_csv("predicted_order.csv", index=False)
    print("\n[INFO] Saved full predictions to predicted_order.csv")


if __name__ == "__main__":
    main()
