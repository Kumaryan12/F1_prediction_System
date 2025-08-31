# main.py  (package: F1_prediction_system)
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import fastf1

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
        print("Qualifying data missing, using proxy...")
        proxy_base = train_df_with_forms[["driver", "date", "grid_pos"]].dropna()
        pred_df = add_quali_proxy(pred_df, proxy_base, window=3)

    return pred_df

    


def _safe_load_model(path: str):
    """Try loading rich artifact; fall back to plain joblib pipeline."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    try:
        # Prefer artifact (model + meta) if available
        from .model import load_model_artifact  # type: ignore
        model, meta = load_model_artifact(str(p))
        return model, meta
    except Exception:
        # Fallback: a plain joblib Pipeline without meta
        model = joblib.load(p)
        return model, {}


def _safe_save_model(model, path: str, meta: dict | None = None):
    """Try saving rich artifact; fall back to plain joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .model import save_model_artifact  # type: ignore
        save_model_artifact(model, str(p), meta or {})
    except Exception:
        joblib.dump(model, p)
    return p.resolve()


def main():
    parser = argparse.ArgumentParser(description="F1 Race Predictor (RF + engineered features)")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--gp", type=str, default="Dutch Grand Prix")

    # Quali proxy controls
    parser.add_argument("--preq", action="store_true",
                        help="Force pre-qualifying mode (ignore Q and use quali proxy)")
    parser.add_argument("--proxy_window", type=int, default=3,
                        help="Window for quali proxy rolling mean (when --preq or grid unknown)")

    # Uncertainty controls
    parser.add_argument("--mc", type=int, default=500,
                        help="Monte-Carlo samples for rank probabilities (0 disables)")
    parser.add_argument("--interval", type=int, choices=(68, 95), default=68,
                        help="Confidence interval width to display in console (68 or 95)")

    # Optional data/features (reserved)
    parser.add_argument("--weather_csv", type=str, default=None,
                        help="CSV with gp,year,date,rain_prob,track_temp_c (optional; not wired yet)")

    # Model persistence / reuse
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a saved model .joblib (artifact or plain Pipeline)")
    parser.add_argument("--save_model", type=str, default=None,
                        help="Where to save the trained model .joblib")
    parser.add_argument("--auto_retrain", action="store_true",
                        help="If a loaded model is stale (newer data or feature mismatch), retrain.")
    parser.add_argument("--force_load", action="store_true",
                        help="Use the loaded model even if features changed (not recommended).")

    # Conformal (reserved)
    parser.add_argument("--use_conformal", action="store_true",
                        help="Add split-conformal PIs (reserved; not wired yet)")
    parser.add_argument("--alpha", type=float, default=0.20,
                        help="Conformal alpha (default 0.20 ~ 80% PI).")

    args = parser.parse_args()
    target_year, target_gp = args.year, args.gp

    # 1) Build training data (needed for forms/context & feature checks)
    print(f"[INFO] Building training frame up to {target_gp} {target_year}…")
    train_df = build_training_frame(target_year, target_gp)
    print(f"[INFO] Training rows: {train_df.shape[0]}")

    # 2) Load or train
    model = None
    loaded_meta = {}

    if args.load_model:
        try:
            model, loaded_meta = _safe_load_model(args.load_model)
            print(f"[INFO] Loaded model from {Path(args.load_model).resolve()}")

            # Optional: feature compatibility + staleness check (only if artifact had meta)
            try:
                from .model import _prep_fe_matrix  # type: ignore
                X_now, feat_list_now = _prep_fe_matrix(train_df.dropna(subset=["finish_pos"]).copy())
                feat_list_saved = set(loaded_meta.get("feat_list", []))
                train_end_saved = pd.to_datetime(loaded_meta.get("train_end_date"))
                train_end_now = pd.to_datetime(train_df["date"]).max()

                feature_mismatch = bool(feat_list_saved) and (feat_list_saved != set(feat_list_now))
                data_stale = (pd.notna(train_end_saved) and pd.notna(train_end_now)
                              and train_end_now > train_end_saved)

                if feature_mismatch:
                    msg = "[WARN] Loaded model feature set differs from current pipeline."
                    if args.force_load:
                        print(msg + " Proceeding due to --force_load.")
                    elif args.auto_retrain:
                        print(msg + " Retraining due to --auto_retrain.")
                        model = None
                    else:
                        print(msg + " Consider retraining (or pass --auto_retrain).")

                if model is not None and data_stale and args.auto_retrain:
                    print("[INFO] Newer training data exists. Retraining due to --auto_retrain.")
                    model = None
            except Exception:
                # If helper not available, just proceed with the loaded pipeline
                pass

        except Exception as e:
            print(f"[WARN] Failed to load model: {e}. Will train a fresh model.")

    if model is None:
        print("[INFO] Training model…")
        model = train_model(train_df)

        errs = oob_errors(model, train_df)
        if errs:
            print(f"[OOB] R2={errs['oob_r2']:.3f} | MAE={errs['oob_mae']:.2f} | RMSE={errs['oob_rmse']:.2f}")
        else:
            print("[OOB] Not available")

        if args.save_model:
            # Build metadata for artifact
            try:
                from .model import _prep_fe_matrix  # type: ignore
                _, feat_list_now = _prep_fe_matrix(train_df.dropna(subset=["finish_pos"]).copy())
            except Exception:
                feat_list_now = []
            
            if isinstance(HIST_YEARS, (list, tuple)):
               hist_years_meta = list(HIST_YEARS)          # store the actual years, e.g. [2023, 2024, 2025]
            else:
              hist_years_meta = int(HIST_YEARS) if isinstance(HIST_YEARS, (int, float)) else str(HIST_YEARS)

            meta = {
                "feat_list": feat_list_now,
                "train_rows": int(train_df.dropna(subset=["finish_pos"]).shape[0]),
                "train_start_date": str(pd.to_datetime(train_df["date"]).min()),
                "train_end_date": str(pd.to_datetime(train_df["date"]).max()),
                "hist_years": list(HIST_YEARS),
                "target_context": {"year": target_year, "gp": target_gp},
                "oob": oob_errors(model, train_df) or {},
                "model": "RandomForestRegressor(n_estimators=500, random_state=42)",
                "code_version": "v1",
            }
            saved_path = _safe_save_model(model, args.save_model, meta)
            print(f"[INFO] Saved model to {saved_path}")
    else:
        
        errs = oob_errors(model, train_df)
        if errs:
            print(f"[OOB] R2={errs['oob_r2']:.3f} | MAE={errs['oob_mae']:.2f} | RMSE={errs['oob_rmse']:.2f}")

    # 3) Build prediction frame
    print(f"[INFO] Building prediction frame for {target_gp} {target_year}…")
    pred_df = build_predict_frame(target_year, target_gp, train_df)

    
    # Optional: Pre-Q behavior 
    if args.preq:
        try:
            session = fastf1.get_session(args.year, args.gp, 'Q')
            session.load()  # Loads the session data

        # Check if qualifying data is available
            if session.laps.empty:
             raise ValueError(f"No qualifying data available for {args.gp} {args.year}.")

        # Map the actual grid positions to the DataFrame
            pred_df = pred_df.copy()
            pred_df["grid_pos"] = pred_df["driver"].map(dict(zip(session.laps['Driver'], session.laps['GridPosition'])))

        except Exception as e:
         print(f"[WARNING] Failed to load qualifying data for {args.gp} {args.year}. Error: {e}")
         # If qualifying data is not available, fallback to proxy
         print(f"Using qualifying proxy for {args.gp} {args.year}.")
         proxy_base = train_df[["driver", "date", "grid_pos"]].dropna()
         pred_df = add_quali_proxy(pred_df, proxy_base, window=args.proxy_window)

    # Sanity checks
    if pred_df.empty:
        raise RuntimeError("Prediction frame is empty; no driver list available.")
    for col in ("driver", "team", "grid_pos"):
        if col not in pred_df.columns:
            raise RuntimeError(f"Prediction frame missing required column: {col}")

    # 4) Predict with uncertainty (adds std, 68/95% intervals; MC adds rank probabilities)
    print("[INFO] Predicting order…")
    out = predict_event_with_uncertainty(
        model,
        pred_df,
        add_intervals=True,
        mc_samples=args.mc
    )

    # 5) Print Top-10 with chosen interval
    lo_col, hi_col = ("pi95_low", "pi95_high") if args.interval == 95 else ("pi68_low", "pi68_high")
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

    # Reserved flags
    if args.weather_csv:
        print("[NOTE] --weather_csv provided, but automatic merge is not wired yet.")
    if args.use_conformal:
        print("[NOTE] --use_conformal requested, but conformal intervals are not wired yet.")


if __name__ == "__main__":
    main()
