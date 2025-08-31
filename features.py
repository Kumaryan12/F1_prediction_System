from __future__ import annotations
import pandas as pd
import numpy as np
from .config import CIRCUIT_VOL, DEFAULT_SC, DEFAULT_VSC, DEFAULT_PIT_LOSS

def add_circuit_context_df(df: pd.DataFrame) -> pd.DataFrame:

    def _lookup(gp: str):
        sc, vsc, pit  = CIRCUIT_VOL.get(gp, (DEFAULT_SC, DEFAULT_VSC, DEFAULT_PIT_LOSS))
        return pd.Series({"sc_prob":sc, "vsc_prob": vsc, "pit_loss": pit})
    
    ctx = df["gp"].apply(_lookup)
    return pd.concat([df.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)

def add_driver_team_form(full_df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling form features.
    Expects: year, gp, date, driver, team, finish_pos
    """
    if "date" not in full_df.columns:
        raise ValueError("full_df must include a 'date' column to sort chronologically")

    # Work on a clean, ordered copy
    df = full_df.sort_values(["date", "year", "gp"]).reset_index(drop=True).copy()

    # Driver 3-race rolling average of finish (shifted to avoid leakage)
    df["drv_form3"] = (
        df.groupby("driver", sort=False)["finish_pos"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # Team event mean per race, then 3-race rolling avg (shifted)
    team_ev = (
        df.groupby(["year", "gp", "date", "team"])["finish_pos"]
          .mean()
          .reset_index(name="team_ev_mean")
    )
    df = df.merge(team_ev, on=["year", "gp", "date", "team"], how="left")

    df["team_form3"] = (
        df.groupby("team", sort=False)["team_ev_mean"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    return df.drop(columns=["team_ev_mean"])


def merge_latest_forms(predict_df: pd.DataFrame, train_df_with_forms: pd.DataFrame) -> pd.DataFrame:
    # pick the last (most recent) row per (driver, team) in the historical frame
    latest = (train_df_with_forms.sort_values("date")
              .groupby(["driver", "team"], as_index=False)
              .tail(1))

    # only keep the columns we need, with the right names
    needed = ["driver", "team", "drv_form3", "team_form3"]
    latest = latest[[c for c in needed if c in latest.columns]]

    out = predict_df.merge(latest, on=["driver", "team"], how="left")

    # fallbacks in case a rookie/sub has no history yet
    for col in ("drv_form3", "team_form3"):
        if col not in out.columns:
            out[col] = pd.NA
        if out[col].isna().any():
            # Fill with global median from the training data as a conservative default
            if col in train_df_with_forms.columns:
                med = train_df_with_forms[col].median(skipna=True)
                out[col] = out[col].fillna(med)

    return out



def add_quali_proxy(predict_df: pd.DataFrame, train_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Adds a qualifying proxy if the grid position is missing in the prediction data.
    """
    # Check if there are any missing grid positions
    missing_grid = predict_df['grid_pos'].isna().sum()

    # If grid positions are missing, apply proxy
    if missing_grid > 0:
        print(f"Missing {missing_grid} grid positions, applying quali proxy")
        tmp = (train_df.sort_values("date")
                .groupby("driver")["grid_pos"]
                .apply(lambda s: s.tail(window).mean())
                .reset_index().rename(columns={"grid_pos": "qual_proxy"}))

        # Merge the proxy values into the prediction dataframe
        out = predict_df.merge(tmp, on="driver", how="left")
        out["grid_pos"] = out["grid_pos"].fillna(out["qual_proxy"])  # Fills missing grid_pos with proxy
        out = out.drop(columns=["qual_proxy"])
        return out
    else:
        print("All grid positions are available, no proxy needed")
        return predict_df  # No proxy applied, return as is



    
