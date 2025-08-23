from __future__ import annotations
import pandas as pd
import fastf1
from typing import List, Tuple
from .config import CACHE_DIR, FALLBACK_EVENTS, EXCLUDE_EVENTS



fastf1.Cache.enable_cache(CACHE_DIR)

def _event_schedule(year: int) -> pd.DataFrame:
    try:
     
     sch = fastf1.get_event_schedule(year)
     if "EventFormat" in sch.columns:
         sch = sch[sch["EventFormat"].str.lower() != "testing"]
     sch = sch[~sch["EventName"].str.contains("testing", case=False, na=False)]
     return sch[["EventName", "EventDate"]].rename(columns={"EventName": "gp", "EventDate": "date"})
    
    except Exception:
        events = FALLBACK_EVENTS.get(year, [])
        if not events:
            raise
        dates = pd.date_range(f"{year}-01-01", periods=len(events), freq="7D")
        return pd.DataFrame({"gp": events, "date": dates})    
    

def _race_has_results(year: int, gp: str) -> bool:
    """Return True only if the race session has real results (i.e., it has been run)."""
    try:
        ses = fastf1.get_session(year, gp, "R")
        # results-only load; avoids heavy timing/telemetry endpoints
        ses.load(telemetry=False, laps=False, weather=False, messages=False)
        res = ses.results
        return (res is not None) and (not res.empty)
    except Exception:
        return False
    
def _load_results_only(year: int, gp: str, sess_name: str) -> pd.DataFrame:
    """Return .results for a session, without laps/telemetry/etc. Raise if missing/empty."""
    ses = fastf1.get_session(year, gp, sess_name)
    # results-only; avoid heavy endpoints that hang
    try:
        ses.load(telemetry=False, laps=False, weather=False, messages=False)
    except TypeError:
        try:
            ses.load(telemetry=False, laps=False)
        except TypeError:
            ses.load()
    res = getattr(ses, "results", None)
    if res is None or res.empty:
        raise ValueError(f"{sess_name} results empty for {gp} {year}")
    return res





def list_gp_events(year: int) -> List[str]:
    return _event_schedule(year)["gp"].tolist()

def list_before_target(year: int, target_gp : str) -> List[str]:
    sch= _event_schedule(year)
    if target_gp not in sch["gp"].values:
        raise ValueError(f"Target gp{target_gp} not found in {year} schedule")
    tgt_date= sch.loc[sch['gp']== target_gp, "date"].iloc[0]
    return sch.loc[sch["date"] < tgt_date, "gp"].tolist()

def _pick(cols:pd.Index, *candidates: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of the candidates found in columns: {list(cols)}")

def _event_date(year:int, gp_name: str):
    sch= _event_schedule(year)
    row = sch.loc[sch["gp"]== gp_name]
    if row.empty:
        return None
    return row["date"].iloc[0]

def extract_event_qr(year: int, gp_name: str) -> pd.DataFrame:
    """Return one row per driver with grid_pos and finish_pos.
       Prefer both from Race results; fall back to Quali for grid only if needed.
    """
    r_res = _load_results_only(year, gp_name, "R")
    if r_res is None or len(r_res) == 0:
        raise RuntimeError("race results empty")

    r_res = r_res.copy()
    r_res["DriverNumber"] = r_res["DriverNumber"].astype(str).str.strip()
    fin_col = "ClassifiedPosition" if "ClassifiedPosition" in r_res.columns else "Position"

    df = None  # <— ensure we don't reference it before assignment

    # 1) Use race grid if available
    if "GridPosition" in r_res.columns and r_res["GridPosition"].notna().any():
        need = ["DriverNumber", "Abbreviation", "TeamName", "GridPosition", fin_col]
        missing = [c for c in need if c not in r_res.columns]
        if not missing:
            df = r_res[need].rename(
                columns={
                    "Abbreviation": "driver",
                    "TeamName": "team",
                    "GridPosition": "grid_pos",
                    fin_col: "finish_pos",
                }
            )

    # 2) Otherwise: take grid from Quali and finish from Race
    if df is None:
        q_res = _load_results_only(year, gp_name, "Q")
        if q_res is None or len(q_res) == 0:
            raise RuntimeError("qualifying results empty and race has no grid")
        q_res = q_res.copy()
        q_res["DriverNumber"] = q_res["DriverNumber"].astype(str).str.strip()
        q_grid_col = "GridPosition" if "GridPosition" in q_res.columns else "Position"

        need_q = ["DriverNumber", "Abbreviation", "TeamName", q_grid_col]
        need_r = ["DriverNumber", fin_col]
        miss_q = [c for c in need_q if c not in q_res.columns]
        miss_r = [c for c in need_r if c not in r_res.columns]
        if miss_q or miss_r:
            raise KeyError(f"Missing columns for merge: Q missing {miss_q}, R missing {miss_r}")

        qi = q_res[need_q].rename(
            columns={q_grid_col: "grid_pos", "Abbreviation": "driver", "TeamName": "team"}
        )
        ri = r_res[need_r].rename(columns={fin_col: "finish_pos"})
        df = qi.merge(ri, on="DriverNumber", how="inner")

    # 3) Validate rows and types
    if df is None or df.empty:
        raise RuntimeError("no rows after assembling Q/R")

    df = df.copy()
    df["grid_pos"] = pd.to_numeric(df["grid_pos"], errors="coerce")
    df["finish_pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")
    df = df.dropna(subset=["grid_pos", "finish_pos"])
    if df.empty:
        raise RuntimeError("positions all NA after coercion")

    # 4) Safe assignments (no chained warnings)
    df = df.copy()
    df.loc[:, "year"] = year
    df.loc[:, "gp"] = gp_name
    df.loc[:, "date"] = _event_date(year, gp_name)
    df.loc[:, "DriverNumber"] = df["DriverNumber"].astype(str)

    return df[["year", "gp", "date", "driver", "team", "grid_pos", "finish_pos", "DriverNumber"]]




def build_training_min(years: List[int]) -> pd.DataFrame:
    out, errors = [], []
    for y in years:
        for gp in list_gp_events(y):
            try:
                df_ev = extract_event_qr(y, gp)
                if df_ev.empty or df_ev["grid_pos"].isna().all() or df_ev["finish_pos"].isna().all():
                    raise ValueError("empty/NaN results")
                out.append(df_ev)
            except Exception as e:
                errors.append((y, gp, str(e)))
    if not out:
        raise RuntimeError(f"No events Loaded. Sample Errors: {errors[:3]}")
    return pd.concat(out, ignore_index=True)

def build_training_until(target_year: int, target_gp: str, hist_years=range(2023, 2025)) -> pd.DataFrame:
    from time import perf_counter

    # Optional exclude support if present in module globals (from config.EXCLUDE_EVENTS)
    EXC = globals().get("EXCLUDE_EVENTS", {})
    def _not_excluded(year: int, gp: str) -> bool:
        return gp not in EXC.get(year, set())

    rows = []

    # ---- History years (e.g., 2023–2024) ----
    for y in hist_years:
        try:
            events_all = list_gp_events(y)
            events = [gp for gp in events_all if _not_excluded(y, gp)]
            print(f"[INFO] {y}: {len(events)} events to load (of {len(events_all)} total)")
        except Exception as e:
            print(f"[SKIP-YEAR] {y}: schedule error: {e}")
            continue

        for gp in events:
            try:
                t0 = perf_counter()
                df_ev = extract_event_qr(y, gp)
                if df_ev.empty or df_ev["grid_pos"].isna().all() or df_ev["finish_pos"].isna().all():
                    raise ValueError("empty/NaN results")
                rows.append(df_ev)
                print(f"[LOAD] {y} {gp} ({len(df_ev)} rows) in {perf_counter() - t0:.1f}s")
            except Exception as e:
                print(f"[SKIP] {y} {gp}: {e}")
                continue

    # ---- Current season up to target (no leakage; prefer races that actually have results) ----
    try:
        pre_events_raw_all = list_before_target(target_year, target_gp)
        pre_events_raw = [gp for gp in pre_events_raw_all if _not_excluded(target_year, gp)]
        # Keep only races that already have race results
        pre_events = [gp for gp in pre_events_raw if _race_has_results(target_year, gp)]
        if not pre_events and pre_events_raw:
            print("[WARN] No verified race results found; falling back to unverified pre-events list.")
            pre_events = pre_events_raw
        print(f"[INFO] {target_year} before '{target_gp}': {len(pre_events)} events (filtered from {len(pre_events_raw_all)})")
    except Exception as e:
        print(f"[SKIP-SEASON] {target_year}: schedule error: {e}")
        pre_events = []

    for gp in pre_events:
        try:
            t0 = perf_counter()
            df_ev = extract_event_qr(target_year, gp)
            if df_ev.empty or df_ev["grid_pos"].isna().all() or df_ev["finish_pos"].isna().all():
                raise ValueError("empty/NaN results")
            rows.append(df_ev)
            print(f"[LOAD] {target_year} {gp} ({len(df_ev)} rows) in {perf_counter() - t0:.1f}s")
        except Exception as e:
            print(f"[SKIP] {target_year} {gp}: {e}")
            continue

    if not rows:
        raise RuntimeError("No training data found before target.")

    full = pd.concat(rows, ignore_index=True)
    full = full.drop_duplicates(subset=["year", "gp", "DriverNumber"])
    if "date" in full.columns:
        full = full.sort_values(["date", "year", "gp", "DriverNumber"]).reset_index(drop=True)
    return full




def get_target_drivers(year: int, gp_name: str) -> pd.DataFrame:
    """
    Return driver/team/grid for the target event.
    Order of preference:
      1) Qualifying results (with grid)
      2) FP1 results (no grid)
      3) Fallback to the latest completed race before target in the same season (no grid)
    """
    # 1) Try Qualifying
    try:
        q_res = _load_results_only(year, gp_name, "Q")
        q_res = q_res.copy()
        q_res["DriverNumber"] = q_res["DriverNumber"].astype(str).str.strip()
        grid_col = "GridPosition" if "GridPosition" in q_res.columns else "Position"
        need = ["DriverNumber", "Abbreviation", "TeamName", grid_col]
        if all(c in q_res.columns for c in need):
            df = q_res[need].rename(
                columns={grid_col: "grid_pos", "Abbreviation": "driver", "TeamName": "team"}
            ).copy()
            df.loc[:, "year"] = year
            df.loc[:, "gp"] = gp_name
            df.loc[:, "date"] = _event_date(year, gp_name)
            return df[["year", "gp", "date", "driver", "team", "grid_pos", "DriverNumber"]]
    except Exception:
        pass  # fall through

    # 2) Try FP1 (entry list, no grid)
    try:
        fp_res = _load_results_only(year, gp_name, "FP1")
        fp_res = fp_res.copy()
        fp_res["DriverNumber"] = fp_res["DriverNumber"].astype(str).str.strip()
        need = ["DriverNumber", "Abbreviation", "TeamName"]
        if all(c in fp_res.columns for c in need):
            df = fp_res[need].rename(columns={"Abbreviation": "driver", "TeamName": "team"}).copy()
            df.loc[:, "grid_pos"] = pd.NA
            df.loc[:, "year"] = year
            df.loc[:, "gp"] = gp_name
            df.loc[:, "date"] = _event_date(year, gp_name)
            return df[["year", "gp", "date", "driver", "team", "grid_pos", "DriverNumber"]]
    except Exception:
        pass  # fall through

    # 3) Fallback: use the most recent completed race before target
    try:
        prior_events = list_before_target(year, gp_name)
    except Exception:
        prior_events = []

    ref = None
    for prev_gp in reversed(prior_events):
        try:
            r_res = _load_results_only(year, prev_gp, "R")
            if r_res is None or len(r_res) == 0:
                continue
            r_res = r_res.copy()
            r_res["DriverNumber"] = r_res["DriverNumber"].astype(str).str.strip()
            need = ["DriverNumber", "Abbreviation", "TeamName"]
            if all(c in r_res.columns for c in need):
                ref = r_res[need].rename(
                    columns={"Abbreviation": "driver", "TeamName": "team"}
                ).copy()
                break
        except Exception:
            continue

    if ref is None or ref.empty:
        raise RuntimeError(
            f"No entry list available for {gp_name} {year}: Q/FP1 empty and no prior race with results."
        )

    ref.loc[:, "grid_pos"] = pd.NA
    ref.loc[:, "year"] = year
    ref.loc[:, "gp"] = gp_name
    ref.loc[:, "date"] = _event_date(year, gp_name)
    return ref[["year", "gp", "date", "driver", "team", "grid_pos", "DriverNumber"]]



