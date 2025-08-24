F1 Race Predictor (RF + Uncertainty)

Predict Formula 1 race finishing order using a leakage-safe feature pipeline, a Random Forest regressor, and uncertainty estimates. The project pulls historical results up to a target Grand Prix, engineers “form” and circuit context features, and produces both point predictions and confidence bands. It’s resilient to missing current-weekend data (e.g., when qualifying hasn’t posted yet).

Highlights

End-to-end pipeline: history ➜ driver/team form (leakage-safe) ➜ circuit context ➜ model.

Uncertainty built-in: per-driver prediction std, 68%/95% bands, and Monte-Carlo rank probabilities (Top-10, Podium, ±1 rank).

Pre-Quali fallback: if Q/FP1 data is missing for the current weekend, a quali proxy from recent races is used.

Model persistence: save/load trained pipelines with metadata (feature list, train dates, OOB metrics).

CLI workflow: one command to train, evaluate, and predict.

Example Output
Predicted Top 10:
driver            team  grid_pos  pred_finish  pred_rank  pred_std  pi68_low  pi68_high  p_top10  p_podium  p_rank_pm1
   PIA         McLaren  2.33          1.71           1     1.28      1.00       2.99      1.000     0.912       0.660
   NOR         McLaren  1.67          2.81           2     3.07      1.00       5.88      0.996     0.624       0.624
   VER Red Bull Racing  2.33          5.21           3     4.39      1.00       9.60      0.864     0.354       0.332
   ...


pred_finish: expected finishing position (lower = better)

pred_std: uncertainty from tree dispersion

pi68_*, pi95_*: 68% / 95% predictive intervals

p_top10, p_podium, p_rank_pm1: MC rank probabilities

Project Structure
F1_prediction_system/
  ├─ __init__.py
  ├─ main.py                 # CLI entry point
  ├─ data.py                 # data loading & target driver selection
  ├─ features.py             # feature engineering (forms, circuit context, quali proxy)
  ├─ model.py                # model training, OOB errors, uncertainty, save/load helpers
  ├─ config.py               # constants: HIST_YEARS, CIRCUIT_VOL, defaults
models/
  └─ rf_latest.joblib        # (optional) saved model artifact (created by you)
predicted_order.csv          # output with full predictions (created by runs)
LICENSE
README.md

Installation

Python: 3.10+ recommended (project known to run with 3.13 too).
Dependencies (core):

pandas, numpy, scikit-learn>=1.1, joblib

FastF1 (uses Ergast/Timing backing APIs; you’ll see FastF1-style logs)

Install:

python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install pandas numpy scikit-learn joblib fastf1


If you use a corporate network, allow FastF1 to fetch data (or pre-warm its cache).

Quickstart
1) Train & Predict (single shot)
python -m F1_prediction_system.main --year 2025 --gp "Dutch Grand Prix"


This will:

Build training data up to the target GP

Train a RandomForest model and print OOB metrics (R²/MAE/RMSE)

Build the prediction frame for the target GP

Print the Top-10 with uncertainty & probabilities

Save predicted_order.csv to your current working directory

2) Save a Trained Model
python -m F1_prediction_system.main \
  --year 2025 --gp "Dutch Grand Prix" \
  --save_model models/rf_latest.joblib


The artifact contains the sklearn pipeline and metadata (feature list, training date range, OOB metrics, etc.).

3) Predict Using a Saved Model (no retrain)
python -m F1_prediction_system.main \
  --year 2025 --gp "Dutch Grand Prix" \
  --load_model models/rf_latest.joblib


Add --auto_retrain to retrain if newer data is detected or features changed.

Add --force_load to proceed even if features differ (not recommended).

4) Pre-Qualifying Mode (force quali proxy)

When qualifying is missing or you want to simulate pre-Q uncertainty:

python -m F1_prediction_system.main \
  --year 2025 --gp "Dutch Grand Prix" \
  --preq --proxy_window 3

Command Line Flags
--year INT                  Target season (e.g., 2025)
--gp STR                    Target GP name (e.g., "Dutch Grand Prix")

--preq                      Force pre-qualifying behavior (ignore Q; use quali proxy)
--proxy_window INT          Rolling window for quali proxy (default 3)

--mc INT                    Monte-Carlo samples for rank probabilities (default 500; 0 = off)
--interval {68,95}          Which interval to display in console (default 68)

--load_model PATH           Load a saved model (artifact or plain joblib Pipeline)
--save_model PATH           Save the trained model artifact
--auto_retrain              If the loaded model is stale or features changed, retrain
--force_load                Use the loaded model even if features differ (not recommended)

--weather_csv PATH          Reserved (CSV with rain/temp); not wired yet
--use_conformal             Reserved (split-conformal intervals); not wired yet
--alpha FLOAT               Reserved (conformal alpha, default 0.20)

How It Works
Data & Target Drivers

Uses the event list up to (but excluding) the target GP to build training rows.

For the prediction event, tries to load Qualifying results (driver list & grid).
If missing, falls back to FP1 (if any). If still missing, uses the quali proxy.

Feature Engineering (leakage-safe)

Driver form (3-race): rolling mean of past 3 finishes, shifted by 1 race ➜ no peeking.

Team form (3-race): compute team average per race, then rolling mean, shifted by 1.

Circuit context: per-GP priors like sc_prob, vsc_prob, pit_loss from config.py.

Quali proxy: driver’s rolling mean grid over the last N races to fill unknown grid_pos.

Model

RandomForestRegressor with:

preprocessing: median impute numeric, one-hot encode categoricals

oob_score=True for quick, leak-resistant diagnostics

Uncertainty:

Per-driver std from per-tree prediction dispersion.

Optional MC sampling to estimate p_top10, p_podium, p_rank±1.

68%/95% intervals as simple normal bands around the mean (informative, not calibrated).

Evaluation (OOB)

Prints OOB R² / MAE / RMSE from the RF’s out-of-bag predictions.

For rigorous evaluation, add chronological backtests (see Roadmap).

Configuration

config.py includes:

HIST_YEARS: list of past seasons to include (e.g., [2023, 2024, 2025])

CIRCUIT_VOL: mapping from GP name ➜ (sc_prob, vsc_prob, pit_loss_seconds)

Defaults for unknown circuits: DEFAULT_SC, DEFAULT_VSC, DEFAULT_PIT_LOSS

You can extend this with more priors (e.g., weather/seasonality) as you refine the model.

Saved Model Artifacts

Artifacts saved via --save_model are joblib files that include:

model: the sklearn Pipeline (preprocess + RF)

meta: JSON-serializable dict with

feat_list, train_rows, train_start_date, train_end_date

hist_years & hist_years_n

oob metrics

model_desc, code_version, saved_at, and target context

Load with --load_model or:

import joblib
artifact = joblib.load("models/rf_latest.joblib")
model = artifact["model"]
meta = artifact["meta"]

Troubleshooting

“No result data … on Ergast” / empty Q/FP1
Normal for very recent sessions. Use --preq to force the quali proxy, or just rely on the built-in fallback.

UserWarning: “Skipping features without any observed values … for imputation”
Harmless. It means a feature had no numeric observations in the training slice; the imputer skips it.

FutureWarning about .fillna downcasting
Also harmless in current versions. We coerce dtypes before fills in recent code paths.

Model not saving
Ensure you passed --save_model with a valid path and that the folder exists or is creatable.
Example: --save_model models/rf_latest.joblib

Using saved model but still retraining
You used --auto_retrain and the code detected newer data or feature mismatches. Add --force_load to use the old model anyway (not recommended).

Roadmap

Backtesting: chronological splits + metrics better aligned with ranking (Spearman, NDCG@10, Top-k hit rate).

More features: sprint weekend flag; tyre/pit priors; DRS effectiveness; upgrades phase; weather merge.

Modeling: compare Gradient Boosting (LightGBM/XGBoost) and pairwise/ranking objectives.

Calibration: split-conformal or quantile forests for coverage-aware intervals.

Experiment tracking: MLflow or simple CSV logs per run.

Tests: basic pytest suite (no leakage, stable feature sets, save/load roundtrip).

Contributing

Issues and PRs are welcome! Please:

Keep features leakage-safe

Add unit tests for new feature transforms

Document new CLI flags in this README

License

This project is released under the MIT License (see LICENSE).

Acknowledgements

Data access via FastF1 (which uses Ergast and timing sources).

Inspiration from standard motorsport analytics workflows and public community tools.

One-liners You’ll Use Often

Train, evaluate, predict, and save:

python -m F1_prediction_system.main \
  --year 2025 --gp "Dutch Grand Prix" \
  --save_model models/rf_latest.joblib


Predict using the saved model (no retrain):

python -m F1_prediction_system.main \
  --year 2025 --gp "Dutch Grand Prix" \
  --load_model models/rf_latest.joblib


Pre-Quali simulation with more MC:

python -m F1_prediction_system.main \
  --year 2025 --gp "Dutch Grand Prix" \
  --preq --proxy_window 3 --mc 2000 --interval 95


You’re ready for race weekend 🚦🏁