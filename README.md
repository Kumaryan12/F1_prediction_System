F1 Race Predictor (RF + Uncertainty)

Predict the finishing order of Formula 1 races with a leakage-safe feature pipeline, a Random Forest regressor, and built-in uncertainty.
The project pulls historical results up to a target Grand Prix, engineers driver/team “form” and circuit context features, and produces point predictions plus confidence bands. It gracefully handles missing current-weekend data (e.g., qualifying not published yet).

<p align="center"> <img src="docs/img/hero.png" alt="F1 Race Predictor overview" width="820"> <br> <em>End-to-end: data ➜ features ➜ model ➜ predictions (with uncertainty)</em> </p>
Table of Contents

Highlights

Architecture

Project Structure

Installation

Quickstart

Train & Predict (single shot)

Save a Trained Model

Predict Using a Saved Model

Pre-Qualifying Mode

Command Line Flags

How It Works

Data & Target Drivers

Feature Engineering (leakage-safe)

Model & Uncertainty

Evaluation (OOB)

Configuration

Saved Model Artifacts

Troubleshooting

Roadmap

Contributing

License

Acknowledgements

One-liners

Highlights

End-to-end pipeline: history → driver/team form (leakage-safe) → circuit context → model.

Uncertainty built-in: per-driver prediction σ, 68%/95% bands, and Monte-Carlo rank probabilities (Top-10, Podium, ±1 rank).

Pre-Quali fallback: if Q/FP1 are missing, a quali proxy from recent races is used.

Model persistence: save/load trained pipelines with metadata (feature list, train dates, OOB metrics).

CLI workflow: one command to train, evaluate, and predict.

Example Output

Predicted Top 10:
driver            team  grid_pos  pred_finish  pred_rank  pred_std  pi68_low  pi68_high  p_top10  p_podium  p_rank_pm1
   PIA         McLaren     2.33         1.71          1     1.28      1.00       2.99     1.000     0.912       0.660
   NOR         McLaren     1.67         2.81          2     3.07      1.00       5.88     0.996     0.624       0.624
   VER Red Bull Racing     2.33         5.21          3     4.39      1.00       9.60     0.864     0.354       0.332
   ...


Field notes

pred_finish: expected finishing position (lower = better)

pred_std: uncertainty from tree dispersion

pi68_*, pi95_*: 68% / 95% predictive intervals

p_top10, p_podium, p_rank_pm1: MC rank probabilities

Architecture
flowchart LR
  A[Historic results (FastF1/Ergast)] --> B[build_training_until()]
  B --> C[add_driver_team_form()]
  C --> D[add_circuit_context_df()]
  D --> E[train_model(RandomForest + OHE + Imputer)]
  E --> F[oob_errors()]
  E --> G[predict_event_with_uncertainty()]
  G --> H[MC ranks & predictive intervals]
  H --> I[predicted_order.csv]


Runtime path (train vs predict)

sequenceDiagram
  participant User
  participant CLI as python -m F1_prediction_system.main
  participant Data as data.py
  participant Feat as features.py
  participant Model as model.py

  User->>CLI: --year 2025 --gp "Dutch Grand Prix"
  CLI->>Data: build_training_until()
  Data-->>CLI: train_df
  CLI->>Feat: add_driver_team_form(), add_circuit_context_df()
  Feat-->>CLI: train_df+
  CLI->>Model: train_model(train_df+)
  Model-->>CLI: pipeline (prep + RF)
  CLI->>Data: get_target_drivers()
  CLI->>Feat: merge_latest_forms(), add_quali_proxy()
  Feat-->>CLI: pred_df
  CLI->>Model: predict_event_with_uncertainty(model, pred_df)
  Model-->>CLI: predictions + intervals + MC probs
  CLI-->>User: Top-10 table & saved CSV

Project Structure
F1_prediction_system/
  ├─ __init__.py
  ├─ main.py                 # CLI entry point
  ├─ data.py                 # data loading & target driver selection
  ├─ features.py             # feature engineering (forms, circuit context, quali proxy)
  ├─ model.py                # train RF, OOB errors, uncertainty, save/load helpers
  ├─ config.py               # HIST_YEARS, CIRCUIT_VOL, defaults
models/
  └─ rf_latest.joblib        # (optional) saved model artifact
predicted_order.csv          # output with full predictions (created by runs)
LICENSE
README.md
docs/
  ├─ img/                    # screenshots/diagrams for README
  └─ diagrams/               # Mermaid / draw.io sources

Installation

Python: 3.10+ recommended (tested on 3.13 as well)

Dependencies: pandas, numpy, scikit-learn>=1.1, joblib, fastf1

python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install pandas numpy scikit-learn joblib fastf1


On corporate networks, ensure FastF1 can fetch data or pre-warm its cache.

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


The artifact stores the sklearn pipeline and metadata (feature list, training date range, OOB metrics, etc.).

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
Flag	Type	Default	Description
--year	int	2025	Target season
--gp	str	"Dutch Grand Prix"	Target GP name
--preq	flag	off	Force pre-qualifying behavior (ignore Q; use quali proxy)
--proxy_window	int	3	Rolling window for quali proxy
--mc	int	500	Monte-Carlo samples for rank probabilities (0 = off)
--interval	{68,95}	68	Which interval to show in console
--load_model	path	–	Load a saved model (artifact or plain Pipeline)
--save_model	path	–	Save the trained model artifact
--auto_retrain	flag	off	Retrain if loaded model is stale or features changed
--force_load	flag	off	Use a loaded model even if features differ (not advised)
--weather_csv	path	–	Reserved (rain/temp merge)
--use_conformal	flag	off	Reserved (split-conformal intervals)
--alpha	float	0.20	Reserved conformal alpha
How It Works
Data & Target Drivers

Uses all events before the target GP to build the training set.

For the prediction event, tries Qualifying results (driver list & grid).

If missing, falls back to FP1. If still missing, uses the quali proxy.

Feature Engineering (leakage-safe)

Driver form (3-race): trailing mean of finishes, shifted by 1 (no peeking).

Team form (3-race): team average per race → trailing mean, shifted by 1.

Circuit context: prior sc_prob, vsc_prob, pit_loss from config.py.

Quali proxy: driver’s trailing mean grid over the last N races to fill unknown grid_pos.

Model & Uncertainty

Model: RandomForestRegressor with a preprocessing pipeline:

numeric → median impute

categoricals (team, driver) → one-hot encode

OOB (oob_score=True): fast, leak-resistant diagnostics.

Uncertainty:

Per-driver std from per-tree prediction dispersion.

Optional Monte Carlo sampling → p_top10, p_podium, p_rank_pm1.

68% / 95% intervals as simple normal bands around the mean (informative, not calibrated).

Evaluation (OOB)

Prints OOB R² / MAE / RMSE from RF’s out-of-bag predictions.

For rigorous evaluation, add chronological backtests (see Roadmap).

Configuration

config.py contains:

HIST_YEARS: list of past seasons to include (e.g., [2023, 2024, 2025])

CIRCUIT_VOL: GP → (sc_prob, vsc_prob, pit_loss_seconds)

Defaults for unknown circuits: DEFAULT_SC, DEFAULT_VSC, DEFAULT_PIT_LOSS
Extend with weather/seasonality or other priors as the model evolves.

Saved Model Artifacts

Artifacts saved via --save_model are joblib files containing:

import joblib
artifact = joblib.load("models/rf_latest.joblib")
model = artifact["model"]   # sklearn Pipeline (prep + RF)
meta  = artifact["meta"]    # dict: feat_list, train dates, oob, etc.

Troubleshooting

“No result data … on Ergast” / empty Q/FP1
Normal for very recent sessions. Use --preq to force the quali proxy, or just rely on the built-in fallback.

UserWarning: “Skipping features without any observed values … for imputation”
Harmless. A feature had no numeric observations in the training slice; the imputer skips it.

FutureWarning about .fillna downcasting
Harmless with current versions. The code coerces dtypes in relevant paths.

Model not saving
Make sure you passed --save_model with a valid path; folders are created automatically.
Example: --save_model models/rf_latest.joblib

Using saved model but still retraining
You used --auto_retrain and newer data or feature mismatches were detected.
Add --force_load to use the old model anyway (not recommended).

Roadmap

Backtesting: chronological splits + ranking metrics (Spearman, NDCG@10, Top-k hit rate).

More features: sprint-weekend flag; tyre/pit priors; DRS effectiveness; upgrades; weather merge.

Modeling: try Gradient Boosting (LightGBM/XGBoost) and ranking objectives.

Calibration: split-conformal or quantile forests for coverage-aware intervals.

Experiment tracking: MLflow or simple per-run CSV logs.

Tests: pytest for leakage checks, feature stability, save/load round-trip.

Contributing

Issues and PRs are welcome! Please:

Keep features leakage-safe

Add unit tests for new feature transforms

Document new CLI flags here in the README

License

Released under the MIT License (see LICENSE).

Acknowledgements

Data access via FastF1 (which uses Ergast and timing sources).

Inspiration from standard motorsport analytics workflows and the F1 analytics community.