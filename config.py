# F1_prediction_system/config.py
from pathlib import Path
from typing import Dict, Tuple  # If you're on 3.9+, you can use dict/tuple built-ins instead.

# Where FastF1 stores/cache data (inside this package folder)
CACHE_DIR: Path = Path(__file__).resolve().parent / "f1cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)  # ensure it exists

# Historical seasons to use for training (2021–2024)
HIST_YEARS = list(range(2023, 2025))

# Defaults used when a circuit isn't in CIRCUIT_VOL
DEFAULT_SC = 0.5
DEFAULT_VSC = 0.5
DEFAULT_PIT_LOSS = 21.0

# Circuit parameters: (SC probability, VSC probability, pit loss seconds)
CIRCUIT_VOL: Dict[str, Tuple[float, float, float]] = {
    "Bahrain Grand Prix": (0.63, 0.50, 22.9),
    "Saudi Arabian Grand Prix": (1.00, 0.50, 19.2),
    "Australian Grand Prix": (0.67, 0.50, 20.0),
    "Japanese Grand Prix": (0.67, 0.50, 22.2),
    "Dutch Grand Prix": (0.60, 0.60, 21.5),
}

FALLBACK_EVENTS: Dict[int, list[str]] = {
2025: [
"Bahrain Grand Prix",
"Saudi Arabian Grand Prix",
"Australian Grand Prix",
"Japanese Grand Prix",
"Chinese Grand Prix",
"Miami Grand Prix",
"Emilia Romagna Grand Prix",
"Monaco Grand Prix",
"Canadian Grand Prix",
"Spanish Grand Prix",
"Austrian Grand Prix",
"British Grand Prix",
"Hungarian Grand Prix",
"Belgian Grand Prix",
"Dutch Grand Prix",
],
}

# Races to exclude from training by year
EXCLUDE_EVENTS = {
    2025: {"Hungarian Grand Prix"},   # add others here if needed
}
