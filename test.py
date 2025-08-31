import fastf1
import pandas as pd

# Hardcoded qualifying grid positions for Dutch Grand Prix 2025
qualifying_positions = {
    'PIA': 1,
    'NOR': 2,
    'VER': 3,
    'LEC': 4,
    'HAM': 5,
    'RUS': 6,
    'ALB': 7,
    'LAW': 8,
    'HUL': 9,
    'BOR': 10,
    'STR': 11,
    'GAS': 12,
    'RIC': 13,
    'TSU': 14,
    'ZHO': 15,
    'DEV': 16,
    'MAG': 17,
    'MSC': 18,
    'SCH': 19,
    'LAT': 20
}

# Fetching the session for Dutch Grand Prix 2025 Qualifying ('Q')
year = 2025
gp_name = 'Dutch Grand Prix'

try:
    session = fastf1.get_session(year, gp_name, 'Q')
    session.load()  # Load the data for the qualifying session

    # Check if the session has laps data
    if session.laps.empty:
        raise ValueError(f"No qualifying data available for {gp_name} {year}.")

    # Create a DataFrame with the hardcoded grid positions for each driver
    qualifying_data = []

    for driver in session.laps['Driver'].unique():
        grid_position = qualifying_positions.get(driver, None)
        if grid_position is not None:
            qualifying_data.append({
                'Driver': driver,
                'Grid Position': grid_position
            })

    # Convert to DataFrame for easier inspection
    qualifying_df = pd.DataFrame(qualifying_data)
    print(qualifying_df)

except Exception as e:
    print(f"[ERROR] Failed to load qualifying data for {gp_name} {year}. Error: {e}")
