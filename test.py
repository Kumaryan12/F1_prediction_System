import fastf1
import pickle

# Example: load driver info
with open("f1cache\2023\2023-03-05_Bahrain_Grand_Prix\2023-03-04_Qualifying\_extended_timing_data.ff1pkl", "rb") as f:
    driver_info = pickle.load(f)

print(driver_info.head())
