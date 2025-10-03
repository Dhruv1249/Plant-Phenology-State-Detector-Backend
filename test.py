import joblib
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# --- 1. Load All Assets ---
print("--- 1. Loading all models and data files... ---")
try:
    # ... (All model loading code remains the same) ...
    combined_model = joblib.load('models/combined_model.joblib')
    combined_encoder = joblib.load('models/combined_encoder.joblib')
    temp_model = xgb.XGBRegressor()
    temp_model.load_model("models/temp_model.json")
    precip_model = xgb.XGBRegressor()
    precip_model.load_model("models/precip_model.json")
    rad_model = xgb.XGBRegressor()
    rad_model.load_model("models/rad_model.json")
    with open('models/scenario_to_plants_lookup.json') as f:
        scenario_to_plants = json.load(f)
    with open('models/name_lookups.json') as f:
        name_lookups = json.load(f)
    print("✅ Assets loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load assets: {e}")
    exit()

# --- 2. Define Sample Inputs ---
print("--- 2. Defining sample inputs... ---")
lat, lng = 32.10, 76.27
month, year = 10, 2025
biome_code = "Cwa"
day_of_year = datetime(year, month, 15).timetuple().tm_yday
print(f"📍 Location: Kangra ({lat}, {lng})")
print(f"📅 Date: {month}/{year} (Day of Year: {day_of_year})")
print(f"🌍 Biome: {biome_code}\n")


# --- 3. Run Climate Predictions ---
print("--- 3. Running climate predictions... ---")
temp_k = float(temp_model.predict(pd.DataFrame([[lat, lng, month, year]], columns=['lat', 'lon', 'month', 'year']))[0])
precip_raw = float(precip_model.predict(pd.DataFrame([[lat, lng, month, year]], columns=['lat', 'lon', 'month', 'year']))[0])
rad = float(rad_model.predict(pd.DataFrame([[lat, lng, month, year]], columns=['lat', 'lon', 'month', 'year']))[0])
print(f"🌡️ Predicted Temperature: {temp_k:.2f} K")
print(f"💧 Predicted Precipitation: {precip_raw:.8f} kg m-2 s-1")
print(f"☀️ Predicted Radiation: {rad:.2f} W m-2\n")


# --- 4. Perform Feature Engineering ---
print("--- 4. Performing feature engineering... ---")
MODEL_FEATURE_ORDER = ['latitude', 'longitude', 'T2M', 'PRECTOTCORR', 'SWGDN', 'biome_cat_Af', 'biome_cat_Am', 'biome_cat_Aw', 'biome_cat_BSh', 'biome_cat_BSk', 'biome_cat_BWh', 'biome_cat_BWk', 'biome_cat_Cfa', 'biome_cat_Cfb', 'biome_cat_Cfc', 'biome_cat_Csa', 'biome_cat_Csb', 'biome_cat_Cwa', 'biome_cat_Cwb', 'biome_cat_Dfa', 'biome_cat_Dfc', 'biome_cat_Dfd', 'biome_cat_Dsa', 'biome_cat_Dsb', 'biome_cat_Dsc', 'biome_cat_Dsd', 'biome_cat_Dwa', 'biome_cat_Dwc', 'biome_cat_Dwd', 'biome_cat_ET', 'biome_cat_Unknown', 'doy_sin', 'doy_cos', 'temp_x_radiation', 'drought_index']
ALL_BIOME_CODES = [code for col in MODEL_FEATURE_ORDER if (code := col.replace('biome_cat_', '')) != col]

doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)
temp_x_radiation = temp_k * rad
drought_index = temp_k / (precip_raw + 1) # Using the corrected formula

data = {'latitude': lat, 'longitude': lng, 'T2M': temp_k, 'PRECTOTCORR': precip_raw, 'SWGDN': rad, 'doy_sin': doy_sin, 'doy_cos': doy_cos, 'temp_x_radiation': temp_x_radiation, 'drought_index': drought_index}
for code in ALL_BIOME_CODES:
    data[f'biome_cat_{code}'] = 1 if code == biome_code else 0

input_df = pd.DataFrame([data])[MODEL_FEATURE_ORDER]
print("✅ Feature engineering complete.\n")

# --- 5. Run Ecology Prediction & Decode Result ---
print("--- 5. Running ecology model... ---")
encoded_result = combined_model.predict(input_df)[0]
scenario_text = combined_encoder.inverse_transform([encoded_result])[0]
print(f"🧠 Model's Raw Encoded Output: {encoded_result}")
print(f"📜 Decoded Scenario Text: '{scenario_text}'\n")


# --- 6. Look Up Results with CORRECTED NESTED LOGIC ---
print("--- 6. Looking up species and pests... ---")
plant_scientific_names = scenario_to_plants.get(scenario_text, [])
try:
    phenophase, pest_scientific_name = [part.strip() for part in scenario_text.split('|')]
except ValueError:
    phenophase, pest_scientific_name = "Unknown", "Unknown"

species_list = []
# Safely access the nested 'plant_names' dictionary
plant_name_dict = name_lookups.get('plant_names', {})
for plant_name in plant_scientific_names:
    species_list.append({
        'scientific_name': plant_name,
        # Look up the stripped name in the nested dictionary
        'common_name': plant_name_dict.get(plant_name.strip(), "Unknown"),
        'phenophase': phenophase
    })

# Safely access the nested 'pest_names' dictionary
pest_name_dict = name_lookups.get('pest_names', {})
pests_list = [{
    'scientific_name_pest': pest_scientific_name,
    # Look up the stripped name in the nested dictionary
    'common_name_pest': pest_name_dict.get(pest_scientific_name.strip(), "Unknown")
}]

print("✅ Lookups complete.\n")


# --- 7. FINAL RESULT ---
print("--- 7. FINAL RESULT ---")
print("Species:", json.dumps(species_list, indent=2))
print("Pests:", json.dumps(pests_list, indent=2))