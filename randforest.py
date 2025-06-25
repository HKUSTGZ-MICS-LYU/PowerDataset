# To run this script on a GPU, you need a machine with an NVIDIA GPU and the
# RAPIDS cuML library installed, matching your system's CUDA version.
# Example installation for CUDA 12:
# conda install -c rapidsai -c nvidia -c conda-forge cuml -c rapidsai-nightly python=3.9 cudatoolkit=12.0

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

# --- Manual Control ---
# Set this to True to bypass GPU checks and force CPU execution.
# This is a workaround for environment issues like CUDA version mismatches.
FORCE_CPU = True

# Attempt to import cuML only if CPU is not forced.
if not FORCE_CPU:
    try:
        from cuml.ensemble import RandomForestRegressor as GPU_RandomForestRegressor
        print("Successfully imported cuML. GPU will be used for training.")
        GPU_ENABLED = True
    except ImportError:
        print("Could not import cuML. Falling back to scikit-learn's CPU implementation.")
        print("To enable GPU training, please install RAPIDS cuML on a supported system.")
        from sklearn.ensemble import RandomForestRegressor as CPU_RandomForestRegressor
        GPU_ENABLED = False
else:
    print("FORCE_CPU is set to True. Bypassing GPU check and using CPU for training.")
    from sklearn.ensemble import RandomForestRegressor as CPU_RandomForestRegressor
    GPU_ENABLED = False


# Load the dataset
df = pd.read_csv('power_with_vex_parameters.csv')

# --- Step 1: Filter for designs with complete backend phases ---

print("\n--- Filtering Data ---")
# Calculate the number of backend phases for each design
phase_counts = df.groupby('Design Name')['Backend Phase'].nunique()

# A "complete" design is assumed to have the maximum number of phases found in the dataset
complete_phases_count = phase_counts.max()
print(f"A complete design has {complete_phases_count} backend phases.")

# Get the list of designs that have the complete set of phases
complete_designs = phase_counts[phase_counts == complete_phases_count].index
original_design_count = df['Design Name'].nunique()
complete_design_count = len(complete_designs)

print(f"Found {complete_design_count} complete designs out of {original_design_count} total designs.")
print(f"Filtering out {original_design_count - complete_design_count} incomplete designs.\n")

# Create a new dataframe containing only the complete designs
df_filtered = df[df['Design Name'].isin(complete_designs)].copy()


# --- Step 2: Run modeling pipelines on the filtered data ---

# Convert boolean columns to integers (0 or 1) for modeling
for col in df_filtered.columns:
    if df_filtered[col].dtype == 'bool':
        df_filtered[col] = df_filtered[col].astype(int)

# Define features (parameters) and target (Typical corner power)
features = list(df.columns[df.columns.get_loc('xlen'):])
target = 'Power3'

# --- Scenario 1 (Filtered Data): Predicting without floorplan power ---

print("--- Scenario 1 (Filtered Data): Predicting without floorplan power ---")
df_chipfinish = df_filtered[df_filtered['Backend Phase'] == 'chipfinish'].copy()

X1 = df_chipfinish[features]
y1 = df_chipfinish[target]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Initialize the model (GPU or CPU version based on availability)
if GPU_ENABLED:
    # cuML's RandomForestRegressor has a similar API to scikit-learn's
    model1_filtered = GPU_RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model1_filtered = CPU_RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

model1_filtered.fit(X1_train, y1_train)

# Make predictions and evaluate the model
y1_pred = model1_filtered.predict(X1_test)
mae1 = mean_absolute_error(y1_test, y1_pred)
mape1 = mean_absolute_percentage_error(y1_test, y1_pred)
r2_1 = r2_score(y1_test, y1_pred)

print(f"Model 1 (Filtered) - MAE: {mae1:.2f}")
print(f"Model 1 (Filtered) - MAPE: {mape1:.2%}")
print(f"Model 1 (Filtered) - R^2 Score: {r2_1:.4f}")

joblib.dump(model1_filtered, 'power_model_filtered_no_floorplan.pkl')
print("Model 1 (Filtered) saved to 'power_model_filtered_no_floorplan.pkl'\n")


# --- Scenario 2 (Filtered Data): Predicting with floorplan power ---

print("--- Scenario 2 (Filtered Data): Predicting with floorplan power as a feature ---")
df_fp = df_filtered[df_filtered['Backend Phase'].isin(['chipfinish', 'floorplan'])]
df_pivot = df_fp.pivot_table(index='Design Name', columns='Backend Phase', values='Power3').reset_index()

df_params = df_filtered.drop_duplicates(subset='Design Name')[['Design Name'] + features]
df_merged = pd.merge(df_pivot, df_params, on='Design Name')

# Since we filtered for complete designs, there should be no NaNs, but it's good practice to check/handle
df_merged.dropna(inplace=True)

features_with_floorplan = ['floorplan'] + features
target_chipfinish = 'chipfinish'

X2 = df_merged[features_with_floorplan]
y2 = df_merged[target_chipfinish]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Initialize the model (GPU or CPU version)
if GPU_ENABLED:
    model2_filtered = GPU_RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model2_filtered = CPU_RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

model2_filtered.fit(X2_train, y2_train)

# Make predictions and evaluate the model
y2_pred = model2_filtered.predict(X2_test)
mae2 = mean_absolute_error(y2_test, y2_pred)
mape2 = mean_absolute_percentage_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print(f"Model 2 (Filtered) - MAE: {mae2:.2f}")
print(f"Model 2 (Filtered) - MAPE: {mape2:.2%}")
print(f"Model 2 (Filtered) - R^2 Score: {r2_2:.4f}")

joblib.dump(model2_filtered, 'power_model_filtered_with_floorplan.pkl')
print("Model 2 (Filtered) saved to 'power_model_filtered_with_floorplan.pkl'")
