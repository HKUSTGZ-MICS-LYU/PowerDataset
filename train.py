import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

TEST_SIZE = 0.2


data_header = ["Design Name","Backend Phase",
               "Corner1","Power1",
               "Corner2","Power2",
               "Corner3","Power3"]

dataset = "power_with_vex_parameters.csv"
data = pd.read_csv(dataset)

power_key = "Power3"
stage_key = "Backend Phase"
design_key = "Design Name"

# Get all Power3 values
stages = data[stage_key].unique()
designs = data[design_key].unique()

print(f"Stages: {stages}")
print(f"# of Designs: {len(designs)}")

x_stage = "floorplan"
y_stage = "chipfinish"
print(f"X stage: {x_stage}, Y stage: {y_stage}")

# filter out designs that are not in both stages
x_designs = set(data[data[stage_key] == x_stage][design_key].unique())
y_designs = set(data[data[stage_key] == y_stage][design_key].unique())
common_designs = x_designs.intersection(y_designs)
# Remove designs that are not in both stages
data = data[data[design_key].isin(common_designs)]

# Get Design Parameters, use all data after Power3
x_params = data[data[stage_key] == x_stage].drop(columns=data_header).values

# Get X data, Y data
x_data = data[data[stage_key] == x_stage][power_key].values
y_data = data[data[stage_key] == y_stage][power_key].values

print("X shape", x_data.shape)
print("Y shape", y_data.shape)

# Experiment 1: use x_params to predict y_data
x_train, x_test, y_train, y_test = train_test_split(
    x_params, y_data, test_size=TEST_SIZE, random_state=42)

print("Train Test Split:", TEST_SIZE)

print("Train with Parameters Only")
model = XGBRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("R^2 score:", r2_score(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))

print(f"Train with Parameters and {x_stage} Power")

x_params_with_power = np.hstack((x_params, x_data.reshape(-1, 1)))

x_train, x_test, y_train, y_test = train_test_split(
    x_params_with_power, y_data, test_size=TEST_SIZE, random_state=42)


model = XGBRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("R^2 score:", r2_score(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))

# Direct Relation
print("Direct Relation")
print("R^2 score:", r2_score(x_data, y_data))
print("MAPE:", mean_absolute_percentage_error(x_data, y_data))