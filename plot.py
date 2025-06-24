import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

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

x_stage = "route"
y_stage = "chipfinish"
print(f"X stage: {x_stage}, Y stage: {y_stage}")

# filter out designs that are not in both stages
x_designs = set(data[data[stage_key] == x_stage][design_key].unique())
y_designs = set(data[data[stage_key] == y_stage][design_key].unique())
common_designs = x_designs.intersection(y_designs)
# Remove designs that are not in both stages
data = data[data[design_key].isin(common_designs)]

# Get X data, Y data
x_data = data[data[stage_key] == x_stage][power_key].values
y_data = data[data[stage_key] == y_stage][power_key].values

print("X shape", x_data.shape)
print("Y shape", y_data.shape)

print("R^2 score:", r2_score(x_data, y_data))

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.5)
plt.title(f"Power: {x_stage} vs {y_stage}")
plt.xlabel(f"{x_stage}")
plt.ylabel(f"{y_stage}")

plt.show()