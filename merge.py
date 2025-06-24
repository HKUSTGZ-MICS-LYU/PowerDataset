import json
import pandas as pd

# Load the JSON files and the CSV file
with open('./vexiiriscv_1024_preprocessing.json', 'r') as f:
    vexiiriscv_data = json.load(f)

with open('./vexiiriscv_parameters.json', 'r') as f:
    parameters_data = json.load(f)

power_df = pd.read_csv('./power_preprocessing.csv')

# Prepare the data from the JSON files
processed_data = []
param_names = list(parameters_data.keys())

for key, value in vexiiriscv_data.items():
    design_name = value['Design Name']
    embedding = value['Embedding']
    
    design_parameters = {'Design Name': design_name}
    for i, param_name in enumerate(param_names):
        param_value_index = embedding[i]
        param_value = parameters_data[param_name][param_value_index]
        design_parameters[param_name] = param_value
    
    processed_data.append(design_parameters)

# Create a DataFrame from the processed data
vexiiriscv_df = pd.DataFrame(processed_data)

# Merge the new DataFrame with the power_preprocessing.csv DataFrame
merged_df = pd.merge(power_df, vexiiriscv_df, on='Design Name')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('./power_with_vex_parameters.csv', index=False)

print("Successfully merged the data and saved it to 'power_with_vex_parameters.csv'")
print(merged_df.head())