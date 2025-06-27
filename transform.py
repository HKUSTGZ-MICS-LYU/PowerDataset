# To run this script on a GPU, you need a machine with an NVIDIA GPU, CUDA,
# and PyTorch installed.
# To run on CPU, only PyTorch is required.
# Example installation:
# pip install torch pandas scikit-learn joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Manual Control & Device Setup ---
# Set this to True to bypass GPU checks and force CPU execution.
FORCE_CPU = False

if not FORCE_CPU and torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA for training.")
else:
    device = torch.device("cpu")
    if FORCE_CPU:
        print("FORCE_CPU is set to True. Using CPU for training.")
    else:
        print("GPU not available. Using CPU for training.")


# --- Transformer Model Definition ---
class TransformerPredictor(nn.Module):
    """
    A Transformer-based model for tabular regression.
    """
    def __init__(self, num_features, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        # Input embedding layer to project features to d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Expects input shape (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output layer for regression
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, src):
        # src shape: (batch_size, num_features)
        
        # 1. Embed input features
        embedded_src = self.input_embedding(src)  # -> (batch_size, d_model)
        
        # 2. Add a sequence dimension for the transformer
        # The transformer expects a sequence, but we have a single set of features.
        # We treat it as a sequence of length 1.
        transformer_input = embedded_src.unsqueeze(1)  # -> (batch_size, 1, d_model)
        
        # 3. Pass through transformer encoder
        transformer_output = self.transformer_encoder(transformer_input) # -> (batch_size, 1, d_model)
        
        # 4. Remove the sequence dimension
        transformer_output = transformer_output.squeeze(1) # -> (batch_size, d_model)
        
        # 5. Pass through the final regression layer
        output = self.output_layer(transformer_output) # -> (batch_size, 1)
        
        return output

# --- Training and Evaluation Function ---
def run_transformer_pipeline(X_train, y_train, X_test, y_test, num_features, model_path, epochs=500, batch_size=32, lr=0.0003):
    """
    Handles the entire process of scaling, training, and evaluating the Transformer model.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = TransformerPredictor(num_features=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Evaluation loop
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mape = mean_absolute_percentage_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Print results and save model
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"R^2 Score: {r2:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, model_path)
    print(f"Model and scaler saved to '{model_path}'\n")


# --- Main Script Logic ---
# Load the dataset
df = pd.read_csv('power_with_vex_parameters.csv')

# --- Step 1: Filter for designs with complete backend phases ---
print("--- Filtering Data ---")
phase_counts = df.groupby('Design Name')['Backend Phase'].nunique()
complete_phases_count = phase_counts.max()
complete_designs = phase_counts[phase_counts == complete_phases_count].index
df_filtered = df[df['Design Name'].isin(complete_designs)].copy()
print(f"Filtered data to {len(complete_designs)} complete designs.\n")

# --- Step 2: Prepare data for modeling ---
for col in df_filtered.columns:
    if df_filtered[col].dtype == 'bool':
        df_filtered[col] = df_filtered[col].astype(int)

features_base = list(df.columns[df.columns.get_loc('xlen'):])
target_col = 'Power3'

# --- Scenario 1: Predicting without floorplan power ---
print("--- Scenario 1 (Filtered Data): Predicting without floorplan power ---")
df_chipfinish = df_filtered[df_filtered['Backend Phase'] == 'chipfinish'].copy()
X1 = df_chipfinish[features_base].values
y1 = df_chipfinish[target_col].values * 1e-6
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
run_transformer_pipeline(X1_train, y1_train, X1_test, y1_test, X1.shape[1], 'transformer_model_no_floorplan.pth')

# --- Scenario 2: Predicting with floorplan power ---
print("--- Scenario 2 (Filtered Data): Predicting with floorplan power as a feature ---")
df_fp_pivot = df_filtered[df_filtered['Backend Phase'].isin(['chipfinish', 'floorplan'])].pivot_table(
    index='Design Name', columns='Backend Phase', values='Power3'
).reset_index()
df_params = df_filtered.drop_duplicates(subset='Design Name')[['Design Name'] + features_base]
df_merged = pd.merge(df_fp_pivot, df_params, on='Design Name').dropna()

features_with_floorplan = ['floorplan'] + features_base
target_chipfinish = 'chipfinish'
X2 = df_merged[features_with_floorplan].values
y2 = df_merged[target_chipfinish].values * 1e-6
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
run_transformer_pipeline(X2_train, y2_train, X2_test, y2_test, X2.shape[1], 'transformer_model_with_floorplan.pth')
