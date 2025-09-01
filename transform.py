# To run this script on a GPU, you need a machine with an NVIDIA GPU, CUDA,
# and PyTorch installed.
# To run on CPU, only PyTorch is required.
# Example installation:
# pip install torch pandas scikit-learn joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os

# --- Manual Control & Device Setup ---
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


# --- Enhanced Transformer Model for Integrated Power Prediction ---
class IntegratedPowerTransformer(nn.Module):
    """
    专门用于预测整合功耗的增强型Transformer模型
    相比单corner模型，增加了以下特性：
    1. Multi-target prediction head (可同时预测多种整合方式)
    2. Cross-corner attention mechanism
    3. Power-aware feature encoding
    """
    def __init__(self, num_features, d_model=160, nhead=8, num_encoder_layers=5, 
                 dim_feedforward=640, dropout=0.12, num_integration_targets=4):
        super(IntegratedPowerTransformer, self).__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.num_integration_targets = num_integration_targets
        
        # Enhanced input embedding for power-aware processing
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Power-specific feature encoding
        # 分组处理不同类型的硬件特征
        self.power_feature_groups = {
            'frequency_related': ['xlen', 'decoders', 'lanes', 'with-mul', 'with-div'],
            'cache_related': ['fetch-l1', 'lsu-l1', 'fetch-l1-sets', 'fetch-l1-ways', 
                            'lsu-l1-sets', 'lsu-l1-ways'],
            'control_related': ['with-gshare', 'with-btb', 'with-ras', 'btb-sets'],
            'pipeline_related': ['dispatcher-at', 'decoder-at', 'relaxed-branch', 'relaxed-shift']
        }
        
        # Group-specific encoders
        self.group_encoders = nn.ModuleDict({
            group: nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model)
            ) for group in self.power_feature_groups.keys()
        })
        
        # Positional encoding for feature importance
        self.positional_encoding = nn.Parameter(torch.randn(1, num_features, d_model))
        
        # Enhanced Transformer Encoder with more layers for complex power relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(d_model),
                nn.GELU()
            ) for k in [1, 3, 5]  # Different receptive fields
        ])
        
        self.fusion_combine = nn.Linear(d_model * 3, d_model)
        
        # Attention pooling instead of simple average pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead//2, dropout=dropout, batch_first=True
        )
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Multi-target prediction heads for different integration methods
        self.prediction_heads = nn.ModuleDict({
            'weighted_avg': self._create_prediction_head(d_model, dropout),
            'worst_case': self._create_prediction_head(d_model, dropout),
            'rms': self._create_prediction_head(d_model, dropout),
            'statistical': self._create_prediction_head(d_model, dropout)
        })
        
        # Feature importance weights (learnable)
        self.feature_weights = nn.Parameter(torch.ones(num_features))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_prediction_head(self, d_model, dropout):
        """创建预测头"""
        return nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(d_model // 4, 1)
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, src, target_type='weighted_avg'):
        batch_size = src.size(0)
        
        # Apply learnable feature weights
        weighted_src = src * self.feature_weights.unsqueeze(0)
        
        # Primary embedding
        embedded = self.input_embedding(weighted_src)  # (batch_size, d_model)
        
        # Create feature tokens for transformer processing
        feature_tokens = weighted_src.unsqueeze(-1).repeat(1, 1, self.d_model)
        feature_tokens = feature_tokens + self.positional_encoding
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(feature_tokens)
        
        # Multi-scale feature fusion
        # Transpose for conv1d: (batch, features, d_model) -> (batch, d_model, features)
        conv_input = transformer_output.transpose(1, 2)
        
        fusion_outputs = []
        for conv_layer in self.feature_fusion:
            fusion_outputs.append(conv_layer(conv_input))
        
        # Combine multi-scale features
        fused_features = torch.cat(fusion_outputs, dim=1)  # (batch, 3*d_model, features)
        fused_features = fused_features.transpose(1, 2)    # (batch, features, 3*d_model)
        fused_features = self.fusion_combine(fused_features)  # (batch, features, d_model)
        
        # Attention-based pooling
        query = self.pooling_query.repeat(batch_size, 1, 1)
        pooled_output, _ = self.attention_pooling(query, fused_features, fused_features)
        pooled_output = pooled_output.squeeze(1)  # (batch_size, d_model)
        
        # Combine with direct embedding (skip connection)
        combined_features = pooled_output + embedded
        
        # Multi-target prediction
        if target_type == 'all':
            # Return all integration types
            outputs = {}
            for target_name, head in self.prediction_heads.items():
                outputs[target_name] = head(combined_features)
            return outputs
        else:
            # Return specific integration type
            return self.prediction_heads[target_type](combined_features)


# --- Integration Target Calculation ---
def calculate_integration_targets(df, power_cols=['Power1', 'Power2', 'Power3']):
    """
    计算不同的整合目标
    """
    targets = {}
    
    # 1. 加权平均 (基于实际使用场景)
    weights = {'Power1': 0.2, 'Power2': 0.2, 'Power3': 0.6}  # Fast, Slow, Typical
    targets['weighted_avg'] = sum(weights[col] * df[col] for col in power_cols)
    
    # 2. 最坏情况 (设计验证用)
    targets['worst_case'] = df[power_cols].max(axis=1)
    
    # 3. RMS平均 (能量角度)
    targets['rms'] = np.sqrt(sum(df[col]**2 for col in power_cols) / len(power_cols))
    
    # 4. 统计方法 (mean + 2*std, 考虑变异性)
    power_mean = df[power_cols].mean(axis=1)
    power_std = df[power_cols].std(axis=1)
    targets['statistical'] = power_mean + 2 * power_std
    
    return targets


# --- Enhanced Training Function for Integrated Model ---
def train_integrated_transformer(X_train, y_targets_train, X_test, y_targets_test, 
                               num_features, model_path, target_type='weighted_avg',
                               epochs=250, batch_size=32, lr=0.0008, weight_decay=1e-4):
    """
    训练整合功耗预测模型
    """
    print(f"Training integrated model for target: {target_type}")
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log transform for power values
    y_train_log = np.log1p(y_targets_train[target_type])
    y_test_log = np.log1p(y_targets_test[target_type])
    
    print(f"Target power range: {y_targets_train[target_type].min():.6f} - {y_targets_train[target_type].max():.6f}")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32), 
        torch.tensor(y_train_log, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32), 
        torch.tensor(y_test_log, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize enhanced integrated model
    model = IntegratedPowerTransformer(num_features=num_features).to(device)
    
    # Use combination of losses for better training
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.HuberLoss(delta=1.0)
    
    def combined_loss(pred, target):
        return 0.7 * huber_criterion(pred, target) + 0.3 * mse_criterion(pred, target)
    
    # Advanced optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler with warm-up
    warmup_epochs = 20
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*2, total_steps=epochs*len(train_loader),
        pct_start=warmup_epochs/epochs, anneal_strategy='cos'
    )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 30
    
    print(f"Starting integrated training with {len(train_loader)} batches per epoch...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, target_type=target_type)
            loss = combined_loss(outputs.squeeze(), targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'num_features': num_features,
                'target_type': target_type
            }, model_path)
            # Save scaler
            scaler_path = model_path.replace('.pth', '_scaler.pkl')
            joblib.dump(scaler, scaler_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 25 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.8f}')
        
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    try:
        checkpoint = torch.load(model_path, weights_only=True)
    except:
        checkpoint = torch.load(model_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = joblib.load(scaler_path)
    
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data, target_type=target_type)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    
    # Convert back from log space
    all_preds = np.expm1(np.array(all_preds))
    all_targets = np.expm1(np.array(all_targets))
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mape = mean_absolute_percentage_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"=== {target_type.upper()} Integration Results ===")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.4%}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Model saved to '{model_path}'\n")
    
    return model, scaler, {'mae': mae, 'mape': mape, 'r2': r2}


# --- Enhanced Data Preprocessing ---
def preprocess_hardware_data(df):
    """
    Enhanced preprocessing for integrated power prediction
    """
    print("--- Enhanced Data Preprocessing for Integrated Model ---")
    
    # Convert boolean columns
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    # Enhanced feature engineering for power prediction
    if 'fetch-l1-sets' in df.columns and 'fetch-l1-ways' in df.columns:
        df['fetch-l1-capacity'] = df['fetch-l1-sets'] * df['fetch-l1-ways']
        df['fetch-l1-complexity'] = df['fetch-l1-sets'] + df['fetch-l1-ways']
    
    if 'lsu-l1-sets' in df.columns and 'lsu-l1-ways' in df.columns:
        df['lsu-l1-capacity'] = df['lsu-l1-sets'] * df['lsu-l1-ways']
        df['lsu-l1-complexity'] = df['lsu-l1-sets'] + df['lsu-l1-ways']
    
    # Power-related complexity indicators
    df['branch_pred_complexity'] = (
        df.get('with-gshare', 0) + df.get('with-btb', 0) + df.get('with-ras', 0)
    )
    
    df['execution_complexity'] = (
        df.get('with-mul', 0) + df.get('with-div', 0) + df.get('with-rvc', 0)
    )
    
    # Cache total complexity
    df['cache_total_complexity'] = (
        df.get('fetch-l1-capacity', 0) + df.get('lsu-l1-capacity', 0)
    )
    
    # Pipeline complexity
    df['pipeline_complexity'] = (
        df.get('decoders', 0) * df.get('lanes', 0) * df.get('dispatcher-at', 1)
    )
    
    print(f"Added {7} engineered features for power prediction")
    return df


# --- Main Integrated Training Pipeline ---
if __name__ == "__main__":
    print("=== 整合功耗预测模型训练 ===\n")
    
    # Load and preprocess data
    df = pd.read_csv('power_with_vex_parameters.csv')
    df = preprocess_hardware_data(df)
    
    # Filter complete designs
    print("--- Filtering Data ---")
    phase_counts = df.groupby('Design Name')['Backend Phase'].nunique()
    complete_phases_count = phase_counts.max()
    complete_designs = phase_counts[phase_counts == complete_phases_count].index
    df_filtered = df[df['Design Name'].isin(complete_designs)].copy()
    print(f"Filtered data to {len(complete_designs)} complete designs.")
    
    # Focus on chipfinish phase
    df_chipfinish = df_filtered[df_filtered['Backend Phase'] == 'chipfinish'].copy()
    
    # Calculate integration targets
    print("--- Calculating Integration Targets ---")
    targets = calculate_integration_targets(df_chipfinish)
    
    # Add targets to dataframe
    for target_name, target_values in targets.items():
        df_chipfinish[f'Power_{target_name}'] = target_values
    
    # Print target statistics
    for target_name in targets.keys():
        values = targets[target_name] * 1e-6  # Convert to mW
        print(f"{target_name:15} - Range: {values.min():.6f} to {values.max():.6f} mW, "
              f"Mean: {values.mean():.6f} mW")
    
    # Prepare features
    feature_start_idx = df_chipfinish.columns.get_loc('xlen')
    # Get features but exclude the original Power1, Power2, Power3 columns and target columns
    all_features = list(df_chipfinish.columns[feature_start_idx:])
    features_base = [f for f in all_features if not f.startswith('Power')]
    
    print(f"\nUsing {len(features_base)} features: {features_base[:10]}...")
    
    X = df_chipfinish[features_base].values
    
    # Prepare targets dictionary (convert to mW)
    y_targets = {}
    for target_name in targets.keys():
        y_targets[target_name] = df_chipfinish[f'Power_{target_name}'].values * 1e-6
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    y_targets_train = {}
    y_targets_test = {}
    for target_name in targets.keys():
        y_train, y_test = train_test_split(y_targets[target_name], test_size=0.2, random_state=42)
        y_targets_train[target_name] = y_train
        y_targets_test[target_name] = y_test
    
    print(f"\nTraining data shape: X={X_train.shape}")
    print(f"Test data shape: X={X_test.shape}")
    
    # Train models for each integration target
    print("\n=== 开始训练整合模型 ===")
    
    integration_results = {}
    
    for target_type in ['weighted_avg', 'worst_case', 'rms', 'statistical']:
        print(f"\n{'='*50}")
        print(f"训练 {target_type.upper()} 整合模型")
        print(f"{'='*50}")
        
        model_path = f'integrated_transformer_{target_type}.pth'
        
        model, scaler, metrics = train_integrated_transformer(
            X_train, y_targets_train, X_test, y_targets_test,
            X.shape[1], model_path, target_type=target_type,
            epochs=250, batch_size=32, lr=0.0008
        )
        
        integration_results[target_type] = metrics
    
    # Final results summary
    print("="*60)
    
    print("\n 各种整合方法的预测精度：")
    for target_type, metrics in integration_results.items():
        print(f"{target_type:15} - MAPE: {metrics['mape']:6.2%}, "
              f"R²: {metrics['r2']:6.4f}, MAE: {metrics['mae']:8.6f}")
    
    # Find best method
    best_method = min(integration_results.keys(), 
                     key=lambda x: integration_results[x]['mape'])
    
    
    print(f"\n 训练好的模型文件：")
    for target_type in integration_results.keys():
        print(f"   - integrated_transformer_{target_type}.pth")
        print(f"   - integrated_transformer_{target_type}_scaler.pkl")
