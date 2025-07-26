import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings('ignore')

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
        print("💻 GPU not available. Using CPU for training.")


# --- Transformer Model Definition ---
class TransformerPredictor(nn.Module):
    def __init__(self, num_features, d_model=256, nhead=8, num_encoder_layers=8, dim_feedforward=256, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        # Input embedding layer to project features to d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        
        # 基于数据分析和物理原理的corner特性
        self.corner_physics = {
            'Fast': {  # High-V, Low-T: 高动态功耗，低漏电
                'dynamic_boost': 2.0,
                'frequency_boost': 1.8,
                'control_boost': 1.5,
                'static_penalty': 0.8,
                'focus_features': ['xlen', 'decoders', 'lanes', 'with-mul', 'with-div', 'pipeline_complexity']
            },
            'Slow': {  # Low-V, High-T: 低动态功耗，高漏电
                'static_boost': 2.2,
                'leakage_boost': 2.0,
                'area_boost': 1.8,
                'dynamic_penalty': 0.7,
                'focus_features': ['fetch-l1-capacity', 'lsu-l1-capacity', 'cache_total_complexity', 'execution_complexity']
            },
            'Typical': {  # Nominal-V, Room-T: 平衡状态
                'balanced_boost': 1.5,
                'control_boost': 1.4,
                'cache_boost': 1.3,
                'general_factor': 1.0,
                'focus_features': ['branch_pred_complexity', 'with-gshare', 'with-btb', 'pipeline_complexity']
            }
        }
        
        # 可学习的物理权重网络
        self.physics_weight_net = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid()  # 输出0-1的权重
        )
        
        # Corner特异性的固定权重 (基于物理分析)
        self.register_buffer('corner_bias', torch.ones(num_features))
        self._initialize_corner_bias()
    
    def _initialize_corner_bias(self):
        """基于corner物理特性初始化偏置"""
        corner_config = self.corner_physics[self.corner_type]
        
        if self.corner_type == 'Fast':
            self.corner_bias *= corner_config['dynamic_boost']
        elif self.corner_type == 'Slow':
            self.corner_bias *= corner_config['static_boost']
        else:  # Typical
            self.corner_bias *= corner_config['balanced_boost']
    
    def forward(self, features):
        # 学习的自适应权重
        learned_weights = self.physics_weight_net(features)
        
        # 结合物理偏置和学习权重
        corner_bias_expanded = self.corner_bias.unsqueeze(0).expand_as(features)
        combined_weights = learned_weights * corner_bias_expanded
        
        # 应用权重
        weighted_features = features * combined_weights
        
        return weighted_features, combined_weights


class UnifiedCornerSpecificTransformer(nn.Module):
    """
    统一维度的Corner特异性Transformer模型
    所有corner使用相同的d_model以避免维度不匹配
    """
    def __init__(self, num_features, corner_type='Fast', d_model=128, nhead=8, 
                 num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.corner_type = corner_type
        self.d_model = d_model  # 统一的d_model
        self.num_features = num_features
        
        # Stage 1集成: 物理感知特征权重
        self.feature_weighting = PhysicsAwareFeatureWeighting(num_features, corner_type)
        
        # Corner特异性配置 - 保持d_model统一，调整其他参数
        corner_configs = {
            'Fast': {
                'nhead': 8, 'num_layers': 4, 'dropout': 0.08,
                'dim_feedforward': 512,
                'description': 'High dynamic power, frequency sensitive'
            },
            'Slow': {
                'nhead': 4, 'num_layers': 3, 'dropout': 0.12,  # 使用4个head确保整除
                'dim_feedforward': 384,
                'description': 'High leakage power, area sensitive'
            },
            'Typical': {
                'nhead': 8, 'num_layers': 4, 'dropout': 0.10,
                'dim_feedforward': 512,
                'description': 'Balanced power characteristics'
            }
        }
        
        config = corner_configs[corner_type]
        
        print(f"Initializing {corner_type} Corner Model: {config['description']}")
        print(f"   d_model={d_model}, nhead={config['nhead']}, layers={config['num_layers']}")
        
        # 输入嵌入层 - 统一输出维度
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 位置编码 - 统一维度
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )
        
        # Corner特异性Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
    def forward(self, src):
        embedded_src = self.input_embedding(src)  # -> (batch_size, d_model)
        transformer_input = embedded_src.unsqueeze(1)  # -> (batch_size, 1, d_model)
        transformer_output = self.transformer_encoder(transformer_input) # -> (batch_size, 1, d_model)
        transformer_output = transformer_output.squeeze(1) # -> (batch_size, d_model)
        output = self.output_layer(transformer_output) # -> (batch_size, 1)
        return output


# --- Training and Evaluation Function ---
def run_transformer_pipeline(X_train, y_train, X_test, y_test, num_features, model_path, epochs=1000, batch_size=8, lr=0.00005, patience=100):
    """
    Handles the entire process of scaling, training, and evaluating the Transformer model.
    Includes learning rate scheduler and early stopping.
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            # 只在非最后一层添加激活和归一化
            if i < len(config['layers']) - 2:
                layers.append(nn.LayerNorm(config['layers'][i+1]))
                if config['activation'] == 'gelu':
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout * (0.5 ** i)))
        
        # Check if validation loss improves
        # scheduler.step(total_loss / len(train_loader))
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler
            }, model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break


# =============================================================================
# STAGE 2: ENSEMBLE INTEGRATION
# =============================================================================

class AdvancedEnsembleIntegrator(nn.Module):
    """
    Stage 2: 高级Ensemble整合器
    智能融合三个corner模型的预测
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # 基于数据分析的先验权重
        self.integration_methods = {
            'weighted_avg': torch.tensor([0.2, 0.2, 0.6]),  # Fast, Slow, Typical
            'worst_case': torch.tensor([0.4, 0.4, 0.2]),     # 更关注极端情况
            'best_case': torch.tensor([0.3, 0.3, 0.4]),      # 更关注典型情况
            'rms': torch.tensor([0.33, 0.33, 0.34]),         # 等权重RMS
            'conservative': torch.tensor([0.1, 0.3, 0.6])     # 保守估计
        }
        
        # 动态权重预测网络
        self.dynamic_weight_net = nn.Sequential(
            nn.Linear(6, d_model),  # 3个预测值 + 3个不确定性值
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Softmax(dim=1)
        )
        
        # 不确定性加权网络
        self.uncertainty_weight_net = nn.Sequential(
            nn.Linear(3, d_model // 2),  # 3个不确定性值
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Softmax(dim=1)
        )
        
        # 上下文感知权重调整
        self.context_adjustment = nn.Sequential(
            nn.Linear(3, d_model // 4),  # 3个预测值
            nn.GELU(),
            nn.Linear(d_model // 4, 3),
            nn.Tanh()  # 输出调整因子
        )
    
    def forward(self, predictions, uncertainties=None, method='adaptive'):
        """
        predictions: [batch_size, 3] - Fast, Slow, Typical预测
        uncertainties: [batch_size, 3] - 对应的不确定性 (可选)
        method: 整合方法
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        if method == 'adaptive' and uncertainties is not None:
            # 自适应权重整合
            
            # 基于预测值和不确定性的动态权重
            input_features = torch.cat([predictions, uncertainties], dim=1)
            dynamic_weights = self.dynamic_weight_net(input_features)
            
            # 基于不确定性的权重 (不确定性越小权重越大)
            uncertainty_weights = self.uncertainty_weight_net(uncertainties)
            inv_uncertainty = 1.0 / (uncertainties + 1e-8)
            inv_uncertainty = inv_uncertainty / inv_uncertainty.sum(dim=1, keepdim=True)
            
            # 上下文调整
            context_adj = self.context_adjustment(predictions)
            
            # 组合权重
            combined_weights = 0.4 * dynamic_weights + 0.4 * inv_uncertainty + 0.2 * uncertainty_weights
            combined_weights = combined_weights * (1 + 0.1 * context_adj)
            
            # 重新归一化
            final_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
            
        elif method in self.integration_methods:
            # 使用预定义的权重
            prior_weights = self.integration_methods[method].to(device)
            final_weights = prior_weights.unsqueeze(0).repeat(batch_size, 1)
            
        else:
            # 默认等权重
            final_weights = torch.ones(batch_size, 3, device=device) / 3
        
        # 执行加权融合
        integrated_prediction = (predictions * final_weights).sum(dim=1, keepdim=True)
        
        return integrated_prediction, final_weights


# =============================================================================
# THREE-STAGE TRAINING PIPELINE
# =============================================================================

class ThreeStageCornerTrainer:
    """
    Corner特异性训练流程
    """
    def __init__(self, df_chipfinish, features):
        self.df_chipfinish = df_chipfinish
        self.features = features
        self.power_cols = ['Power1', 'Power2', 'Power3']
        self.corner_names = ['Fast', 'Slow', 'Typical']
        
        # 存储训练结果
        self.corner_models = {}
        self.corner_scalers = {}
        self.stage1_results = {}
        
        print(f"Initializing Three-Stage Corner Training")
        print(f"Dataset: {len(df_chipfinish)} designs, {len(features)} features")
        print(f"Corners: {self.corner_names}")
    
    def stage1_train_corner_models(self, epochs=300, batch_size=32, lr=0.001):
        """
        Stage 1: 训练Corner特异性模型
        """
        print(f"\n" + "="*60)
        print(f"STAGE 1: CORNER-SPECIFIC MODEL TRAINING")
        print(f"="*60)
        
        for power_col, corner_name in zip(self.power_cols, self.corner_names):
            print(f"\nTraining {corner_name} Corner Model")
            print(f"-" * 40)
            
            # 准备数据
            X = self.df_chipfinish[self.features].select_dtypes(include=[np.number]).values
            y = self.df_chipfinish[power_col].values * 1e-6  # 转换为mW
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 特征缩放 (使用QuantileTransformer更robust)
            scaler = QuantileTransformer(output_distribution='normal', random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 目标变换 (log变换)
            y_train_log = np.log1p(y_train)
            y_test_log = np.log1p(y_test)
            
            print(f"   Data: {X_train.shape[0]} train, {X_test.shape[0]} test")
            print(f"   Power range: {y_train.min():.6f} - {y_train.max():.6f} mW")
            
            # 创建数据加载器
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.FloatTensor(y_train_log)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_scaled),
                torch.FloatTensor(y_test_log)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # 创建统一维度的corner特异性模型
            model = UnifiedCornerSpecificTransformer(
                num_features=X.shape[1],
                corner_type=corner_name,
                d_model=128  # 统一的d_model
            ).to(device)
            
            # 优化器和损失函数
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, eta_min=lr*0.01
            )
            
            # 组合损失函数
            def corner_specific_loss(pred, target, corner_type):
                huber_loss = F.huber_loss(pred.squeeze(), target, delta=0.5)
                mse_loss = F.mse_loss(pred.squeeze(), target)
                
                # Corner特异性权重
                if corner_type == 'Fast':
                    return 0.8 * huber_loss + 0.2 * mse_loss  # 更关注robustness
                elif corner_type == 'Slow':
                    return 0.6 * huber_loss + 0.4 * mse_loss  # 平衡
                else:  # Typical
                    return 0.7 * huber_loss + 0.3 * mse_loss
            
            # 训练循环
            best_loss = float('inf')
            patience = 0
            patience_limit = 25
            
            print(f"   Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                num_batches = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = corner_specific_loss(pred, batch_y, corner_name)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                
                # 验证和早停
                if epoch % 20 == 0 or epoch == epochs - 1:
                    model.eval()
                    val_preds = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            batch_x = batch_x.to(device)
                            pred = model(batch_x)
                            val_preds.extend(pred.cpu().numpy().flatten())
                            val_targets.extend(batch_y.numpy())
                    
                    # 转换回原始尺度
                    val_preds = np.expm1(np.array(val_preds))
                    val_targets = np.expm1(np.array(val_targets))
                    
                    val_mape = mean_absolute_percentage_error(val_targets, val_preds)
                    val_r2 = r2_score(val_targets, val_preds)
                    
                    print(f"   Epoch [{epoch+1:3d}/{epochs}]: Loss={avg_loss:.6f}, "
                          f"Val MAPE={val_mape:.2%}, Val R²={val_r2:.4f}")
                
                # 早停检查
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience = 0
                    
                    # 保存最佳模型
                    model_path = f'{corner_name.lower()}_corner_model_unified.pth'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'loss': best_loss,
                        'corner_type': corner_name
                    }, model_path)
                    
                    scaler_path = f'{corner_name.lower()}_corner_scaler_unified.pkl'
                    joblib.dump(scaler, scaler_path)
                else:
                    patience += 1
                
                if patience >= patience_limit:
                    print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # 加载最佳模型并进行最终评估
            checkpoint = torch.load(f'{corner_name.lower()}_corner_model_unified.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 最终评估
            model.eval()
            final_preds = []
            final_targets = []
            final_uncertainties = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    pred, uncertainty = model(batch_x, return_uncertainty=True)
                    
                    final_preds.extend(pred.cpu().numpy().flatten())
                    final_uncertainties.extend(uncertainty.cpu().numpy().flatten())
                    final_targets.extend(batch_y.numpy())
            
            final_preds = np.expm1(np.array(final_preds))
            final_targets = np.expm1(np.array(final_targets))
            
            mae = mean_absolute_error(final_targets, final_preds)
            mape = mean_absolute_percentage_error(final_targets, final_preds)
            r2 = r2_score(final_targets, final_preds)
            
            print(f"   Final Results: MAE={mae:.6f}, MAPE={mape:.2%}, R²={r2:.4f}")
            
            # 存储结果
            self.corner_models[corner_name] = model
            self.corner_scalers[corner_name] = scaler
            self.stage1_results[corner_name] = {
                'mae': mae, 'mape': mape, 'r2': r2,
                'model_path': f'{corner_name.lower()}_corner_model_unified.pth',
                'scaler_path': f'{corner_name.lower()}_corner_scaler_unified.pkl'
            }
        
        return self.stage1_results
    
    # Print results
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"R^2 Score: {r2:.4f}")
    
    print(f"Model and scaler saved to '{model_path}'\n")


# --- Main Script Logic ---
df = pd.read_csv('preprocessed_data.csv')

def main_three_stage_corner_training():
    """
    主函数：corner特异性训练
    """
    print(f"Loading data and starting three-stage corner training...")
    
    # 使用您现有的数据预处理逻辑
    df = pd.read_csv('power_with_vex_parameters.csv')
    df = preprocess_hardware_data(df)
    
    # 过滤完整设计
    phase_counts = df.groupby('Design Name')['Backend Phase'].nunique()
    complete_phases_count = phase_counts.max()
    complete_designs = phase_counts[phase_counts == complete_phases_count].index
    df_filtered = df[df['Design Name'].isin(complete_designs)].copy()
    
    # 聚焦chipfinish阶段
    df_chipfinish = df_filtered[df_filtered['Backend Phase'] == 'chipfinish'].copy()
    
    # 获取特征
    feature_start_idx = df_chipfinish.columns.get_loc('xlen')
    all_features = list(df_chipfinish.columns[feature_start_idx:])
    features_base = [f for f in all_features if not f.startswith('Power')]
    
    print(f"Data loaded and preprocessed:")
    print(f"   {len(df_chipfinish)} designs")
    print(f"   {len(features_base)} features")
    print(f"   ⚡ 3 power corners (Fast, Slow, Typical)")
    
    # 创建并运行训练器
    trainer = ThreeStageCornerTrainer(df_chipfinish, features_base)
    stage1_results, stage2_results, stage3_results = trainer.run_complete_pipeline()
    
    return trainer, stage1_results, stage2_results, stage3_results


# 使用您现有的预处理函数
def preprocess_hardware_data(df):
    """
    Enhanced preprocessing for integrated power prediction
    """
    print("--- Enhanced Data Preprocessing for Corner-Specific Models ---")
    
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
    
    print(f"Added {7} engineered features for corner-specific power prediction")
    return df


# # --- Scenario 2: Predicting with floorplan power ---
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
