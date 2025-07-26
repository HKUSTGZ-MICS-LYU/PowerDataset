import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

print("500 Comprehensive Hardware Power Prediction Data Analysis")
print("="*80)


class ComprehensiveDataAnalyzer:
    """
    综合数据分析器 - 深度分析硬件功耗数据特性
    """
    
    def __init__(self, df):
        self.df = df
        self.power_cols = ['Power1', 'Power2', 'Power3']
        self.corner_names = ['Fast', 'Slow', 'Typical']
        self.corner_conditions = ['High-V, Low-T', 'Low-V, High-T', 'Nominal-V, Room-T']
        self.analysis_results = {}
        
        # 获取chipfinish阶段数据
        self.df_chip = df[df['Backend Phase'] == 'chipfinish'].copy()
        
        # 获取特征列
        feature_start = self.df_chip.columns.get_loc('xlen')
        self.all_features = list(self.df_chip.columns[feature_start:])
        self.features = [f for f in self.all_features if not f.startswith('Power')]
        
        print(f"Analysis Dataset: {len(self.df_chip)} designs, {len(self.features)} features")
        print(f"Hardware Features: {self.features[:10]}..." if len(self.features) > 10 else f"Hardware Features: {self.features}")
    
    def phase_1_basic_statistics(self):
        """
        Phase 1: 基础统计分析
        """
        print(f"\nPhase 1: Basic Statistical Analysis")
        print("="*50)
        
        stats_results = {}
        
        # 1.1 功耗分布统计
        print("1.1 Power Distribution Statistics")
        power_stats = pd.DataFrame()
        
        for i, (power_col, corner_name, condition) in enumerate(zip(
            self.power_cols, self.corner_names, self.corner_conditions)):
            
            power_data = self.df_chip[power_col] * 1e-6  # 转换为mW
            
            stats_dict = {
                'Corner': corner_name,
                'Condition': condition,
                'Count': len(power_data),
                'Mean': power_data.mean(),
                'Std': power_data.std(),
                'Min': power_data.min(),
                'Max': power_data.max(),
                'Range': power_data.max() - power_data.min(),
                'CV': power_data.std() / power_data.mean(),  # 变异系数
                'Skewness': stats.skew(power_data),
                'Kurtosis': stats.kurtosis(power_data),
                'Q25': power_data.quantile(0.25),
                'Q50': power_data.quantile(0.50),
                'Q75': power_data.quantile(0.75),
                'IQR': power_data.quantile(0.75) - power_data.quantile(0.25)
            }
            
            power_stats = pd.concat([power_stats, pd.DataFrame([stats_dict])], ignore_index=True)
        
        print(power_stats.round(6))
        stats_results['power_distribution'] = power_stats
        
        # 1.2 功耗比值分析
        print(f"\n1.2 Power Ratio Analysis")
        self.df_chip['Fast_to_Slow'] = self.df_chip['Power1'] / self.df_chip['Power2']
        self.df_chip['Fast_to_Typical'] = self.df_chip['Power1'] / self.df_chip['Power3']
        self.df_chip['Slow_to_Typical'] = self.df_chip['Power2'] / self.df_chip['Power3']
        
        ratio_cols = ['Fast_to_Slow', 'Fast_to_Typical', 'Slow_to_Typical']
        ratio_stats = {}
        
        for ratio_col in ratio_cols:
            data = self.df_chip[ratio_col]
            ratio_stats[ratio_col] = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'cv': data.std() / data.mean()
            }
            print(f"{ratio_col:20}: Mean={data.mean():.3f}, Std={data.std():.3f}, "
                  f"Range=[{data.min():.3f}, {data.max():.3f}], CV={data.std()/data.mean():.3f}")
        
        stats_results['ratio_analysis'] = ratio_stats
        
        # 1.3 分布检验
        print(f"\n500 1.3 Distribution Tests")
        distribution_tests = {}
        
        for power_col, corner_name in zip(self.power_cols, self.corner_names):
            power_data = self.df_chip[power_col] * 1e-6
            
            # 正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(power_data)
            
            # 对数正态性检验
            log_power = np.log(power_data + 1e-10)
            log_shapiro_stat, log_shapiro_p = stats.shapiro(log_power)
            
            distribution_tests[corner_name] = {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05,
                'log_shapiro_stat': log_shapiro_stat,
                'log_shapiro_p': log_shapiro_p,
                'is_log_normal': log_shapiro_p > 0.05
            }
            
            print(f"{corner_name:8} - Normal: {shapiro_p > 0.05} (p={shapiro_p:.6f}), "
                  f"Log-Normal: {log_shapiro_p > 0.05} (p={log_shapiro_p:.6f})")
        
        stats_results['distribution_tests'] = distribution_tests
        
        self.analysis_results['basic_statistics'] = stats_results
        return stats_results
    
    def phase_2_correlation_analysis(self):
        """
        Phase 2: 相关性分析
        """
        print(f"\nPhase 2: Correlation Analysis")
        print("="*50)
        
        corr_results = {}
        
        # 2.1 Corner间功耗相关性
        print("2.1 Inter-Corner Power Correlations")
        power_matrix = self.df_chip[self.power_cols]
        
        # Pearson相关性
        pearson_corr = power_matrix.corr(method='pearson')
        print("Pearson Correlations:")
        for i in range(len(self.power_cols)):
            for j in range(i+1, len(self.power_cols)):
                corr_val = pearson_corr.iloc[i, j]
                print(f"  {self.corner_names[i]} vs {self.corner_names[j]}: {corr_val:.4f}")
        
        # Spearman相关性 (处理非线性关系)
        spearman_corr = power_matrix.corr(method='spearman')
        print(f"\nSpearman Correlations:")
        for i in range(len(self.power_cols)):
            for j in range(i+1, len(self.power_cols)):
                corr_val = spearman_corr.iloc[i, j]
                print(f"  {self.corner_names[i]} vs {self.corner_names[j]}: {corr_val:.4f}")
        
        corr_results['power_correlations'] = {
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
        
        # 2.2 功耗与特征的相关性分析
        print(f"\n2.2 Power-Feature Correlations by Corner")
        feature_correlations = {}
        
        for power_col, corner_name in zip(self.power_cols, self.corner_names):
            print(f"\n{corner_name} Corner Top Correlations:")
            
            correlations = []
            for feature in self.features:
                if self.df_chip[feature].dtype in ['int64', 'float64']:
                    corr, p_value = pearsonr(self.df_chip[feature], self.df_chip[power_col])
                    correlations.append((feature, corr, p_value))
            
            # 按相关性强度排序
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # 显示top 10
            for i, (feature, corr, p_val) in enumerate(correlations[:10]):
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {i+1:2d}. {feature:25}: {corr:7.4f} {significance}")
            
            feature_correlations[corner_name] = correlations
        
        corr_results['feature_correlations'] = feature_correlations
        
        # 2.3 相关性差异分析
        print(f"\n2.3 Correlation Differences Across Corners")
        correlation_differences = {}
        
        # 计算特征相关性在不同corner间的差异
        for feature in self.features:
            if self.df_chip[feature].dtype in ['int64', 'float64']:
                corner_corrs = []
                for power_col in self.power_cols:
                    corr, _ = pearsonr(self.df_chip[feature], self.df_chip[power_col])
                    corner_corrs.append(corr)
                
                corr_std = np.std(corner_corrs)
                corr_range = max(corner_corrs) - min(corner_corrs)
                
                correlation_differences[feature] = {
                    'correlations': corner_corrs,
                    'std': corr_std,
                    'range': corr_range
                }
        
        # 显示相关性差异最大的特征
        top_diff_features = sorted(correlation_differences.items(), 
                                 key=lambda x: x[1]['std'], reverse=True)[:15]
        
        print("Features with Highest Correlation Variance Across Corners:")
        for i, (feature, diff_data) in enumerate(top_diff_features):
            print(f"  {i+1:2d}. {feature:25}: Std={diff_data['std']:.4f}, Range={diff_data['range']:.4f}")
            for j, corr in enumerate(diff_data['correlations']):
                print(f"      {self.corner_names[j]:8}: {corr:7.4f}")
        
        corr_results['correlation_differences'] = correlation_differences
        
        self.analysis_results['correlation_analysis'] = corr_results
        return corr_results
    
    def phase_3_feature_importance_analysis(self):
        """
        Phase 3: 特征重要性分析
        """
        print(f"\nPhase 3: Feature Importance Analysis")
        print("="*50)
        
        importance_results = {}
        
        # 3.1 随机森林特征重要性
        print("3.1 Random Forest Feature Importance by Corner")
        rf_importance = {}
        
        for power_col, corner_name in zip(self.power_cols, self.corner_names):
            print(f"\n{corner_name} Corner Feature Importance:")
            
            X = self.df_chip[self.features].select_dtypes(include=[np.number])
            y = self.df_chip[power_col] * 1e-6
            
            # 训练随机森林
            rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
            rf.fit(X, y)
            
            # 获取特征重要性
            importance = rf.feature_importances_
            feature_names = X.columns
            
            # 排序并显示
            importance_pairs = list(zip(feature_names, importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, imp) in enumerate(importance_pairs[:15]):
                print(f"  {i+1:2d}. {feature:25}: {imp:.6f}")
            
            rf_importance[corner_name] = dict(importance_pairs)
        
        importance_results['random_forest'] = rf_importance
        
        # 3.2 特征重要性差异分析
        print(f"\n3.2 Feature Importance Variance Analysis")
        
        all_features_set = set()
        for corner_imp in rf_importance.values():
            all_features_set.update(corner_imp.keys())
        
        importance_variance = {}
        for feature in all_features_set:
            importances = [rf_importance[corner].get(feature, 0) for corner in self.corner_names]
            variance = np.var(importances)
            mean_imp = np.mean(importances)
            cv = variance / (mean_imp + 1e-8)
            
            importance_variance[feature] = {
                'importances': importances,
                'variance': variance,
                'cv': cv,
                'mean': mean_imp
            }
        
        # 显示重要性差异最大的特征
        top_variance_features = sorted(importance_variance.items(), 
                                     key=lambda x: x[1]['cv'], reverse=True)[:20]
        
        print("Features with Highest Importance Variance (CV) Across Corners:")
        for i, (feature, var_data) in enumerate(top_variance_features):
            print(f"  {i+1:2d}. {feature:25}: CV={var_data['cv']:.4f}, Var={var_data['variance']:.6f}")
            for j, imp in enumerate(var_data['importances']):
                print(f"      {self.corner_names[j]:8}: {imp:.6f}")
        
        importance_results['importance_variance'] = importance_variance
        
        self.analysis_results['feature_importance'] = importance_results
        return importance_results
    
    def phase_4_clustering_analysis(self):
        """
        Phase 4: 聚类分析 - 发现设计模式
        """
        print(f"\nPhase 4: Clustering Analysis")
        print("="*50)
        
        clustering_results = {}
        
        # 4.1 基于功耗的聚类
        print("4.1 Power-based Clustering")
        
        # 准备功耗数据
        power_data = self.df_chip[self.power_cols].values * 1e-6
        
        # K-means聚类
        n_clusters = 5
        kmeans_power = KMeans(n_clusters=n_clusters, random_state=42)
        power_clusters = kmeans_power.fit_predict(power_data)
        
        self.df_chip['power_cluster'] = power_clusters
        
        # 分析每个聚类的特征
        print(f"Power Clusters Analysis ({n_clusters} clusters):")
        for cluster_id in range(n_clusters):
            cluster_data = self.df_chip[self.df_chip['power_cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            print(f"\nCluster {cluster_id} (n={cluster_size}):")
            for power_col, corner_name in zip(self.power_cols, self.corner_names):
                power_vals = cluster_data[power_col] * 1e-6
                print(f"  {corner_name:8}: {power_vals.mean():8.3f} ± {power_vals.std():6.3f} mW")
        
        clustering_results['power_clustering'] = {
            'clusters': power_clusters,
            'n_clusters': n_clusters,
            'centroids': kmeans_power.cluster_centers_
        }
        
        # 4.2 基于特征的聚类
        print(f"\n4.2 Feature-based Clustering")
        
        # 准备特征数据
        feature_data = self.df_chip[self.features].select_dtypes(include=[np.number])
        
        # 标准化
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        # K-means聚类
        kmeans_feature = KMeans(n_clusters=n_clusters, random_state=42)
        feature_clusters = kmeans_feature.fit_predict(feature_data_scaled)
        
        self.df_chip['feature_cluster'] = feature_clusters
        
        # 分析特征聚类的功耗特征
        print(f"Feature Clusters Power Analysis ({n_clusters} clusters):")
        for cluster_id in range(n_clusters):
            cluster_data = self.df_chip[self.df_chip['feature_cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            print(f"\nFeature Cluster {cluster_id} (n={cluster_size}):")
            for power_col, corner_name in zip(self.power_cols, self.corner_names):
                power_vals = cluster_data[power_col] * 1e-6
                print(f"  {corner_name:8}: {power_vals.mean():8.3f} ± {power_vals.std():6.3f} mW")
        
        clustering_results['feature_clustering'] = {
            'clusters': feature_clusters,
            'n_clusters': n_clusters
        }
        
        self.analysis_results['clustering_analysis'] = clustering_results
        return clustering_results
    
    def phase_5_dimensionality_analysis(self):
        """
        Phase 5: 维度分析 - PCA和t-SNE
        """
        print(f"\nPhase 5: Dimensionality Analysis")
        print("="*50)
        
        dim_results = {}
        
        # 5.1 PCA分析
        print("5.1 Principal Component Analysis")
        
        # 准备数据
        feature_data = self.df_chip[self.features].select_dtypes(include=[np.number])
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        # PCA
        pca = PCA()
        pca_features = pca.fit_transform(feature_data_scaled)
        
        # 解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print("PCA Explained Variance:")
        for i in range(min(10, len(explained_variance_ratio))):
            print(f"  PC{i+1:2d}: {explained_variance_ratio[i]:6.4f} "
                  f"(cumulative: {cumulative_variance[i]:6.4f})")
        
        # 找到解释90%方差的主成分数量
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        print(f"\nComponents needed for 90% variance: {n_components_90}/{len(explained_variance_ratio)}")
        
        dim_results['pca'] = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_90': n_components_90,
            'components': pca.components_
        }
        
        # 5.2 特征在主成分中的贡献
        print(f"\n5.2 Feature Contributions to Principal Components")
        
        # 分析前3个主成分
        feature_names = feature_data.columns
        for pc_idx in range(min(3, len(pca.components_))):
            print(f"\nPrincipal Component {pc_idx + 1} (explains {explained_variance_ratio[pc_idx]:.1%}):")
            
            # 获取特征贡献并排序
            contributions = list(zip(feature_names, pca.components_[pc_idx]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (feature, contrib) in enumerate(contributions[:10]):
                print(f"  {i+1:2d}. {feature:25}: {contrib:7.4f}")
        
        dim_results['feature_contributions'] = contributions
        
        self.analysis_results['dimensionality_analysis'] = dim_results
        return dim_results
    
    def phase_6_predictability_analysis(self):
        """
        Phase 6: 可预测性分析
        """
        print(f"\nPhase 6: Predictability Analysis")
        print("="*50)
        
        predict_results = {}
        
        # 6.1 单变量预测能力
        print("6.1 Univariate Predictive Power")
        
        univariate_r2 = {}
        for corner_name, power_col in zip(self.corner_names, self.power_cols):
            print(f"\n{corner_name} Corner - Top Univariate Predictors:")
            
            feature_r2 = []
            for feature in self.features:
                if self.df_chip[feature].dtype in ['int64', 'float64']:
                    X = self.df_chip[[feature]].values
                    y = self.df_chip[power_col].values * 1e-6
                    
                    # 简单线性回归的R²
                    try:
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression()
                        lr.fit(X, y)
                        r2 = lr.score(X, y)
                        feature_r2.append((feature, r2))
                    except:
                        continue
            
            # 排序并显示
            feature_r2.sort(key=lambda x: x[1], reverse=True)
            for i, (feature, r2) in enumerate(feature_r2[:10]):
                print(f"  {i+1:2d}. {feature:25}: R² = {r2:.6f}")
            
            univariate_r2[corner_name] = dict(feature_r2)
        
        predict_results['univariate_r2'] = univariate_r2
        
        # 6.2 交叉验证性能分析
        print(f"\n6.2 Cross-Validation Performance Analysis")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        
        cv_results = {}
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        for corner_name, power_col in zip(self.corner_names, self.power_cols):
            print(f"\n{corner_name} Corner CV Performance:")
            
            X = self.df_chip[self.features].select_dtypes(include=[np.number])
            y = self.df_chip[power_col] * 1e-6
            
            corner_cv = {}
            for model_name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    corner_cv[model_name] = {
                        'mean_r2': scores.mean(),
                        'std_r2': scores.std(),
                        'scores': scores
                    }
                    print(f"  {model_name:15}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
                except Exception as e:
                    print(f"  {model_name:15}: Error - {str(e)}")
                    corner_cv[model_name] = None
            
            cv_results[corner_name] = corner_cv
        
        predict_results['cross_validation'] = cv_results
        
        self.analysis_results['predictability_analysis'] = predict_results
        return predict_results
    
    def phase_7_corner_independence_test(self):
        """
        Phase 7: Corner独立性测试
        """
        print(f"\nPhase 7: Corner Independence Test")
        print("="*50)
        
        independence_results = {}
        
        # 7.1 功耗预测独立性测试
        print("500 7.1 Power Prediction Independence Test")
        
        # 使用一个corner的模型预测另一个corner
        cross_corner_performance = {}
        
        feature_data = self.df_chip[self.features].select_dtypes(include=[np.number])
        
        for train_corner, train_power in zip(self.corner_names, self.power_cols):
            cross_corner_performance[train_corner] = {}
            
            # 训练模型
            X_train = feature_data
            y_train = self.df_chip[train_power] * 1e-6
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 在所有corner上测试
            for test_corner, test_power in zip(self.corner_names, self.power_cols):
                y_test = self.df_chip[test_power] * 1e-6
                y_pred = model.predict(X_train)
                
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                cross_corner_performance[train_corner][test_corner] = {
                    'r2': r2,
                    'mape': mape
                }
        
        # 显示交叉预测性能
        print("Cross-Corner Prediction Performance (R²):")
        print("Train\\Test    ", end="")
        for test_corner in self.corner_names:
            print(f"{test_corner:>10}", end="")
        print()
        
        for train_corner in self.corner_names:
            print(f"{train_corner:12}", end="")
            for test_corner in self.corner_names:
                r2 = cross_corner_performance[train_corner][test_corner]['r2']
                print(f"{r2:10.4f}", end="")
            print()
        
        print(f"\nCross-Corner Prediction Performance (MAPE):")
        print("Train\\Test    ", end="")
        for test_corner in self.corner_names:
            print(f"{test_corner:>10}", end="")
        print()
        
        for train_corner in self.corner_names:
            print(f"{train_corner:12}", end="")
            for test_corner in self.corner_names:
                mape = cross_corner_performance[train_corner][test_corner]['mape']
                print(f"{mape:9.1%}", end="")
            print()
        
        independence_results['cross_corner_performance'] = cross_corner_performance
        
        # 7.2 特征模式差异测试
        print(f"\n7.2 Feature Pattern Difference Test")
        
        # 计算特征在不同corner下的预测重要性差异
        from sklearn.inspection import permutation_importance
        
        feature_importance_diff = {}
        
        for corner_name, power_col in zip(self.corner_names, self.power_cols):
            X = feature_data
            y = self.df_chip[power_col] * 1e-6
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)  # 减少数量以加快速度
            model.fit(X, y)
            
            # 置换重要性
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, scoring='r2'
            )
            
            feature_importance_diff[corner_name] = dict(zip(
                feature_data.columns, perm_importance.importances_mean
            ))
        
        # 计算特征重要性的标准差
        importance_std = {}
        for feature in feature_data.columns:
            importances = [feature_importance_diff[corner][feature] for corner in self.corner_names]
            importance_std[feature] = np.std(importances)
        
        # 显示重要性差异最大的特征
        top_diff_features = sorted(importance_std.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print("Features with Highest Importance Variability Across Corners:")
        for i, (feature, std_val) in enumerate(top_diff_features):
            print(f"  {i+1:2d}. {feature:25}: Std = {std_val:.6f}")
            for corner in self.corner_names:
                imp = feature_importance_diff[corner][feature]
                print(f"      {corner:8}: {imp:7.6f}")
        
        independence_results['feature_importance_diff'] = feature_importance_diff
        independence_results['importance_std'] = importance_std
        
        self.analysis_results['corner_independence'] = independence_results
        return independence_results
    
    def generate_comprehensive_visualizations(self):
        """
        生成综合可视化图表
        """
        print(f"\nGenerating Comprehensive Visualizations")
        print("="*50)
        
        # 创建综合分析图
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 功耗分布对比
        plt.subplot(3, 4, 1)
        power_data = []
        for power_col in self.power_cols:
            power_data.append(self.df_chip[power_col] * 1e-6)
        
        plt.boxplot(power_data, labels=self.corner_names)
        plt.title('Power Distribution by Corner')
        plt.ylabel('Power (mW)')
        plt.yscale('log')
        
        # 2. 相关性热图
        plt.subplot(3, 4, 2)
        corr_matrix = self.df_chip[self.power_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=self.corner_names, yticklabels=self.corner_names)
        plt.title('Inter-Corner Correlations')
        
        # 3. 功耗比值分布
        plt.subplot(3, 4, 3)
        plt.hist(self.df_chip['Fast_to_Slow'], bins=30, alpha=0.7, label='Fast/Slow', density=True)
        plt.hist(self.df_chip['Fast_to_Typical'], bins=30, alpha=0.7, label='Fast/Typical', density=True)
        plt.hist(self.df_chip['Slow_to_Typical'], bins=30, alpha=0.7, label='Slow/Typical', density=True)
        plt.xlabel('Power Ratio')
        plt.ylabel('Density')
        plt.title('Power Ratio Distributions')
        plt.legend()
        
        # 4. PCA解释方差
        plt.subplot(3, 4, 4)
        if 'dimensionality_analysis' in self.analysis_results:
            pca_data = self.analysis_results['dimensionality_analysis']['pca']
            cumulative_var = pca_data['cumulative_variance'][:20]  # 只显示前20个
            plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
            plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
            plt.xlabel('Principal Component')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance')
            plt.legend()
        
        # 5-8. 散点图矩阵
        for i, (power_col_1, corner_1) in enumerate(zip(self.power_cols, self.corner_names)):
            for j, (power_col_2, corner_2) in enumerate(zip(self.power_cols, self.corner_names)):
                if i < j:
                    subplot_idx = 5 + (i * 3 + j - i - 1)
                    if subplot_idx <= 8:
                        plt.subplot(3, 4, subplot_idx)
                        plt.scatter(self.df_chip[power_col_1] * 1e-6, 
                                  self.df_chip[power_col_2] * 1e-6, 
                                  alpha=0.6, s=20)
                        plt.xlabel(f'{corner_1} Power (mW)')
                        plt.ylabel(f'{corner_2} Power (mW)')
                        plt.title(f'{corner_1} vs {corner_2}')
                        plt.loglog()
        
        # 9. 特征重要性差异
        plt.subplot(3, 4, 9)
        if 'corner_independence' in self.analysis_results:
            importance_std = self.analysis_results['corner_independence']['importance_std']
            top_features = sorted(importance_std.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, stds = zip(*top_features)
            plt.barh(range(len(features)), stds)
            plt.yticks(range(len(features)), [f[:15] for f in features])
            plt.xlabel('Importance Std Across Corners')
            plt.title('Top Variable Features')
        
        # 10. 交叉预测性能
        plt.subplot(3, 4, 10)
        if 'corner_independence' in self.analysis_results:
            cross_perf = self.analysis_results['corner_independence']['cross_corner_performance']
            
            # 创建性能矩阵
            perf_matrix = np.zeros((3, 3))
            for i, train_corner in enumerate(self.corner_names):
                for j, test_corner in enumerate(self.corner_names):
                    perf_matrix[i, j] = cross_perf[train_corner][test_corner]['r2']
            
            sns.heatmap(perf_matrix, annot=True, cmap='viridis',
                       xticklabels=self.corner_names, yticklabels=self.corner_names)
            plt.title('Cross-Corner R² Performance')
            plt.xlabel('Test Corner')
            plt.ylabel('Train Corner')
        
        # 11. 功耗聚类
        plt.subplot(3, 4, 11)
        if 'clustering_analysis' in self.analysis_results:
            clusters = self.df_chip['power_cluster']
            for cluster_id in range(5):
                cluster_data = self.df_chip[clusters == cluster_id]
                if len(cluster_data) > 0:
                    plt.scatter(cluster_data['Power1'] * 1e-6, 
                              cluster_data['Power3'] * 1e-6,
                              label=f'Cluster {cluster_id}', alpha=0.7)
            plt.xlabel('Fast Power (mW)')
            plt.ylabel('Typical Power (mW)')
            plt.title('Power-based Clusters')
            plt.legend()
            plt.loglog()
        
        # 12. 变异系数分析
        plt.subplot(3, 4, 12)
        cv_data = []
        for _, row in self.df_chip.iterrows():
            powers = [row['Power1'], row['Power2'], row['Power3']]
            cv = np.std(powers) / np.mean(powers)
            cv_data.append(cv)
        
        plt.hist(cv_data, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Coefficient of Variation')
        plt.ylabel('Number of Designs')
        plt.title('Power Variability Across Corners')
        
        plt.tight_layout()
        plt.savefig('comprehensive_power_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comprehensive analysis plots saved as 'comprehensive_power_analysis.png'")
    
    def generate_summary_report(self):
        """
        生成分析总结报告
        """
        print(f"\nAnalysis Summary Report")
        print("="*80)
        
        print(f"COMPREHENSIVE HARDWARE POWER ANALYSIS REPORT")
        print(f"Generated: 2025-07-26 05:09:07 UTC")
        print(f"User: yuangungun-ty")
        print(f"Dataset: {len(self.df_chip)} designs, {len(self.features)} features")
        print("="*80)
        
        # 关键发现
        print(f"\nKEY FINDINGS:")
        
        # 1. 功耗差异
        power_ranges = {}
        for power_col, corner_name in zip(self.power_cols, self.corner_names):
            power_data = self.df_chip[power_col] * 1e-6
            power_ranges[corner_name] = power_data.max() / power_data.min()
        
        print(f"\n1. POWER RANGE ANALYSIS:")
        for corner, ratio in power_ranges.items():
            print(f"   {corner:8} Corner: {ratio:.1f}x dynamic range")
        
        # 2. 相关性分析
        if 'correlation_analysis' in self.analysis_results:
            corr_data = self.analysis_results['correlation_analysis']['power_correlations']['pearson']
            min_corr = corr_data.min().min()
            max_corr = corr_data.max().max()
            print(f"\n2. INTER-CORNER CORRELATIONS:")
            print(f"   Range: {min_corr:.3f} to {max_corr:.3f}")
            
            # 找出最低相关性
            for i in range(len(self.power_cols)):
                for j in range(i+1, len(self.power_cols)):
                    corr_val = corr_data.iloc[i, j]
                    print(f"   {self.corner_names[i]} vs {self.corner_names[j]}: {corr_val:.3f}")
        
        # 3. 特征重要性差异
        if 'feature_importance' in self.analysis_results:
            importance_var = self.analysis_results['feature_importance']['importance_variance']
            high_var_features = [f for f, data in importance_var.items() if data['cv'] > 0.5]
            
            print(f"\n3. FEATURE IMPORTANCE VARIABILITY:")
            print(f"   {len(high_var_features)} features show high variability (CV > 0.5) across corners")
            print(f"   Top variable features: {high_var_features[:5]}")
        
        # 4. 交叉预测性能
        if 'corner_independence' in self.analysis_results:
            cross_perf = self.analysis_results['corner_independence']['cross_corner_performance']
            
            # 计算对角线和非对角线平均性能
            diagonal_r2 = []
            off_diagonal_r2 = []
            
            for i, train_corner in enumerate(self.corner_names):
                for j, test_corner in enumerate(self.corner_names):
                    r2 = cross_perf[train_corner][test_corner]['r2']
                    if i == j:
                        diagonal_r2.append(r2)
                    else:
                        off_diagonal_r2.append(r2)
            
            diagonal_avg = np.mean(diagonal_r2)
            off_diagonal_avg = np.mean(off_diagonal_r2)
            performance_drop = diagonal_avg - off_diagonal_avg
            
            print(f"\n4. CROSS-CORNER PREDICTION PERFORMANCE:")
            print(f"   Same-corner R²: {diagonal_avg:.3f}")
            print(f"   Cross-corner R²: {off_diagonal_avg:.3f}")
            print(f"   Performance drop: {performance_drop:.3f} ({performance_drop/diagonal_avg*100:.1f}%)")
        
        
        print(f"\n" + "="*80)
        print(f"Analysis complete. Proceed with corner-specific modeling strategy.")


def main_comprehensive_analysis():
    """
    执行完整的数据分析流程
    """
    print("Starting Comprehensive Data Analysis Pipeline")
    print("="*80)
    
    # 加载数据
    try:
        df = pd.read_csv('power_with_vex_parameters.csv')
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("Error: power_with_vex_parameters.csv not found!")
        return None
    
    # 数据预处理
    print(f"\nData Preprocessing...")
    
    # 转换布尔列
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    # 特征工程
    if 'fetch-l1-sets' in df.columns and 'fetch-l1-ways' in df.columns:
        df['fetch-l1-capacity'] = df['fetch-l1-sets'] * df['fetch-l1-ways']
    
    if 'lsu-l1-sets' in df.columns and 'lsu-l1-ways' in df.columns:
        df['lsu-l1-capacity'] = df['lsu-l1-sets'] * df['lsu-l1-ways']
    
    df['branch_pred_complexity'] = (
        df.get('with-gshare', 0) + df.get('with-btb', 0) + df.get('with-ras', 0)
    )
    df['execution_complexity'] = (
        df.get('with-mul', 0) + df.get('with-div', 0) + df.get('with-rvc', 0)
    )
    
    # 过滤完整设计
    phase_counts = df.groupby('Design Name')['Backend Phase'].nunique()
    complete_phases_count = phase_counts.max()
    complete_designs = phase_counts[phase_counts == complete_phases_count].index
    df_filtered = df[df['Design Name'].isin(complete_designs)].copy()
    
    print(f"Filtered to {len(complete_designs)} complete designs")
    
    # 创建分析器
    analyzer = ComprehensiveDataAnalyzer(df_filtered)
    
    # 执行所有分析阶段
    print(f"\n500 Executing Analysis Phases...")
    
    # Phase 1: 基础统计
    analyzer.phase_1_basic_statistics()
    
    # Phase 2: 相关性分析
    analyzer.phase_2_correlation_analysis()
    
    # Phase 3: 特征重要性分析
    analyzer.phase_3_feature_importance_analysis()
    
    # Phase 4: 聚类分析
    analyzer.phase_4_clustering_analysis()
    
    # Phase 5: 维度分析
    analyzer.phase_5_dimensionality_analysis()
    
    # Phase 6: 可预测性分析
    analyzer.phase_6_predictability_analysis()
    
    # Phase 7: Corner独立性测试
    analyzer.phase_7_corner_independence_test()
    
    # 生成可视化
    analyzer.generate_comprehensive_visualizations()
    
    # 生成总结报告
    analyzer.generate_summary_report()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main_comprehensive_analysis()