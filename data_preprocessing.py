import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 加载数据
df = pd.read_csv('power_with_vex_parameters.csv')

# 1. 分离数值特征和非数值特征
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

# 2. 对数值特征进行均值填充
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# 3. 对非数值特征进行最频繁值填充
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# 4. 输出填充后的数据
print("填充后的数据：")
print(df.head())

# 保存处理后的数据
df.to_csv('preprocessed_data.csv', index=False)
print("预处理后的数据已保存为 'preprocessed_data.csv'")
