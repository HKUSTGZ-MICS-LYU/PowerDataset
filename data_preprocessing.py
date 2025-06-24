# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys

def process_csv_file(input_path, output_path):
    """
    读取CSV文件，拆分'File Path'列，并保存为新文件。

    参数:
    input_path (str): 输入CSV文件的路径。
    output_path (str): 处理后要保存的CSV文件的路径。
    """
    try:
        # 1. 读取CSV文件到pandas DataFrame
        # 使用'utf-8-sig'编码以正确处理可能由Excel等软件保存时添加的BOM头
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"成功读取文件: {input_path}")
        print("原始数据前5行:")
        print(df.head())

        # 2. 检查'File Path'列是否存在
        if 'File Path' not in df.columns:
            print(f"错误: 输入文件中未找到 'File Path' 列。")
            print(f"可用列为: {df.columns.tolist()}")
            return

        # 3. 提取 'Design Name'
        # 逻辑:
        # a. .str.split('/') -> 按'/'拆分路径
        # b. .str[-4] -> 获取倒数第四个元素 (例如 'VexiiRiscv_***_100')
        # c. .str.rsplit('_', n=1) -> 从右边按最后一个'_'拆分 (例如 ['VexiiRiscv_***', '100'])
        # d. .str[0] -> 获取拆分后的第一个部分 (例如 'VexiiRiscv_***')
        df['Design Name'] = df['File Path'].str.split('/').str[-4].str.rsplit('_', n=1).str[0]

        # 4. 提取 'Backend Phase'
        # 逻辑:
        # a. .str.split('/') -> 按'/'拆分路径
        # b. .str[-1] -> 获取最后一个元素, 即文件名 (例如 'synthesis.report_power')
        # c. .str.split('.') -> 按'.'拆分文件名 (例如 ['synthesis', 'report_power'])
        # d. .str[0] -> 获取第一个部分 (例如 'synthesis')
        df['Backend Phase'] = df['File Path'].str.split('/').str[-1].str.split('.').str[0]

        # 5. 删除原始的 'File Path' 列 (可选)
        df_processed = df.drop(columns=['File Path'])
        
        # 6. 将新的列移动到前面，方便查看 (可选)
        # 获取所有列的列表
        cols = df_processed.columns.tolist()
        # 将 'Design Name' 和 'Backend Phase' 移动到列表最前面
        new_order = ['Design Name', 'Backend Phase'] + [col for col in cols if col not in ['Design Name', 'Backend Phase']]
        df_final = df_processed[new_order]


        # 7. 将处理后的DataFrame保存到新的CSV文件
        # index=False 表示在保存时不要将DataFrame的索引写入文件
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n文件处理完成。")
        print(f"处理后的数据已保存到: {output_path}")
        print("处理后数据前5行:")
        print(df_final.head())

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {input_path}")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

if __name__ == '__main__':
    # --- 脚本执行入口 ---
    # 该脚本从命令行接收参数。
    # 使用方法: python your_script_name.py <input_file.csv> <output_file.csv>

    # 1. 检查命令行参数数量是否正确
    if len(sys.argv) != 3:
        print("\n错误: 参数数量不正确。")
        print("用法: python process_csv.py <输入文件路径> <输出文件路径>")
        
        # 创建一个示例文件，以便用户知道输入格式
        print("\n正在为您创建一个示例输入文件 'input_sample.csv'...")
        create_sample_csv("input_sample.csv")
        print("示例文件创建成功。请参照此格式准备您的输入文件。")
        sys.exit(1) # 退出脚本

    # 2. 从命令行获取输入和输出文件路径
    input_csv_file = sys.argv[1]
    output_csv_file = sys.argv[2]

    # 3. 执行处理函数
    print("--------------------------------------------------")
    print(f"准备处理文件...")
    print(f"输入文件: {input_csv_file}")
    print(f"输出文件: {output_csv_file}")
    print("--------------------------------------------------")
    process_csv_file(input_csv_file, output_csv_file)