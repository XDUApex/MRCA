import os
import pandas as pd
from config import get_dataset_config

def convert_data(dataset_name):
    """根据数据集配置将聚合数据转换为统一的2维格式，并覆盖原始文件"""
    # 获取数据集配置
    config = get_dataset_config(dataset_name)

    # 处理 normal 数据
    process_data(config['processed_data_path'], config['normal_data']['dates'], 'normal')

    # 处理 abnormal 数据
    process_data(config['processed_data_path'], config['abnormal_data']['dates'], 'abnormal')

def process_data(base_path, dates, data_type):
    """处理指定类型的数据（normal 或 abnormal）"""
    for date in dates:
        input_folder = os.path.join(base_path, data_type, date, 'aggregation')
        if not os.path.exists(input_folder):
            print(f"Warning: Aggregation folder for {data_type} data on date {date} does not exist. Skipping...")
            continue

        for filename in os.listdir(input_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(input_folder, filename)

                # 读取数据
                data = pd.read_csv(file_path)

                # 检查数据是否包含必要的列
                if 'Timestamp' in data.columns:
                    # 聚合日志模板特征
                    # log_template_sum = data.iloc[:, 1:-1].sum(axis=1)  # 求和日志模板特征
                    log_template_max = data.iloc[:, 1:-1].max(axis=1)  # 求最大值日志模板特征
                    trace_latency = data.iloc[:, -1]  # 提取最后一列作为 TraceLatency
                else:
                    print(f"Warning: {filename} does not contain required columns. Skipping...")
                    continue

                # 创建新的 DataFrame
                converted_data = pd.DataFrame({
                    'Timestamp': data['Timestamp'],
                    'LogTemplateSum': log_template_max,
                    'TraceLatency': trace_latency
                })

                # 覆盖原始文件
                converted_data.to_csv(file_path, index=False)
                print(f"Converted data saved to {file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert aggregated data to 2D format for specified dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., ob, tt, gaia, aiops)")
    args = parser.parse_args()

    convert_data(args.dataset)