import os
import time
import pandas as pd
import argparse
from datetime import datetime, timedelta
from config import get_dataset_config

def process_trace_files(input_folder, output_folder):
    """处理单个文件夹的trace文件"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 清理目标文件夹中的旧文件
    for existing_file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, existing_file)
        if os.path.isfile(file_path):
            print(f"Deleting old file: {file_path}")
            os.remove(file_path)

    # 开始处理 trace 文件
    for filename in sorted(os.listdir(input_folder)):  # 按文件名排序，确保按时间顺序处理
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")
            file_path = os.path.join(input_folder, filename)
            data = pd.read_csv(file_path)

            # 检查数据是否包含必要的列
            required_columns = ['SpanID', 'ParentID', 'PodName', 'StartTimeUnixNano', 'Duration']
            if not all(col in data.columns for col in required_columns):
                print(f"Warning: Missing required columns in {filename}. Skipping...")
                continue

            # 创建 ParentPodName 映射
            id_to_podname = data.set_index('SpanID')['PodName'].to_dict()
            data['ParentPodName'] = data['ParentID'].map(id_to_podname)

            # 分组并按时间窗口聚合
            grouped = data.groupby('ParentPodName')
            for pod_name, group in grouped:
                if pd.notna(pod_name):
                    # 按 5 秒窗口聚合，直接基于 StartTimeUnixNano 的纳秒格式
                    group['StartTimeUnixNano'] = group['StartTimeUnixNano'].astype(int)
                    group['WindowStart'] = (group['StartTimeUnixNano'] // (5 * 10**9)) * (5 * 10**9)  # 计算 5 秒窗口起点
                    aggregated_trace = group.groupby('WindowStart')['Duration'].mean().reset_index()  # 取平均值
                    aggregated_trace['Duration'] = aggregated_trace['Duration'].round().astype(int)  # 保留整数

                    # 重命名列，保持原始格式
                    aggregated_trace.rename(columns={'WindowStart': 'StartTimeUnixNano'}, inplace=True)

                    # 保存到文件，追加数据
                    output_file = os.path.join(output_folder, f"{pod_name}.csv")
                    if os.path.exists(output_file):
                        existing_data = pd.read_csv(output_file)
                        combined_data = pd.concat([existing_data, aggregated_trace]).drop_duplicates(subset='StartTimeUnixNano').sort_values('StartTimeUnixNano')
                        combined_data.to_csv(output_file, index=False)
                    else:
                        aggregated_trace.to_csv(output_file, index=False)

def process_multiple_days_trace(config, data_type):
    """处理多天的trace数据"""
    data_config = config[f'{data_type}_data']
    base_input_folder = data_config['path']
    base_output_folder = os.path.join(config['processed_data_path'], data_type)
    start_date = min(data_config['dates'])
    end_date = max(data_config['dates'])
    
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date_dt:
        date_str = current_date.strftime('%Y-%m-%d')
        input_folder = os.path.join(base_input_folder, date_str, 'trace')
        output_folder = os.path.join(base_output_folder, date_str, 'trace_latency')
        
        if os.path.exists(input_folder):
            print(f"Processing {data_type} trace data for {date_str}")
            process_trace_files(input_folder, output_folder)
        else:
            print(f"Warning: Trace folder not found for {date_str}")
            
        current_date += timedelta(days=1)

def main():
    parser = argparse.ArgumentParser(description='Process trace data for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (ob, tt, gaia, aiops)')
    
    args = parser.parse_args()
    
    # 获取数据集配置
    config = get_dataset_config(args.dataset)
    
    print(f"Processing {config['name']} trace data...")
    
    print("Processing normal data...")
    process_multiple_days_trace(config, 'normal')
    
    print("Processing abnormal data...")
    process_multiple_days_trace(config, 'abnormal')
    
    print(f"Trace processing completed for {config['name']}")

if __name__ == "__main__":
    main()