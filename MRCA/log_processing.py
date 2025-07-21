import os
import json
import pandas as pd
import argparse
from drain3 import TemplateMiner
from pathlib import Path
import time
from datetime import datetime, timedelta
import shutil
from config import get_dataset_config

def sort_and_save_logs(input_folder, output_folder):
    """处理单个文件夹的log文件"""
    os.makedirs(output_folder, exist_ok=True)
    all_data = pd.DataFrame()

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # print(f"Processing file: {filename}")
            csv_path = os.path.join(input_folder, filename)
            data = pd.read_csv(csv_path, on_bad_lines="skip") # tt数据集有时log不符合预期格式:2023-01-29 10_06_log.csv
            all_data = pd.concat([all_data, data], ignore_index=True)

    if 'Timestamp' in all_data.columns:
        all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], errors='coerce')
        all_data = all_data.dropna(subset=['Timestamp']) # tt数据集有时会出现时间格式错误:2023-01-29 08_59_log.csv

    for pod_name, group in all_data.groupby('PodName'):
        sorted_group = group.sort_values('Timestamp')
        safe_filename = f"{pod_name.replace('-', '_')}.csv"
        file_path = os.path.join(output_folder, safe_filename)
        sorted_group.to_csv(file_path, index=False)
        # print(f"All data for PodName {pod_name} saved in {file_path}")

def parse_log(log):
    try:
        outer_json = json.loads(log)
        if isinstance(outer_json['log'], str):
            return outer_json['log']
        else:
            inner_json = json.loads(outer_json['log'])
            return inner_json['message']
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in log: {log}")
        return None  # 跳过无法解析的日志
    except Exception as e:
        print(f"Error processing log: {log}")
        return None  # 跳过其他错误

def process_log_file(log_path, output_dir):
    """处理单个log文件"""
    log_data = pd.read_csv(log_path)
    log_data['Log'] = log_data['Log'].apply(parse_log)
    
    # 删除无法解析的日志行
    log_data = log_data.dropna(subset=['Log'])
    # print(f"Parsed logs for {log_path}: {len(log_data)} rows")

    template_miner = TemplateMiner()
    log_data['template_id'] = log_data['Log'].apply(lambda log_message: template_miner.add_log_message(log_message)['cluster_id'])
    # print(f"Generated template IDs for {log_path}: {log_data['template_id'].nunique()} unique templates")

    log_data['Timestamp'] = pd.to_datetime(log_data['Timestamp'], errors='coerce')
    log_data = log_data.dropna(subset=['Timestamp'])  # 删除无效时间戳
    log_data.set_index('Timestamp', inplace=True)

    frequency = log_data.groupby('template_id').resample('5s').size().unstack(level=0, fill_value=0)
    # print(f"Frequency data for {log_path}: {frequency.shape}")

    # 调整过滤条件，避免过于严格
    threshold = 0.85 * len(frequency)
    frequency = frequency.loc[:, (frequency == 0).sum(axis=0) < threshold]
    frequency = frequency.loc[~(frequency == 0).all(axis=1)]
    # print(f"Filtered frequency data for {log_path}: {frequency.shape}")

    if frequency.empty:
        print(f"Warning: Frequency data is empty for {log_path}")

    filename = Path(log_path).stem + '_frequency.csv'
    frequency.to_csv(os.path.join(output_dir, filename))

def process_multiple_days_logs(config, data_type):
    """处理多天的log数据"""
    data_config = config[f'{data_type}_data']
    base_input_folder = data_config['path']
    base_output_folder = os.path.join(config['processed_data_path'], data_type)
    start_date = min(data_config['dates'])
    end_date = max(data_config['dates'])
    
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date_dt:
        date_str = current_date.strftime('%Y-%m-%d')
        input_folder = os.path.join(base_input_folder, date_str, 'log')
        temp_output_folder = os.path.join('temp_processed_data', data_type, date_str, 'log_classification')
        final_output_folder = os.path.join(base_output_folder, date_str, 'log_template')
        
        if os.path.exists(input_folder):
            print(f"Processing {data_type} log data for {date_str}")
            
            # 第一步：按PodName分类
            sort_and_save_logs(input_folder, temp_output_folder)
            
            # 第二步：提取log模板
            os.makedirs(final_output_folder, exist_ok=True)
            for filename in os.listdir(temp_output_folder):
                if filename.endswith('.csv'):
                    log_path = os.path.join(temp_output_folder, filename)
                    process_log_file(log_path, final_output_folder)
            
            # 清理临时文件
            if os.path.exists(temp_output_folder):
                shutil.rmtree(temp_output_folder)
                print(f"Cleaned up temporary files for {date_str}")
        else:
            print(f"Warning: Log folder not found for {date_str}")
            
        current_date += timedelta(days=1)
    
    # 清理整个临时目录
    if os.path.exists('temp_processed_data'):
        shutil.rmtree('temp_processed_data')
        print("Cleaned up all temporary files")

def main():
    parser = argparse.ArgumentParser(description='Process log data for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (ob, tt, gaia, aiops)')
    
    args = parser.parse_args()
    
    # 获取数据集配置
    config = get_dataset_config(args.dataset)
    
    print("Processing normal data...")
    process_multiple_days_logs(config, 'normal')
    
    print("Processing abnormal data...")
    process_multiple_days_logs(config, 'abnormal')
    
    print(f"Data processing completed for {config['name']}")

if __name__ == "__main__":
    main()