import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
from config import get_dataset_config

def process_files(input_folder1, input_folder2, output_folder):
    """聚合单天的数据"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    trace_files = [f for f in os.listdir(input_folder2) if f.endswith('.csv')] if os.path.exists(input_folder2) else []

    for filename1 in os.listdir(input_folder1):
        if filename1.endswith('.csv'):
            file1 = os.path.join(input_folder1, filename1)
            output_file = os.path.join(output_folder, filename1)
            
            # 读取日志文件
            if os.path.exists(file1):
                data_1 = pd.read_csv(file1)
                if 'Timestamp' in data_1.columns:
                    data_1['Timestamp'] = pd.to_datetime(data_1['Timestamp'], errors='coerce').astype(int) / 10 ** 9
                else:
                    print(f"Warning: No 'Timestamp' column in {filename1}. Using empty log data.")
                    data_1 = pd.DataFrame(columns=['Timestamp'])
            else:
                print(f"Warning: Log file {filename1} does not exist. Using empty log data.")
                data_1 = pd.DataFrame(columns=['Timestamp'])

            # 检查日志数据是否为空
            if data_1.empty or data_1['Timestamp'].isnull().all():
                print(f"Warning: Log data for {filename1} is empty or invalid. Using trace data only.")
                data_1 = pd.DataFrame(columns=['Timestamp'])

            # 匹配 trace 文件
            matching_trace = [f for f in trace_files if filename1[:7] == f[:7]]
            
            if matching_trace:
                file2 = os.path.join(input_folder2, matching_trace[0])
                data_2 = pd.read_csv(file2)
                if 'StartTimeUnixNano' in data_2.columns:
                    data_2['StartTimeUnixNano'] = pd.to_datetime(data_2['StartTimeUnixNano'].astype(str).str[:10].astype(int), unit='s', errors='coerce')
                else:
                    print(f"Warning: No 'StartTimeUnixNano' column in {matching_trace[0]}. Using empty trace data.")
                    data_2 = pd.DataFrame(columns=['StartTimeUnixNano', 'Duration'])
            else:
                print(f"Warning: Trace file for {filename1} does not exist. Using log data only.")
                data_2 = pd.DataFrame(columns=['StartTimeUnixNano', 'Duration'])

            # 检查 trace 数据是否为空
            if data_2.empty or data_2['StartTimeUnixNano'].isnull().all():
                print(f"Warning: Trace data for {filename1} is empty or invalid. Using log data only.")
                data_2 = pd.DataFrame(columns=['StartTimeUnixNano', 'Duration'])

            # 聚合特征
            if not data_1.empty and not data_2.empty:
                data_1['Duration'] = 0
                for i, row in data_2.iterrows():
                    if pd.notna(row['StartTimeUnixNano']):
                        closest_idx = (data_1['Timestamp'] - row['StartTimeUnixNano'].timestamp()).abs().idxmin()
                        data_1.at[closest_idx, 'Duration'] = row['Duration']
            elif not data_1.empty:
                data_1['Duration'] = 0
            elif not data_2.empty:
                data_1 = data_2.rename(columns={'StartTimeUnixNano': 'Timestamp'})
            else:
                print(f"Warning: Both log and trace data are empty for {filename1}. Skipping...")
                continue

            # 保存聚合结果
            new_columns = ['Timestamp'] + [str(i) for i in range(1, len(data_1.columns))]
            data_1.columns = new_columns
            data_1.to_csv(output_file, index=False)
            print(f"Aggregated data saved to {output_file}")

def process_files_multiple_days(base_input_folder1, base_input_folder2, base_output_folder, start_date, end_date):
    """处理多天的数据聚合"""
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date_dt:
        date_str = current_date.strftime('%Y-%m-%d')
        
        input_folder1 = os.path.join(base_input_folder1, date_str, 'log_template')
        input_folder2 = os.path.join(base_input_folder2, date_str, 'trace_latency')
        output_folder = os.path.join(base_output_folder, date_str, 'aggregation')
        
        if os.path.exists(input_folder1) or os.path.exists(input_folder2):
            print(f"Processing aggregation for {date_str}")
            process_files(input_folder1, input_folder2, output_folder)
        else:
            print(f"Warning: No data found for {date_str}. Skipping...")
            
        current_date += timedelta(days=1)

def main():
    parser = argparse.ArgumentParser(description='Aggregate data for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (ob, tt)')
    
    args = parser.parse_args()
    
    # 获取数据集配置
    config = get_dataset_config(args.dataset)
    
    print(f"Aggregating data for {config['name']}...")
    
    # 聚合正常数据
    print("Step 1: Aggregating normal data...")
    process_files_multiple_days(
        base_input_folder1=os.path.join(config['processed_data_path'], 'normal'),
        base_input_folder2=os.path.join(config['processed_data_path'], 'normal'),
        base_output_folder=os.path.join(config['processed_data_path'], 'normal'),
        start_date=min(config['normal_data']['dates']),
        end_date=max(config['normal_data']['dates'])
    )
    
    # 聚合异常数据
    print("Step 2: Aggregating abnormal data...")
    process_files_multiple_days(
        base_input_folder1=os.path.join(config['processed_data_path'], 'abnormal'),
        base_input_folder2=os.path.join(config['processed_data_path'], 'abnormal'),
        base_output_folder=os.path.join(config['processed_data_path'], 'abnormal'),
        start_date=min(config['abnormal_data']['dates']),
        end_date=max(config['abnormal_data']['dates'])
    )
    
    print(f"Data aggregation completed for {config['name']}")

if __name__ == "__main__":
    main()