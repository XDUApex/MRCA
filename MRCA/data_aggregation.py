import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
from config import get_dataset_config
import shutil

# --- 关键修正 1：函数定义增加 dataset_name 参数 ---
def process_files(input_folder1, input_folder2, output_folder, dataset_name):
    """聚合单天的数据"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    trace_files = [f for f in os.listdir(input_folder2) if f.endswith('.csv')] if os.path.exists(input_folder2) else []

    if not os.path.exists(input_folder1):
        print(f"警告: 日志模板文件夹未找到: {input_folder1}。将只处理仅有追踪文件的服务。")
        log_files_to_iterate = [tf.replace('.csv', '_frequency.csv') for tf in trace_files]
    else:
        log_files_to_iterate = os.listdir(input_folder1)

    for filename1 in log_files_to_iterate:
        if not filename1.endswith('.csv'):
            continue

        file1 = os.path.join(input_folder1, filename1)
        output_file = os.path.join(output_folder, filename1)
        
        # --- 1. 读取日志文件 (data_1) ---
        if os.path.exists(file1):
            data_1 = pd.read_csv(file1)
            if not data_1.empty:
                data_1['Timestamp'] = pd.to_datetime(data_1['Timestamp'], errors='coerce').astype(int) / 10**9
        else:
            data_1 = pd.DataFrame() 

        # --- 2. 读取追踪文件 (data_2) ---
        # --- 关键修正 2：根据数据集名称选择匹配逻辑 ---
        if dataset_name in ['gaia', 'aiops']:
            # 对于 gaia 和 aiops, 使用新的、精确的逻辑
            service_name_part = filename1.replace('_frequency.csv', '')
            matching_trace = [f for f in trace_files if f.replace('.csv', '') == service_name_part]
        else:
            # 对于 ob 和 tt, 保持旧的、兼容的逻辑
            matching_trace = [f for f in trace_files if filename1[:7] == f[:7]]
        
        if matching_trace:
            file2 = os.path.join(input_folder2, matching_trace[0])
            data_2 = pd.read_csv(file2)
            if 'StartTimeUnixNano' in data_2.columns and not data_2.empty:
                data_2['StartTimeUnixNano'] = pd.to_datetime(data_2['StartTimeUnixNano'].astype(str), unit='ns', errors='coerce')
            else:
                data_2 = pd.DataFrame()
        else:
            data_2 = pd.DataFrame()

        # --- 3. 核心聚合逻辑 (此部分无需修改) ---
        if data_1.empty and data_2.empty:
            continue

        agg_data = pd.DataFrame()

        if not data_1.empty:
            agg_data = data_1.copy()
            if not data_2.empty:
                agg_data['Duration'] = 0
                for _, row in data_2.iterrows():
                    if pd.notna(row['StartTimeUnixNano']):
                        closest_idx = (agg_data['Timestamp'] - row['StartTimeUnixNano'].timestamp()).abs().idxmin()
                        agg_data.at[closest_idx, 'Duration'] = row['Duration']
            else:
                agg_data['Duration'] = 0
        
        elif not data_2.empty:
            agg_data = data_2.rename(columns={'StartTimeUnixNano': 'Timestamp', 'Duration': 'TraceLatency'})
            agg_data.insert(1, 'LogPlaceholder', 0)

        # --- 4. 标准化列名并保存 (此部分无需修改) ---
        if agg_data.empty:
            continue
            
        agg_data.rename(columns={agg_data.columns[-1]: 'Duration'}, inplace=True)

        num_log_cols = len(agg_data.columns) - 2
        new_columns = ['Timestamp'] + [str(i) for i in range(1, num_log_cols + 1)] + ['Duration']
        agg_data.columns = new_columns

        agg_data.to_csv(output_file, index=False)
        print(f"已聚合数据并保存到: {output_file}")

# --- 关键修正 3：函数定义增加 dataset_name 参数 ---
def process_files_multiple_days(base_input_folder1, base_input_folder2, base_output_folder, dates_to_process, dataset_name):
    """处理多天的数据聚合"""
    for date_str in dates_to_process:
        input_folder1 = os.path.join(base_input_folder1, date_str, 'log_template')
        input_folder2 = os.path.join(base_input_folder2, date_str, 'trace_latency')
        output_folder = os.path.join(base_output_folder, date_str, 'aggregation')
        
        if os.path.exists(output_folder):
            print(f"正在清理旧的聚合文件: {output_folder}")
            shutil.rmtree(output_folder)
        
        if os.path.exists(input_folder1) or os.path.exists(input_folder2):
            print(f"正在为 {date_str} 进行数据聚合...")
            # --- 关键修正 4：将 dataset_name 传递下去 ---
            process_files(input_folder1, input_folder2, output_folder, dataset_name)
        else:
            print(f"警告: 未找到 {date_str} 的数据，已跳过。")


def main():
    parser = argparse.ArgumentParser(description='Aggregate data for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (ob, tt, gaia, aiops)')
    
    args = parser.parse_args()
    
    config = get_dataset_config(args.dataset)
    
    print(f"正在为 {config['name']} 进行数据聚合...")
    
    # --- 关键修正 5：在调用时传入 dataset_name ---
    print("步骤 1: 聚合正常数据...")
    process_files_multiple_days(
        base_input_folder1=os.path.join(config['processed_data_path'], 'normal'),
        base_input_folder2=os.path.join(config['processed_data_path'], 'normal'),
        base_output_folder=os.path.join(config['processed_data_path'], 'normal'),
        dates_to_process=config['normal_data']['dates'],
        dataset_name=args.dataset
    )
    
    print("步骤 2: 聚合异常数据...")
    process_files_multiple_days(
        base_input_folder1=os.path.join(config['processed_data_path'], 'abnormal'),
        base_input_folder2=os.path.join(config['processed_data_path'], 'abnormal'),
        base_output_folder=os.path.join(config['processed_data_path'], 'abnormal'),
        dates_to_process=config['abnormal_data']['dates'],
        dataset_name=args.dataset
    )
    
    print(f"{config['name']} 的数据聚合已完成。")

if __name__ == "__main__":
    main()