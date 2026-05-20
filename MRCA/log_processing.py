import os
import json
import pandas as pd
import argparse
from drain3 import TemplateMiner
from pathlib import Path
import shutil
from config import get_dataset_config
from MRCA.rq2_profiling import RQ2Profiler

DATASET = None

def sort_and_save_logs(input_folder, output_folder, profiler=None):
    """处理单个文件夹的log文件"""
    os.makedirs(output_folder, exist_ok=True)
    all_data = pd.DataFrame()

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            csv_path = os.path.join(input_folder, filename)
            data = pd.read_csv(csv_path, on_bad_lines="skip")
            if profiler is not None:
                profiler.add_input_file(csv_path, len(data))
            all_data = pd.concat([all_data, data], ignore_index=True)

    global DATASET
    if DATASET == 'gaia' and 'service' in all_data.columns:
        all_data.rename(columns={'service': 'PodName'}, inplace=True)
    if DATASET == 'aiops' and 'service' in all_data.columns:
        all_data.rename(columns={'service': 'PodName'}, inplace=True)

    if DATASET == 'gaia' and 'timestamp' in all_data.columns:
        all_data.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
    if DATASET == 'aiops' and 'timestamp' in all_data.columns:
        all_data.rename(columns={'timestamp': 'Timestamp'}, inplace=True)

    if DATASET == 'gaia' and 'message' in all_data.columns:
        all_data.rename(columns={'message': 'Log'}, inplace=True)
    if DATASET == 'aiops' and 'message' in all_data.columns:
        all_data.rename(columns={'message': 'Log'}, inplace=True)

    if 'PodName' not in all_data.columns:
        print(f"Warning: 'PodName' column not found in data from {input_folder}. Cannot process logs.")
        return

    if 'Timestamp' in all_data.columns:
        if pd.api.types.is_numeric_dtype(all_data['Timestamp']):
             all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], unit='s', errors='coerce')
        else:
             all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], errors='coerce')
        all_data = all_data.dropna(subset=['Timestamp'])

    for pod_name, group in all_data.groupby('PodName'):
        safe_filename = f"{pod_name.replace('-', '_')}.csv"
        file_path = os.path.join(output_folder, safe_filename)
        output_columns = ['Timestamp', 'PodName', 'Log']
        final_group = group[[col for col in output_columns if col in group.columns]]
        final_group.to_csv(file_path, index=False)

def parse_log(log):
    global DATASET
    # ob/tt 数据集使用JSON解析
    if DATASET in ['ob', 'tt']:
        try:
            outer_json = json.loads(log)
            if isinstance(outer_json['log'], str):
                # 兼容单层JSON包装的日志
                try:
                    inner_json = json.loads(outer_json['log'])
                    return inner_json.get('message', outer_json['log'])
                except json.JSONDecodeError:
                    return outer_json['log']
            else:
                # 兼容旧的双层JSON格式
                inner_json = json.loads(outer_json['log'])
                return inner_json['message']
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # print(f"JSON Decode Error or other parsing error in log: {log}")
            return str(log) # 返回原始字符串作为后备
            
    # gaia 数据集直接返回消息
    elif DATASET == 'gaia':
        return str(log)
        
    # aiops 或其他未来数据集的逻辑
    elif DATASET == 'aiops':
        # 此处为 aiops 留空，暂时返回原始字符串
        return str(log)
        
    else:
        # 默认行为
        return str(log)

def process_log_file(log_path, output_dir, profiler=None):
    """处理单个log文件"""
    log_data = pd.read_csv(log_path)
    log_data['Log'] = log_data['Log'].apply(parse_log)

    log_data = log_data.dropna(subset=['Log'])

    template_miner = TemplateMiner()
    log_data['template_id'] = log_data['Log'].apply(lambda log_message: template_miner.add_log_message(log_message)['cluster_id'])

    if profiler is not None:
        profiler.add_log_template_count(int(log_data['template_id'].nunique()))

    log_data['Timestamp'] = pd.to_datetime(log_data['Timestamp'], errors='coerce')
    log_data = log_data.dropna(subset=['Timestamp'])
    log_data.set_index('Timestamp', inplace=True)

    frequency = log_data.groupby('template_id').resample('5s').size().unstack(level=0, fill_value=0)

    threshold = 0.95 * len(frequency)
    frequency = frequency.loc[:, (frequency == 0).sum(axis=0) < threshold]
    frequency = frequency.loc[~(frequency == 0).all(axis=1)]

    if frequency.empty:
        print(f"Warning: Frequency data is empty for {log_path}")

    filename = Path(log_path).stem + '_frequency.csv'
    frequency.to_csv(os.path.join(output_dir, filename))

def process_multiple_days_logs(config, data_type, profiler=None):
    """处理多天的log数据"""
    data_config = config[f'{data_type}_data']
    base_input_folder = data_config['path']
    base_output_folder = os.path.join(config['processed_data_path'], data_type)
    dates_to_process = data_config['dates']

    for date_str in dates_to_process:
        input_folder = os.path.join(base_input_folder, date_str, 'log')
        temp_output_folder = os.path.join('temp_processed_data', data_type, date_str, 'log_classification')
        final_output_folder = os.path.join(base_output_folder, date_str, 'log_template')

        if os.path.exists(input_folder):
            print(f"Processing {data_type} log data for {date_str}")

            sort_and_save_logs(input_folder, temp_output_folder, profiler=profiler)

            os.makedirs(final_output_folder, exist_ok=True)
            for filename in os.listdir(temp_output_folder):
                if filename.endswith('.csv'):
                    log_path = os.path.join(temp_output_folder, filename)
                    process_log_file(log_path, final_output_folder, profiler=profiler)

            if os.path.exists(temp_output_folder):
                shutil.rmtree(temp_output_folder)
                print(f"Cleaned up temporary files for {date_str}")
        else:
            print(f"Warning: Log folder not found for {date_str}")

    if os.path.exists('temp_processed_data'):
        shutil.rmtree('temp_processed_data')
        print("Cleaned up all temporary files")

def main():
    parser = argparse.ArgumentParser(description='Process log data for specified dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (ob, tt, gaia, aiops)')
    parser.add_argument('--experiment', type=str, default='rq1', choices=['rq1', 'rq3'],
                        help='Experiment namespace for path layout')
    parser.add_argument('--variant', type=str, default='base',
                        help='Scenario suffix to avoid overwriting results, e.g. base/1/2/3')

    args = parser.parse_args()

    global DATASET
    DATASET = args.dataset

    config = get_dataset_config(args.dataset, experiment=args.experiment, variant=args.variant)

    profiler = RQ2Profiler(
        dataset=args.dataset,
        script_name='log_processing',
        stage='preprocess',
        modality='L',
        experiment=args.experiment,
        variant=args.variant,
    )

    print(f"Experiment: {args.experiment}, Variant: {args.variant}")

    with profiler.phase('preprocess'):
        print("Processing normal data...")
        process_multiple_days_logs(config, 'normal', profiler=profiler)

        print("Processing abnormal data...")
        process_multiple_days_logs(config, 'abnormal', profiler=profiler)

    artifact = profiler.write_json()
    print(f"[RQ2] Profiling artifact saved: {artifact}")

    print(f"Data processing completed for {config['name']}")

if __name__ == "__main__":
    main()