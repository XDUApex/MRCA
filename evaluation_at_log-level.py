import pandas as pd
import json
import os
import re
import argparse
from config import get_dataset_config

DATASET = None

def extract_service_name(inject_pod=None, service_file=None):
    """提取服务名称，根据数据集类型处理"""
    global DATASET  # 使用全局变量

    name_source = inject_pod if inject_pod else service_file
    if not name_source:
        return None
    
    if DATASET == 'tt':
        if inject_pod:
            # Ground Truth 服务名称提取
            return ''.join(inject_pod.split('-')[:3])
        elif service_file:
            # 实验结果服务名称提取
            return ''.join(service_file.replace('_frequency.csv', '').split('_')[:3])
    elif DATASET == 'ob':
        if inject_pod:
            # Ground Truth 服务名称提取
            return inject_pod.split('_metric')[0].split('-')[0]
        elif service_file:
            # 实验结果服务名称提取
            return service_file.replace('_frequency.csv', '').split('_')[0]
    elif DATASET == 'gaia':
        # 新逻辑: 只清理文件名后缀，保留实例编号 (如 'mobservice1')
        return name_source.replace('.csv', '').replace('_frequency', '')
    elif DATASET == 'aiops':
        # 为 aiops 留空
        pass
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")
    
    return name_source.replace('.csv', '')

def load_ground_truth(config):
    """
    智能加载所有真值数据，并返回一个统一格式的DataFrame。
    """
    global DATASET
    all_ground_truth_dfs = []
    print(f"Loading ground truth for '{DATASET}' dataset...")

    if DATASET in ['ob', 'tt']:
        fault_files = config.get('fault_files', [])
        for gt_file in fault_files:
            if os.path.exists(gt_file):
                with open(gt_file, 'r') as f:
                    fault_data = json.load(f)
                gt_df = df_trans(fault_data) # 复用旧的转换函数
                all_ground_truth_dfs.append(gt_df)
    
    elif DATASET == 'gaia':
        fault_files = config.get('fault_files', [])
        for gt_file in fault_files:
            if os.path.exists(gt_file):
                try:
                    temp_df = pd.read_csv(gt_file)
                    # 检查关键列
                    if 'st_time' not in temp_df.columns or 'instance' not in temp_df.columns:
                        print(f"Warning: {gt_file} is missing 'st_time' or 'instance' columns. Skipping.")
                        continue
                    
                    # 转换列以匹配内部格式
                    temp_df['inject_time_minute'] = pd.to_datetime(temp_df['st_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                    # 新逻辑: 使用 'instance' 列作为正确的故障服务实例
                    temp_df['inject_pod'] = temp_df['instance'] 
                    temp_df.rename(columns={'anomaly_type': 'inject_type'}, inplace=True)
                    
                    # 筛选并组合
                    required_cols = ['inject_time_minute', 'inject_pod', 'inject_type']
                    final_cols = [col for col in required_cols if col in temp_df.columns]
                    all_ground_truth_dfs.append(temp_df[final_cols])
                except Exception as e:
                    print(f"Error processing ground truth file {gt_file}: {e}")

    elif DATASET == 'aiops':
        # 为 aiops 留空
        pass

    if not all_ground_truth_dfs:
        return pd.DataFrame()
    return pd.concat(all_ground_truth_dfs, ignore_index=True).dropna(subset=['inject_time_minute', 'inject_pod'])

def df_trans(fault_data):
    """将故障数据转换为DataFrame格式"""
    ground_truth_records = []
    for hour_faults in fault_data.values():
        for fault in hour_faults:
            inject_pod = fault['inject_pod']
            service_name = extract_service_name(inject_pod=inject_pod)
            
            inject_time_full = fault['inject_time']
            inject_time_minute = inject_time_full[:16]
            
            ground_truth_records.append({
                'inject_time_minute': inject_time_minute,
                'inject_pod': service_name,
                'inject_type': fault['inject_type']
            })
    return pd.DataFrame(ground_truth_records)

def extract_dates_from_anomaly_results(anomaly_score_folder):
    """从异常检测结果文件中提取日期"""
    dates = set()
    if not os.path.exists(anomaly_score_folder):
        return dates
    
    for item in os.listdir(anomaly_score_folder):
        item_path = os.path.join(anomaly_score_folder, item)
        if os.path.isdir(item_path) and re.match(r'\d{4}-\d{2}-\d{2}', item):
            dates.add(item)
    
    return sorted(dates)

def parse_anomaly_results(anomaly_score_folder, target_dates):
    """解析异常检测结果文件"""
    experiment_results = []
    
    if not os.path.exists(anomaly_score_folder):
        return pd.DataFrame()
    
    for date in target_dates:
        date_folder = os.path.join(anomaly_score_folder, date)
        if not os.path.exists(date_folder):
            print(f"Warning: No results folder found for date {date}")
            continue
            
        files = [f for f in os.listdir(date_folder) if f.startswith('ranked_services_') and f.endswith('.csv')]
        
        for filename in files:
            file_path = os.path.join(date_folder, filename)
            parse_single_result_file(file_path, filename, date, experiment_results)
    
    return pd.DataFrame(experiment_results)

def parse_single_result_file(file_path, filename, date, experiment_results):
    """解析单个结果文件"""
    time_match = re.search(r'ranked_services_(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})\.csv', filename)
    
    if time_match:
        date_part = time_match.group(1)
        hour = time_match.group(2)
        minute = time_match.group(3)
        time_str_minute = f"{date_part} {hour}:{minute}"
    else:
        return
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) >= 2:
                service_file = parts[0]
                mse_score = float(parts[1])
                service_name = extract_service_name(service_file=service_file)
                
                experiment_results.append({
                    'InjectionTime_minute': time_str_minute,
                    'Date': date,
                    'ServiceName': service_name,
                    'MSEScore': mse_score,
                    'Rank': idx + 1
                })
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")

def evaluation_stage1(experiment_result, ground_truth_df, k_values):
    """评估第一阶段异常检测结果"""
    def calculate_pr_at_k(experiment_results, ground_truth_df, k):
        match_counts = 0
        total_times = len(ground_truth_df['inject_time_minute'].unique())
        
        for inject_time_minute in ground_truth_df['inject_time_minute'].unique():
            gt_entries = ground_truth_df[ground_truth_df['inject_time_minute'] == inject_time_minute]
            gt_services = gt_entries['inject_pod'].tolist()
            
            exp_results_at_minute = experiment_results[experiment_results['InjectionTime_minute'] == inject_time_minute]
            
            if not exp_results_at_minute.empty:
                top_k_results = exp_results_at_minute[exp_results_at_minute['Rank'] <= k]
                top_k_services = top_k_results['ServiceName'].tolist()
                
                # print(f"Time: {inject_time_minute}, Ground Truth: {gt_services}, Top K Services: {top_k_services}")
                
                if any(service in gt_services for service in top_k_services):
                    match_counts += 1
        
        return match_counts, total_times
    
    pr_results = {}
    
    for k in k_values:
        pr_counts, total = calculate_pr_at_k(experiment_result, ground_truth_df, k)
        pr_results[k] = (pr_counts, total)
    
    return pr_results

def calculate_pr_stage1_by_dataset(dataset_config):
    """按数据集计算第一阶段的PR指标"""
    anomaly_score_folder = dataset_config['anomaly_output_path']
    available_dates = extract_dates_from_anomaly_results(anomaly_score_folder)
    
    if not available_dates:
        print("No anomaly detection results found!")
        return
    
    print(f"Found dates: {len(available_dates)} days")
    print(f"Date range: {min(available_dates)} to {max(available_dates)}")
    
    eval_params = dataset_config['evaluation_params']
    k_values = eval_params['k_values']
    show_individual = eval_params['show_individual_results']
    max_individual_display = eval_params['max_individual_display']
    
    fault_files = dataset_config['fault_files']
    available_fault_files = [ff for ff in fault_files if any(date in ff for date in available_dates) and os.path.exists(ff)]
    
    if not available_fault_files:
        print("No corresponding ground truth files found!")
        return
    
    print(f"Found {len(available_fault_files)} fault files")
    
    experiment_results = parse_anomaly_results(anomaly_score_folder, available_dates)
    
    if experiment_results.empty:
        print("No experiment results found!")
        return
    
    print(f"Loaded {len(experiment_results)} experiment results")
    
    all_ground_truth = load_ground_truth(dataset_config)
    
    if all_ground_truth.empty:
        print("No ground truth data could be loaded!")
        return
    print(f"Loaded {len(all_ground_truth)} ground truth records in total.")
    
    overall_pr_results = evaluation_stage1(experiment_results, all_ground_truth, k_values)
    
    print('=' * 60)
    print(f"{dataset_config['name']} - Stage 1 Anomaly Detection Evaluation Results")
    print('=' * 60)
    
    print("Overall Results:")
    for k in k_values:
        counts, total = overall_pr_results[k]
        pr_rate = counts / total if total > 0 else 0
        print(f"  PR@{k}: {pr_rate:.2%} ({counts}/{total})")
    
    print('=' * 60)
    
    if show_individual and max_individual_display > 0:
        print("Sample Results by Date:")
        display_count = min(max_individual_display, len(available_dates))
        
        for date in available_dates[:display_count]:
            exp_results_date = experiment_results[experiment_results['Date'] == date]
            gt_file = next((ff for ff in available_fault_files if date in ff), None)
            
            if gt_file:
                with open(gt_file, 'r') as f:
                    fault_data = json.load(f)
                gt_df = df_trans(fault_data)
                
                if not exp_results_date.empty and not gt_df.empty:
                    dataset_pr_results = evaluation_stage1(exp_results_date, gt_df, k_values)
                    
                    print(f"{dataset_config['name']} - {date}:")
                    for k in k_values:
                        counts, total = dataset_pr_results[k]
                        pr_rate = counts / total if total > 0 else 0
                        print(f"  PR@{k}: {pr_rate:.2%} ({counts}/{total})")
                    print("-" * 40)
        
        if len(available_dates) > display_count:
            print(f"... and {len(available_dates) - display_count} more dates")
    
    print("\nEvaluation Configuration:")
    print(f"  K values: {k_values}")
    print(f"  Show individual results: {show_individual}")
    print(f"  Max individual display: {max_individual_display}")

def main():
    global DATASET  # 使用全局变量
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection results for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (ob, tt, gaia, aiops)')
    
    args = parser.parse_args()
    DATASET = args.dataset  # 初始化全局变量

    config = get_dataset_config(args.dataset)
    
    print(f"Evaluating {config['name']}...")
    calculate_pr_stage1_by_dataset(config)

if __name__ == "__main__":
    main()