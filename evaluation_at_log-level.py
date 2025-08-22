import pandas as pd
import json
import os
import re
import argparse
from config import get_dataset_config

DATASET = None

def get_base_service_name(service_instance_name):
    """
    从服务实例名中提取基础服务名。
    针对不同数据集有不同的提取策略。
    """
    if not isinstance(service_instance_name, str):
        return ''

    if DATASET == 'tt':
        # 规则: 提取服务名部分并压缩
        # e.g., "ts-contacts-service-..." -> "tscontactsservice"
        parts = re.split('[-_]', service_instance_name)
        service_parts = []
        for part in parts:
            if part.isalpha():
                service_parts.append(part)
            else:
                break
        return "".join(service_parts)

    elif DATASET == 'ob':
        # 规则: 取第一个分隔符(下划线或连字符)之前的部分
        # e.g., "adservice_5f6585d649_fnmft" -> "adservice"
        return re.split('[-_]', service_instance_name)[0]

    elif DATASET == 'aiops':
        # aiops 的逻辑保持不变
        base_name = re.sub(r'_\d+$', '', service_instance_name)
        return re.sub(r'\d+$', '', base_name)

    else: # gaia 和其他默认情况
        # 默认逻辑: 移除末尾的数字
        # e.g., "mobservice1" -> "mobservice"
        return re.sub(r'\d+$', '', service_instance_name)


def extract_service_name(inject_pod=None, service_file=None):
    """提取完整的服务实例名"""
    name_source = inject_pod if inject_pod else service_file
    if not name_source:
        return None
    return name_source.replace('.csv', '').replace('_frequency', '').replace('_metric', '')

def load_ground_truth(config):
    """
    智能加载所有真值数据，并返回一个统一格式的DataFrame。
    """
    global DATASET
    all_ground_truth_dfs = []
    # This function is called for each modality, but the ground truth is the same.
    # The print statement will appear multiple times, which is acceptable.
    print(f"正在加载 '{DATASET}' 数据集的真值数据...")

    if DATASET in ['ob', 'tt']:
        fault_files = config.get('fault_files', [])
        for gt_file in fault_files:
            if os.path.exists(gt_file):
                with open(gt_file, 'r') as f:
                    fault_data = json.load(f)
                gt_df = df_trans(fault_data)
                all_ground_truth_dfs.append(gt_df)
    
    elif DATASET in ['gaia', 'aiops']:
        fault_files = config.get('fault_files', [])
        for gt_file in fault_files:
            if os.path.exists(gt_file):
                try:
                    temp_df = pd.read_csv(gt_file)
                    if 'st_time' not in temp_df.columns or 'instance' not in temp_df.columns:
                        print(f"警告: 文件 {gt_file} 缺少 'st_time' 或 'instance' 列，已跳过。")
                        continue
                    
                    temp_df['inject_time_minute'] = pd.to_datetime(temp_df['st_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                    
                    instance_names = temp_df['instance'].str.replace('-', '_')
                    temp_df['inject_pod_instance'] = instance_names
                    temp_df['inject_pod'] = instance_names.apply(get_base_service_name)
                    
                    temp_df.rename(columns={'anomaly_type': 'inject_type'}, inplace=True)
                    
                    required_cols = ['inject_time_minute', 'inject_pod', 'inject_pod_instance', 'inject_type']
                    final_cols = [col for col in required_cols if col in temp_df.columns]
                    all_ground_truth_dfs.append(temp_df[final_cols])
                except Exception as e:
                    print(f"处理真值文件 {gt_file} 时出错: {e}")

    if not all_ground_truth_dfs:
        return pd.DataFrame()
    return pd.concat(all_ground_truth_dfs, ignore_index=True).dropna(subset=['inject_time_minute', 'inject_pod'])

def df_trans(fault_data):
    """将故障数据转换为DataFrame格式 (适用于 ob/tt)"""
    ground_truth_records = []
    for hour_faults in fault_data.values():
        for fault in hour_faults:
            inject_pod_raw = fault['inject_pod']
            service_instance_name = extract_service_name(inject_pod=inject_pod_raw)
            base_service_name = get_base_service_name(service_instance_name)
            
            inject_time_full = fault['inject_time']
            inject_time_minute = inject_time_full[:16]
            
            ground_truth_records.append({
                'inject_time_minute': inject_time_minute,
                'inject_pod_instance': service_instance_name,
                'inject_pod': base_service_name,
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
    
    return sorted(list(dates))

def parse_anomaly_results(anomaly_score_folder, target_dates):
    """解析异常检测结果文件"""
    experiment_results = []
    
    if not os.path.exists(anomaly_score_folder):
        return pd.DataFrame()
    
    for date in target_dates:
        date_folder = os.path.join(anomaly_score_folder, date)
        if not os.path.exists(date_folder):
            # This is now an expected warning if a date exists for one modality but not another
            # print(f"警告: 未找到日期 {date} 的结果文件夹")
            continue
            
        files = [f for f in os.listdir(date_folder) if f.startswith('ranked_services_') and f.endswith('.csv')]
        
        for filename in files:
            file_path = os.path.join(date_folder, filename)
            parse_single_result_file(file_path, filename, date, experiment_results)
    
    return pd.DataFrame(experiment_results)

def parse_single_result_file(file_path, filename, date, experiment_results):
    """解析单个结果文件，同时记录基础名和实例名"""
    time_match = re.search(r'ranked_services_(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})\.csv', filename)
    
    if not time_match:
        return

    time_str_minute = f"{time_match.group(1)} {time_match.group(2)}:{time_match.group(3)}"

    if os.path.getsize(file_path) == 0:
        experiment_results.append({'InjectionTime_minute': time_str_minute, 'Date': date, 'IsEmpty': True})
        return
    
    try:
        df = pd.read_csv(file_path, header=None)
        if df.empty:
            return
            
        df.columns = ['ServiceFile', 'MSEScore']
        
        for idx, row in df.iterrows():
            service_file = row['ServiceFile']
            mse_score = float(row['MSEScore'])
            
            service_instance_name = extract_service_name(service_file=service_file)
            base_service_name = get_base_service_name(service_instance_name)
            
            experiment_results.append({
                'InjectionTime_minute': time_str_minute,
                'Date': date,
                'ServiceName': base_service_name,
                'ServiceInstanceName': service_instance_name,
                'MSEScore': mse_score,
                'Rank': idx + 1,
                'IsEmpty': False
            })
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")

def evaluation_stage1_standard(experiment_result, ground_truth_df, k_values):
    def calculate_pr_at_k(experiment_results, ground_truth_df, k):
        match_counts = 0
        unique_fault_times = ground_truth_df['inject_time_minute'].unique()
        total_times = len(unique_fault_times)
        if total_times == 0: return 0, 0
            
        for inject_time_minute in unique_fault_times:
            gt_base_services = set(ground_truth_df[ground_truth_df['inject_time_minute'] == inject_time_minute]['inject_pod'].unique())
            exp_results_at_minute = experiment_results[experiment_results['InjectionTime_minute'] == inject_time_minute]
            if exp_results_at_minute.empty or exp_results_at_minute.iloc[0]['IsEmpty']: continue
            
            top_k_base_services = set(exp_results_at_minute[exp_results_at_minute['Rank'] <= k]['ServiceName'].unique())
            
            if not gt_base_services.isdisjoint(top_k_base_services):
                match_counts += 1
        return match_counts, total_times
    
    pr_results = {}
    for k in k_values:
        pr_counts, total = calculate_pr_at_k(experiment_result, ground_truth_df, k)
        pr_results[k] = (pr_counts, total)
    return pr_results

def analyze_failures_standard(experiment_result, ground_truth_df, max_k, max_failures_to_print=5):
    stats = { 'total': 0, 'success': 0, 'missing_data': 0, 'not_in_list': 0, 'low_rank': 0 }
    failures_printed = 0
    unique_fault_times = ground_truth_df['inject_time_minute'].unique()
    stats['total'] = len(unique_fault_times)

    for inject_time_minute in unique_fault_times:
        exp_results_at_minute = experiment_result[experiment_result['InjectionTime_minute'] == inject_time_minute]
        gt_base_services = set(ground_truth_df[ground_truth_df['inject_time_minute'] == inject_time_minute]['inject_pod'].unique())
        
        if exp_results_at_minute.empty or exp_results_at_minute.iloc[0]['IsEmpty']:
            stats['missing_data'] += 1
            if failures_printed < max_failures_to_print:
                print(f"  - 失败案例 (数据缺失): 时间 {inject_time_minute}, 预期根因: {list(gt_base_services)}")
                failures_printed += 1
            continue

        all_ranked_base_services = set(exp_results_at_minute['ServiceName'].unique())
        top_k_base_services = set(exp_results_at_minute[exp_results_at_minute['Rank'] <= max_k]['ServiceName'].unique())
        
        is_present = not gt_base_services.isdisjoint(all_ranked_base_services)
        is_in_top_k = not gt_base_services.isdisjoint(top_k_base_services)
        
        if is_in_top_k: stats['success'] += 1
        elif is_present: stats['low_rank'] += 1
        else: stats['not_in_list'] += 1
    return stats

def evaluation_stage1_aiops(experiment_result, ground_truth_df, all_gt_base_services, k_values):
    def calculate_pr_at_k_new(processed_rankings, k):
        match_counts = 0
        total_times = len(processed_rankings)
        if total_times == 0: return 0, 0
        for time_key, data in processed_rankings.items():
            if not set(data['ranked_list'][:k]).isdisjoint(data['gt_services']):
                match_counts += 1
        return match_counts, total_times

    processed_rankings = {}
    unique_fault_times = ground_truth_df['inject_time_minute'].unique()

    for inject_time_minute in unique_fault_times:
        gt_services = ground_truth_df[ground_truth_df['inject_time_minute'] == inject_time_minute]['inject_pod'].unique().tolist()
        exp_results_at_minute = experiment_result[experiment_result['InjectionTime_minute'] == inject_time_minute]
        final_ranked_list = []
        if not exp_results_at_minute.empty and not exp_results_at_minute.iloc[0]['IsEmpty']:
            agg_ranks = exp_results_at_minute.loc[exp_results_at_minute.groupby('ServiceName')['Rank'].idxmin()].sort_values('Rank')
            final_ranked_list = agg_ranks['ServiceName'].tolist()
        missing_in_this_rank = [s for s in all_gt_base_services if s not in final_ranked_list]
        final_ranked_list.extend(missing_in_this_rank)
        processed_rankings[inject_time_minute] = {'ranked_list': final_ranked_list, 'gt_services': gt_services}

    pr_results = {}
    for k in k_values:
        pr_counts, total = calculate_pr_at_k_new(processed_rankings, k)
        pr_results[k] = (pr_counts, total)
    return pr_results, processed_rankings

def analyze_failures_aiops(processed_rankings, max_k, max_failures_to_print=5):
    stats = { 'total': 0, 'success': 0, 'low_rank': 0 }
    failures_printed = 0
    stats['total'] = len(processed_rankings)

    for time_key, data in processed_rankings.items():
        if not set(data['ranked_list'][:max_k]).isdisjoint(data['gt_services']):
            stats['success'] += 1
        else:
            stats['low_rank'] += 1
            if failures_printed < max_failures_to_print:
                actual_rank = "N/A"
                for i, service in enumerate(data['ranked_list']):
                    if service in data['gt_services']: actual_rank = i + 1; break
                print(f"  - 失败案例 (模型性能): 时间 {time_key}, 预期根因: {data['gt_services']}, 实际排名: {actual_rank}")
                failures_printed += 1
    return stats

def calculate_pr_stage1_by_dataset(dataset_config):
    anomaly_score_folder = dataset_config['anomaly_output_path']
    available_dates = extract_dates_from_anomaly_results(anomaly_score_folder)
    if not available_dates:
        print(f"未在指定目录中找到异常检测结果！路径: {anomaly_score_folder}")
        return
    
    print(f"发现日期: {len(available_dates)} 天, 日期范围: {min(available_dates)} 到 {max(available_dates)}")
    
    experiment_results = parse_anomaly_results(anomaly_score_folder, available_dates)
    if experiment_results.empty:
        print("未能从文件中解析出有效的实验结果！")
        return
    
    experiment_results.fillna({'ServiceName': '', 'ServiceInstanceName': ''}, inplace=True)
    print(f"已加载 {len(experiment_results)} 条排名服务条目。")
    
    all_ground_truth = load_ground_truth(dataset_config)
    if all_ground_truth.empty:
        print("未能加载任何真值数据！")
        return
    print(f"总共加载 {len(all_ground_truth)} 条真值记录。")
    
    print('=' * 60)

    if DATASET == 'aiops':
        print("正在为 AIOps 数据集应用“聚合排名+补全”评估策略...")
        all_base_services_from_config = dataset_config.get('all_base_services')
        if not all_base_services_from_config:
            print("错误: config.py 中未找到 'all_base_services' 列表，无法使用新策略。")
            return
            
        all_base_services = sorted(list(all_base_services_from_config))
        k_values = list(range(1, len(all_base_services) + 1))
        
        overall_pr_results, processed_rankings = evaluation_stage1_aiops(experiment_results, all_ground_truth, all_base_services, k_values)
        
        print(f"{dataset_config['name']} - 第1阶段 异常检测评估结果 (新策略)")
        print('=' * 60)
        
        print("总体结果:")
        for k in k_values:
            counts, total = overall_pr_results[k]
            pr_rate = counts / total if total > 0 else 0
            print(f"  PR@{k}: {pr_rate:.2%} ({counts}/{total})")
        
        print('=' * 60)
        
        if k_values:
            max_k = max(k_values)
            analysis = analyze_failures_aiops(processed_rankings, max_k, 0)
            print(f"失败案例分析 (基于 PR@{max_k})")
            print('-' * 60)
            print(f"总故障注入次数: {analysis['total']}\n  - 成功检测 (排名前 {max_k}): {analysis['success']}\n  - 失败总数: {analysis['low_rank']}")
            if analysis['low_rank'] > 0:
                print("\n失败案例抽样 (最多显示5例):")
                analyze_failures_aiops(processed_rankings, max_k, 5)
            print('=' * 60)

    else: # gaia, ob, tt
        print(f"正在为 {DATASET} 数据集应用标准评估策略...")
        eval_params = dataset_config['evaluation_params']
        k_values = eval_params['k_values']
        
        overall_pr_results = evaluation_stage1_standard(experiment_results, all_ground_truth, k_values)
        
        print(f"{dataset_config['name']} - 第1阶段 异常检测评估结果 (标准策略)")
        print('=' * 60)
        
        print("总体结果:")
        for k in k_values:
            counts, total = overall_pr_results[k]
            pr_rate = counts / total if total > 0 else 0
            print(f"  PR@{k}: {pr_rate:.2%} ({counts}/{total})")
            
        print('=' * 60)

        if k_values:
            max_k = max(k_values)
            analysis = analyze_failures_standard(experiment_results, all_ground_truth, max_k, 0)
            total_failures = analysis['missing_data'] + analysis['not_in_list'] + analysis['low_rank']
            print(f"失败案例分析 (基于 PR@{max_k})")
            print('-' * 60)
            print(f"总故障注入次数: {analysis['total']}\n  - 成功检测 (排名前 {max_k}): {analysis['success']}\n  - 失败总数: {total_failures}")
            if total_failures > 0:
                print(f"    - 因数据缺失导致的失败: {analysis['missing_data']}")
                print(f"    - 因服务未被排名导致的失败: {analysis['not_in_list']}")
                print(f"    - 因模型性能导致的失败: {analysis['low_rank']}")
                print("\n失败案例抽样 (最多显示5例):")
                analyze_failures_standard(experiment_results, all_ground_truth, max_k, 5)
            print('=' * 60)

def main():
    global DATASET
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection results for a specified dataset across all modalities.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (ob, tt, gaia, aiops)')
    
    args = parser.parse_args()
    DATASET = args.dataset

    config = get_dataset_config(args.dataset)
    base_results_path = config['anomaly_output_path']
    
    print(f"开始对 {config['name']} 数据集进行多模态评估...")
    print(f"基础结果路径: {base_results_path}")

    # 定义要检查的模态子目录
    modalities_to_check = ['all', 'log', 'trace']
    found_any_results = False

    for modality in modalities_to_check:
        modality_path = os.path.join(base_results_path, modality)
        
        if os.path.exists(modality_path):
            found_any_results = True
            print(f"\n\n{'='*25} 正在评估模态: {modality.upper()} {'='*25}")
            
            # 创建一个配置副本，以避免修改原始配置
            modality_config = config.copy()
            # 将评估路径指向特定的模态子目录
            modality_config['anomaly_output_path'] = modality_path
            
            # 使用特定于模态的配置运行评估
            calculate_pr_stage1_by_dataset(modality_config)
        else:
            print(f"\n---> 未找到模态 '{modality}' 的结果目录，跳过。 (路径: {modality_path})")

    if not found_any_results:
        print("\n错误: 在基础路径下未找到任何模态 ('all', 'log', 'trace') 的结果目录。请检查路径或是否已运行异常检测脚本。")

if __name__ == "__main__":
    main()