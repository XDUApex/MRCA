import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config import build_rca_output_dir, get_dataset_config, get_stage1_modality, normalize_modality
from MRCA.rq2_profiling import RQ2Profiler

warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

REQUIRED_COLUMNS = [
    'CpuUsageRate(%)', 'MemoryUsageRate(%)',
    'PodClientLatencyP90(s)', 'PodClientLatencyP95(s)', 'PodClientLatencyP99(s)',
    'PodServerLatencyP90(s)', 'PodServerLatencyP95(s)', 'PodServerLatencyP99(s)',
    'PodWorkload(Ops)', 'PodSuccessRate(%)',
    'NetworkReceiveBytes', 'NetworkTransmitBytes',
    'SyscallRead', 'SyscallWrite',
]
FRONT_SERVICE_METRICS = ['SuccessRate(%)', 'LatencyP50(s)', 'LatencyP90(s)', 'LatencyP95(s)', 'LatencyP99(s)']

# Legacy mode: only 3 backend metrics (matching original root_cause_localization.py)
LEGACY_BACKEND_METRICS = ['CpuUsageRate(%)', 'PodClientLatencyP99(s)', 'NodeNetworkReceiveBytes']

# Patterns used to select a concise key-metric subset when LEGACY_BACKEND_METRICS are absent (e.g. GAIA, AIOps).
_FALLBACK_METRIC_PATTERNS = [
    'cpu_total_pct', 'cpu_total_norm_pct',  # GAIA
    'cpu_usage_seconds',                     # AIOps
    'memory_usage_pct', 'memory_rss_pct',   # GAIA
    'memory_usage_mb',                       # AIOps (case-insensitive match)
    'network_in_bytes', 'network_inbound_bytes',
    'network_out_bytes', 'network_outbound_bytes',
    'network_receive_mb', 'network_transmit_mb',  # AIOps
]

def _select_fallback_columns(df):
    """Return a focused subset of numeric columns via pattern matching."""
    cols = list(df.select_dtypes(include='number').columns)
    selected = []
    for col in cols:
        col_lower = col.lower()
        if any(pat in col_lower for pat in _FALLBACK_METRIC_PATTERNS):
            selected.append(col)
    if not selected:
        selected = cols[:10]
    return selected


def normalize_service_name(service_name, dataset_name):
    import re as _re
    if not isinstance(service_name, str):
        return ''

    name = service_name.replace('.csv', '').replace('_frequency', '').replace('_metric', '')

    # GAIA metric files: "{IP}.{service}{N}_metric.csv" — strip the IP prefix first
    if dataset_name == 'gaia':
        name = _re.sub(r'^(\d+\.){3,}\d+\.', '', name)

    if dataset_name == 'tt':
        # 统一将下划线转为横线
        normalized_name = name.replace('_', '-')
        suffix = '-service'
        if suffix in normalized_name:
            return normalized_name[:normalized_name.find(suffix) + len(suffix)]
        return normalized_name

    if dataset_name in ['ob', 'aiops', 'gaia']:
        if dataset_name == 'ob':
            normalized_name = name.replace('_', '-')
            return normalized_name.split('-')[0]
        else:  # aiops, gaia：去掉数字后缀
            result = name.replace('_', '').replace('-', '')
            return ''.join(ch for ch in result if not ch.isdigit())

    return name


def load_ground_truth(ground_truth_path, dataset_name):
    if not os.path.exists(ground_truth_path):
        return pd.DataFrame(columns=['InjectionTime', 'ServiceName', 'InjectType'])

    if ground_truth_path.endswith('.json'):
        with open(ground_truth_path, 'r') as file:
            data = json.load(file)

        records = []
        for entries in data.values():
            for entry in entries:
                injection_time = pd.to_datetime(entry.get('inject_time'), errors='coerce')
                if pd.isna(injection_time):
                    ts_value = entry.get('inject_timestamp')
                    if ts_value is not None:
                        injection_time = pd.to_datetime(pd.to_numeric(ts_value), unit='s', errors='coerce')

                if pd.isna(injection_time):
                    continue

                records.append({
                    'InjectionTime': injection_time,
                    'ServiceName': normalize_service_name(entry.get('inject_pod', ''), dataset_name),
                    'InjectType': entry.get('inject_type', 'unknown')
                })

        return pd.DataFrame(records)

    df = pd.read_csv(ground_truth_path)
    time_col = next((c for c in ['st_time', 'inject_time', 'time', 'timestamp', 'date'] if c in df.columns), None)
    service_col = next((c for c in ['instance', 'inject_pod', 'service', 'service_name'] if c in df.columns), None)
    type_col = next((c for c in ['anomaly_type', 'inject_type', 'fault_type', 'type'] if c in df.columns), None)

    if not time_col or not service_col:
        return pd.DataFrame(columns=['InjectionTime', 'ServiceName', 'InjectType'])

    result = pd.DataFrame()
    result['InjectionTime'] = pd.to_datetime(df[time_col], errors='coerce')
    result['ServiceName'] = df[service_col].apply(lambda s: normalize_service_name(str(s), dataset_name))
    result['InjectType'] = df[type_col] if type_col else 'unknown'
    return result.dropna(subset=['InjectionTime'])


def load_injection_times(times_file, ground_truth_df):
    if os.path.exists(times_file):
        with open(times_file, 'r') as file:
            times = [line.strip() for line in file.readlines() if line.strip()]
        parsed = [pd.to_datetime(time_str, errors='coerce') for time_str in times]
        return [t for t in parsed if not pd.isna(t)]

    return sorted(ground_truth_df['InjectionTime'].dropna().unique().tolist())


def find_fault_file_for_date(date_str, fault_files, abnormal_data_root):
    for file_path in fault_files:
        if date_str in file_path.replace('\\', '/'):
            return file_path

    candidates = [
        os.path.join(abnormal_data_root, date_str, f'{date_str}-fault_list.json'),
        os.path.join(abnormal_data_root, date_str, 'groundtruth.csv'),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return ''


def find_time_file_for_date(date_str, abnormal_data_root):
    base = os.path.join(abnormal_data_root, date_str)
    candidates = [
        os.path.join(base, 'time.txt'),
        os.path.join(base, 'inject anomaly time point.txt'),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return os.path.join(base, 'time.txt')


def convert_service_name_for_metric_file(first_stage_name):
    service_name = first_stage_name.replace('_frequency', '_metric')
    parts = service_name.rsplit('_', 1)
    if len(parts) == 2:
        base_name, suffix = parts
        return f"{base_name.replace('_', '-')}_{suffix}"
    return service_name


def load_first_stage_results(anomaly_output_path, first_stage_modality, date_str, injection_times, top_percentage):
    """
    加载一阶段结果，返回两类信息：
    1. selected_services_by_time: 用于二阶段Granger的服务集合（按top_percentage截断，控制计算量）
    2. mse_scores_by_time: 所有服务的MSE分数（用于最终融合排序）
    """
    selected_services_by_time = {}
    mse_scores_by_time = {}

    candidate_folders = [
        os.path.join(anomaly_output_path, first_stage_modality, date_str),
        os.path.join(anomaly_output_path, date_str),
    ]
    first_stage_folder = next((folder for folder in candidate_folders if os.path.exists(folder)), '')

    if not first_stage_folder:
        for ts in injection_times:
            selected_services_by_time[ts] = set()
            mse_scores_by_time[ts] = {}
        return selected_services_by_time, mse_scores_by_time

    for injection_time in injection_times:
        safe_time = pd.to_datetime(injection_time).strftime('%Y-%m-%d %H-%M-%S')
        result_file = os.path.join(first_stage_folder, f'ranked_services_{safe_time}.csv')

        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            selected_services_by_time[injection_time] = set()
            mse_scores_by_time[injection_time] = {}
            continue

        ranked_df = pd.read_csv(result_file, header=None, names=['service_file', 'mse_score'])
        ranked_df['service_name'] = ranked_df['service_file'].apply(lambda x: os.path.splitext(str(x))[0])

        # 保存所有服务的MSE分数（用于融合排序）
        mse_scores_by_time[injection_time] = dict(zip(ranked_df['service_name'], ranked_df['mse_score']))

        # 用于二阶段Granger的服务集合（控制计算量）
        top_count = max(1, int(len(ranked_df) * top_percentage))
        selected_services_by_time[injection_time] = set(ranked_df.head(top_count)['service_name'].tolist())

    return selected_services_by_time, mse_scores_by_time


def load_front_service_window(front_service_path, window_start, window_end):
    if not os.path.exists(front_service_path):
        return pd.DataFrame()

    data = pd.read_csv(front_service_path)
    if 'Time' not in data.columns:
        return pd.DataFrame()

    data['Time'] = data['Time'].astype(str).str.replace(r'\s*\+.*', '', regex=True)
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data = data.dropna(subset=['Time']).set_index('Time')

    available_metrics = [metric for metric in FRONT_SERVICE_METRICS if metric in data.columns]
    if not available_metrics:
        available_metrics = _select_fallback_columns(data)
    if not available_metrics:
        return pd.DataFrame()

    result = data[available_metrics].loc[window_start:window_end].dropna(how='all')
    # Resample when data is high-frequency (>30 pts) to avoid spurious Granger significance
    if len(result) > 30:
        rule = os.environ.get('MRCA_RESAMPLE_RULE', '1T')
        result = result.resample(rule).mean().dropna(how='all')
    return result


def load_backend_services_window(metrics_folder, window_start, window_end, selected_services=None, profiler=None, required_columns=None):
    if not os.path.exists(metrics_folder):
        return {}

    import re as _re

    target_services = None
    if selected_services is not None:
        target_services = {
            os.path.splitext(convert_service_name_for_metric_file(service_name))[0]
            for service_name in selected_services
        }

    def _strip_ip_prefix(name):
        """Strip GAIA-style IP prefix like '0.0.0.1.' from a service file name."""
        return _re.sub(r'^(\d+\.){3,}\d+\.', '', name)

    required_columns = REQUIRED_COLUMNS if required_columns is None else required_columns

    services_data = {}
    for metric_file in glob.glob(os.path.join(metrics_folder, '*.csv')):
        service_name = os.path.splitext(os.path.basename(metric_file))[0]
        if target_services is not None:
            base_name = _strip_ip_prefix(service_name)
            if service_name not in target_services and base_name not in target_services:
                continue

        data = pd.read_csv(metric_file)
        if profiler is not None:
            profiler.add_input_file(metric_file, len(data))
        if 'Time' not in data.columns:
            continue

        data['Time'] = data['Time'].astype(str).str.replace(r'\s*\+.*', '', regex=True)
        data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        data = data.dropna(subset=['Time'])
        data = data.loc[(data['Time'] >= window_start) & (data['Time'] <= window_end)]

        if data.empty:
            continue

        data = data.set_index('Time')
        available_columns = [column for column in required_columns if column in data.columns]
        if not available_columns:
            available_columns = _select_fallback_columns(data)
        if not available_columns:
            continue

        filtered = data[available_columns].replace([np.inf, -np.inf], np.nan).dropna(how='all')
        if filtered.empty:
            continue

        # Resample when data is high-frequency (>30 pts) to avoid spurious Granger significance
        if len(filtered) > 30:
            rule = os.environ.get('MRCA_RESAMPLE_RULE', '1T')
            filtered = filtered.resample(rule).mean().dropna(how='all')
        if filtered.empty:
            continue

        if profiler is not None:
            profiler.add_metric_series(service_name, available_columns)
            profiler.add_records(len(filtered))

        services_data[service_name] = filtered

    return services_data


def perform_causality_test(merged_df, front_metrics, other_metrics, max_lag=2):
    results = []

    valid_front_metrics = []
    valid_other_metrics = []

    for metric in front_metrics:
        if metric in merged_df.columns and merged_df[metric].dropna().nunique() > 1:
            valid_front_metrics.append(metric)

    for metric in other_metrics:
        if metric in merged_df.columns and merged_df[metric].dropna().nunique() > 1:
            valid_other_metrics.append(metric)

    for front_metric in valid_front_metrics:
        for other_metric in valid_other_metrics:
            try:
                subset = merged_df[[front_metric, other_metric]].dropna()
                if len(subset) < max_lag + 1:
                    continue

                test_result = grangercausalitytests(subset, max_lag, verbose=False)
                p_value = min(test_result[lag][0]['ssr_ftest'][1] for lag in test_result)
                results.append((front_metric, other_metric, p_value))
            except Exception:
                continue

    return sorted(results, key=lambda x: x[2])


def aggregate_pvalues(p_values, method='min'):
    """将多个p-value聚合为一个分数。method: min/avg/significant_count/fisher/bonferroni"""
    if not p_values:
        return 1.0

    clean = [p for p in p_values if p is not None and not np.isnan(p)]
    if not clean:
        return 1.0

    if method == 'min':
        return min(clean)
    elif method == 'avg':
        return np.mean(clean)
    elif method == 'significant_count':
        # 显著检验越多 → 分数越低（排序靠前）
        threshold = 0.05
        return -sum(1 for p in clean if p < threshold)
    elif method == 'bonferroni':
        # Bonferroni校正后取最小
        return min(p * len(clean) for p in clean)
    elif method == 'fisher':
        # Fisher's method: -2 * sum(ln(p))，值越大越显著 → 取负值用于排序
        try:
            from scipy.stats import chi2
            stat = -2.0 * sum(np.log(max(p, 1e-300)) for p in clean)
            # 转为p-value（自由度=2k），越小越显著
            return chi2.sf(stat, 2 * len(clean))
        except Exception:
            return min(clean)
    return min(clean)


def score_by_correlation(front_service_data, services_data):
    """基于相关性排序：取front_service与每个后端服务的最大|相关系数|"""
    service_scores = {}
    for service_name, service_data in services_data.items():
        merged_data = pd.merge_asof(
            front_service_data.sort_index(),
            service_data.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
        )
        if merged_data.empty or len(merged_data) < 3:
            continue

        max_corr = 0.0
        for fc in front_service_data.columns:
            if fc not in merged_data.columns:
                continue
            for bc in service_data.columns:
                if bc not in merged_data.columns:
                    continue
                pair = merged_data[[fc, bc]].dropna()
                if len(pair) < 3:
                    continue
                r = pair[fc].corr(pair[bc])
                if not np.isnan(r):
                    max_corr = max(max_corr, abs(r))

        # 分数 = 1 - max_corr，越小（相关性越强）越可疑
        service_scores[service_name] = 1.0 - max_corr

    return service_scores


def score_by_anomaly(services_data, window_start, window_end):
    """基于异常幅度排序：每个服务在窗口内的指标z-score最大值"""
    service_scores = {}
    for service_name, service_data in services_data.items():
        max_zscore = 0.0
        n_metrics = 0
        for col in service_data.columns:
            series = service_data[col].dropna()
            if len(series) < 2:
                continue
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                zs = abs((series - mean_val) / std_val).max()
                max_zscore = max(max_zscore, zs)
                n_metrics += 1

        if n_metrics == 0:
            continue
        # 分数 = -max_zscore，越大（越异常）排序越靠前
        service_scores[service_name] = -max_zscore

    return service_scores


def score_by_causal_outflow(services_data, max_lag=2, scoring='min', source_services=None):
    """
    全对因果外溢：对每个服务 X，测试 X 的指标是否 Granger-cause 其他所有服务的指标。
    因果外溢最强（p-value 聚合最低）的服务最可能是根因——故障从它传播出去。

    source_services: 如果指定，只对这些服务做因果外溢测试（大幅减少计算量）
    返回: {service_name: aggregated_pvalue}，越小越可疑
    """
    service_names = list(services_data.keys())
    n = len(service_names)
    if n < 2:
        return {}

    # 如果指定了源服务候选，只测试它们的因果外溢
    if source_services is not None:
        source_indices = [i for i, name in enumerate(service_names) if name in source_services]
        if not source_indices:
            return {}
    else:
        source_indices = range(n)

    # 预计算所有有向配对 Granger p-value: (src, tgt) -> [p_values]
    pairwise = {}
    for i in source_indices:
        src_name = service_names[i]
        src_data = services_data[src_name]
        for j, tgt_name in enumerate(service_names):
            if i == j:
                continue
            tgt_data = services_data[tgt_name]

            merged = pd.merge_asof(
                src_data.sort_index(),
                tgt_data.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest',
                suffixes=('_src', '_tgt'),
            )
            if merged.empty or len(merged) < max_lag + 1:
                continue

            pair_pvals = []
            for src_col in src_data.columns:
                src_key = f'{src_col}_src' if src_col in tgt_data.columns else src_col
                if src_key not in merged.columns:
                    continue
                for tgt_col in tgt_data.columns:
                    tgt_key = f'{tgt_col}_tgt' if tgt_col in src_data.columns else tgt_col
                    if tgt_key not in merged.columns:
                        continue
                    try:
                        subset = merged[[tgt_key, src_key]].dropna()
                        if len(subset) < max_lag + 1:
                            continue
                        if subset[tgt_key].nunique() <= 1 or subset[src_key].nunique() <= 1:
                            continue
                        result = grangercausalitytests(subset, max_lag, verbose=False)
                        p_value = min(result[lag][0]['ssr_ftest'][1] for lag in result)
                        pair_pvals.append(p_value)
                    except Exception:
                        continue
            if pair_pvals:
                pairwise[(i, j)] = pair_pvals

    outflow_scores = {}
    inflow_scores = {}
    for i, src_name in enumerate(service_names):
        # outflow: src → all others
        out_pvals = []
        for j in range(n):
            if i == j:
                continue
            if (i, j) in pairwise:
                out_pvals.extend(pairwise[(i, j)])
        if out_pvals:
            outflow_scores[src_name] = aggregate_pvalues(out_pvals, method=scoring)

        # inflow: all others → src
        in_pvals = []
        for j in range(n):
            if i == j:
                continue
            if (j, i) in pairwise:
                in_pvals.extend(pairwise[(j, i)])
        if in_pvals:
            inflow_scores[src_name] = aggregate_pvalues(in_pvals, method=scoring)

    score_mode = os.environ.get('MRCA_OUTFLOW_SCORE_MODE', 'net').strip().lower()
    if score_mode == 'outflow':
        return outflow_scores
    if score_mode == 'inflow':
        return inflow_scores

    # 净因果外溢: outflow_score - inflow_score
    # p-value 越小 = 因果越强, 所以 net = inflow_p - outflow_p (正值 = 外溢 > 流入 = 更可疑)
    # fuse_scores 期望 p-value 风格（越小越可疑），所以取负值
    net_scores = {}
    for svc in outflow_scores:
        out_p = outflow_scores[svc]
        in_p = inflow_scores.get(svc, 1.0)
        net_scores[svc] = -(in_p - out_p)  # 负值，越小（越负）越可疑

    return net_scores


def find_causality(front_service_data, services_data, max_lag=2, scoring='min', legacy=False):
    """
    执行Granger因果检验
    scoring: min/avg/significant_count/fisher/bonferroni
    legacy=False 返回: {service_name: score} (分数越低越可疑)
    legacy=True 返回: [(service_name, front_metric, other_metric, p_value), ...]
    """
    if legacy:
        all_results = []
        for service_name, service_data in services_data.items():
            merged_data = pd.merge_asof(
                front_service_data.sort_index(),
                service_data.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest',
            )

            if len(merged_data) < max_lag + 1:
                continue

            causality_results = perform_causality_test(
                merged_data,
                list(front_service_data.columns),
                list(service_data.columns),
                max_lag=max_lag,
            )
            all_results.extend((service_name, front_metric, other_metric, p_value)
                               for front_metric, other_metric, p_value in causality_results)
        return all_results

    service_scores = {}

    for service_name, service_data in services_data.items():
        merged_data = pd.merge_asof(
            front_service_data.sort_index(),
            service_data.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
        )

        if merged_data.empty:
            continue

        causality_results = perform_causality_test(
            merged_data,
            list(front_service_data.columns),
            list(service_data.columns),
            max_lag=max_lag,
        )

        if causality_results:
            all_pvalues = [p for _, _, p in causality_results]
            service_scores[service_name] = aggregate_pvalues(all_pvalues, method=scoring)

    return service_scores


def normalize_service_key(service_name):
    """将服务名归一化为统一格式，用于融合时的键匹配"""
    import re as _re
    # 去掉后缀
    name = service_name.replace('_frequency', '').replace('_metric', '')
    # 去掉GAIA IP前缀 (e.g. "0.0.0.1.")
    name = _re.sub(r'^(\d+\.){3,}\d+\.', '', name)
    # 统一为下划线格式（与stage-1输出一致）
    name = name.replace('-', '_')
    return name


def fuse_scores(mse_scores, granger_pvalues, alpha=0.5, beta=0.5):
    """
    融合一阶段MSE分数和二阶段Granger p-value

    策略：
    - MSE越大，异常程度越高 → 归一化后作为"先验异常度"（0-1，越大越异常）
    - p-value越小，因果证据越强 → 归一化后作为"因果证据"（0-1，越小证据越强）
    - 最终分数 = alpha * 先验异常度 + beta * (1 - 因果证据归一化)

    参数:
        mse_scores: {service_name: mse_score}
        granger_pvalues: {service_name: p_value}
        alpha: 一阶段先验权重
        beta: 二阶段因果权重

    返回:
        排序后的 [(service_name, fused_score), ...]，分数越高越可疑
        service_name 使用归一化后的统一格式
    """
    if not mse_scores and not granger_pvalues:
        return []

    # 只有一阶段结果
    if not granger_pvalues:
        max_mse = max(mse_scores.values()) if mse_scores else 1
        return sorted(mse_scores.items(), key=lambda x: x[1], reverse=True)

    # 归一化mse_scores的键
    normalized_mse = {}
    for service_name, mse_val in mse_scores.items():
        normalized_key = normalize_service_key(service_name)
        # 如果同一个服务出现多次，保留MSE较大的值
        if normalized_key not in normalized_mse or mse_val > normalized_mse[normalized_key]:
            normalized_mse[normalized_key] = mse_val

    # 归一化granger_pvalues的键
    normalized_granger = {}
    for service_name, p_val in granger_pvalues.items():
        normalized_key = normalize_service_key(service_name)
        # 如果同一个服务出现多次，保留p-value较小的值（证据更强）
        if normalized_key not in normalized_granger or p_val < normalized_granger[normalized_key]:
            normalized_granger[normalized_key] = p_val

    # 所有服务（使用归一化后的键）
    all_services = set(normalized_mse.keys()) | set(normalized_granger.keys())

    # 归一化MSE（越大越异常）
    mse_values = list(normalized_mse.values())
    mse_max = max(mse_values) if mse_values else 1
    mse_min = min(mse_values) if mse_values else 0
    mse_range = mse_max - mse_min if mse_max != mse_min else 1

    # 归一化p-value（越小证据越强，所以用1-normalized）
    p_values = [p for p in normalized_granger.values() if p is not None and not np.isnan(p)]
    p_max = max(p_values) if p_values else 1
    p_min = min(p_values) if p_values else 0
    p_range = p_max - p_min if p_max != p_min else 1

    fused_results = []
    for service in all_services:
        # 一阶段分数（归一化到0-1，越大越异常）
        mse = normalized_mse.get(service, 0)
        normalized_mse_score = (mse - mse_min) / mse_range if mse_range > 0 else 0

        # 二阶段分数（归一化到0-1，越小证据越强 → 转换后越大证据越强）
        p_val = normalized_granger.get(service)
        if p_val is not None and not np.isnan(p_val):
            normalized_p = (p_val - p_min) / p_range if p_range > 0 else 0
            causality_score = 1 - normalized_p  # p-value越小，分数越高
        else:
            causality_score = 0  # 没有因果证据的服务，分数为0

        # 融合分数
        fused = alpha * normalized_mse_score + beta * causality_score
        fused_results.append((service, fused))

    return sorted(fused_results, key=lambda x: x[1], reverse=True)


def aggregate_scores_to_service_base(mse_scores, granger_pvalues, dataset_name):
    """Aggregate instance-level keys to service-level keys for service-level evaluation."""
    aggregated_mse = {}
    for service_name, mse_val in mse_scores.items():
        service_base = normalize_service_name(str(service_name), dataset_name)
        if service_base not in aggregated_mse or mse_val > aggregated_mse[service_base]:
            aggregated_mse[service_base] = mse_val

    aggregated_granger = {}
    for service_name, p_val in granger_pvalues.items():
        service_base = normalize_service_name(str(service_name), dataset_name)
        if service_base not in aggregated_granger or p_val < aggregated_granger[service_base]:
            aggregated_granger[service_base] = p_val

    return aggregated_mse, aggregated_granger


def load_legacy_longtable_services_data(metrics_folder, dataset_name, profiler=None):
    if not os.path.exists(metrics_folder):
        return {}

    all_data = {}
    file_patterns = glob.glob(os.path.join(metrics_folder, '*.csv')) + glob.glob(os.path.join(metrics_folder, '*.csv.csv'))
    for metric_file in sorted(set(file_patterns)):
        metric_name = os.path.basename(metric_file)
        if metric_name.endswith('.csv.csv'):
            metric_name = metric_name[:-8]
        elif metric_name.endswith('.csv'):
            metric_name = metric_name[:-4]

        try:
            data = pd.read_csv(metric_file)
            if profiler is not None:
                profiler.add_input_file(metric_file, len(data))

            time_col = next((c for c in ['timestamp', 'time', 'Time', 'date'] if c in data.columns), None)
            service_id_col = next((c for c in ['cmdb_id', 'service', 'pod', 'instance', 'service_name'] if c in data.columns), None)
            value_col = 'value' if 'value' in data.columns else None
            if not all([time_col, service_id_col, value_col]):
                continue

            if time_col == 'timestamp':
                data[time_col] = pd.to_datetime(data[time_col], unit='s', errors='coerce')
            else:
                data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
            data = data.dropna(subset=[time_col, service_id_col])
            if data.empty:
                continue

            data = data.set_index(time_col)
            data['OriginalServiceName'] = data[service_id_col].astype(str)
            for service_name, group in data.groupby('OriginalServiceName'):
                if service_name not in all_data:
                    all_data[service_name] = {}
                all_data[service_name][metric_name] = group[value_col].resample('1T').mean()
        except Exception:
            continue

    services = {}
    for service_name, metric_map in all_data.items():
        df = pd.DataFrame(metric_map).replace([np.inf, -np.inf], np.nan).ffill()
        if df.empty:
            continue
        if profiler is not None:
            profiler.add_metric_series(service_name, list(df.columns))
            profiler.add_records(len(df))
        services[service_name] = df

    return services


def choose_legacy_front_service(all_services_data, dataset_name, entrypoint_service=None):
    if not all_services_data:
        return '', pd.DataFrame(), {}

    if entrypoint_service:
        for service_name, service_df in all_services_data.items():
            normalized = normalize_service_name(service_name, dataset_name)
            if entrypoint_service in service_name or normalized == entrypoint_service:
                backend = {k: v for k, v in all_services_data.items() if k != service_name}
                return service_name, service_df, backend

    preferred_names = ['frontend'] if dataset_name == 'aiops' else []
    if preferred_names:
        for service_name, service_df in all_services_data.items():
            normalized = normalize_service_name(service_name, dataset_name)
            if normalized in preferred_names or any(name in service_name for name in preferred_names):
                backend = {k: v for k, v in all_services_data.items() if k != service_name}
                return service_name, service_df, backend

    first_service = sorted(all_services_data.keys())[0]
    backend = {k: v for k, v in all_services_data.items() if k != first_service}
    return first_service, all_services_data[first_service], backend


def extract_legacy_front_metrics(front_service_df):
    if front_service_df.empty:
        return pd.DataFrame()

    lowered = {str(col).lower(): col for col in front_service_df.columns}
    preferred = [
        col for col in front_service_df.columns
        if any(token in str(col).lower() for token in ['latency', 'success', 'qps', 'request', 'response', 'availability'])
    ]
    if preferred:
        return front_service_df[preferred].dropna(axis=1, how='all')

    fallback = [
        col for col in front_service_df.columns
        if any(token in str(col).lower() for token in ['cpu', 'memory', 'network', 'io', 'load'])
    ]
    if fallback:
        return front_service_df[fallback].dropna(axis=1, how='all')

    numeric = front_service_df.select_dtypes(include='number')
    return numeric.dropna(axis=1, how='all')


def load_legacy_backend_services_window(metrics_folder, window_start, window_end, profiler=None):
    if not os.path.exists(metrics_folder):
        return {}

    services_data = {}
    for metric_file in glob.glob(os.path.join(metrics_folder, '*.csv')):
        service_name = os.path.splitext(os.path.basename(metric_file))[0]
        data = pd.read_csv(metric_file)
        if profiler is not None:
            profiler.add_input_file(metric_file, len(data))
        if 'Time' not in data.columns:
            continue

        data['Time'] = data['Time'].apply(lambda x: str(x).split(' +')[0])
        data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        data = data.dropna(subset=['Time'])
        data = data.loc[(data['Time'] >= window_start) & (data['Time'] <= window_end)]
        data = data.set_index('Time')

        if all(column in data.columns for column in LEGACY_BACKEND_METRICS):
            use_cols = LEGACY_BACKEND_METRICS
        else:
            use_cols = _select_fallback_columns(data)
            if not use_cols:
                continue

        filtered_data = data[use_cols].copy()
        filtered_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        filtered_data.dropna(how='all', inplace=True)
        if filtered_data.empty:
            continue

        if profiler is not None:
            profiler.add_metric_series(service_name, use_cols)
            profiler.add_records(len(filtered_data))

        services_data[service_name] = filtered_data

    return services_data


def perform_legacy_causality_test(df, front_service_metrics, other_service_metrics, max_lag=2):
    results = []
    for front_metric in front_service_metrics:
        for other_metric in other_service_metrics:
            try:
                subset = df[[front_metric, other_metric]].dropna()
                if len(subset) < max_lag + 1:
                    continue
                if subset[front_metric].nunique() <= 1 or subset[other_metric].nunique() <= 1:
                    continue
                causality_result = grangercausalitytests(subset, max_lag, verbose=False)
                p_value = min(causality_result[lag][0]['ssr_ftest'][1] for lag in causality_result)
                results.append((front_metric, other_metric, p_value))
            except Exception:
                continue
    return sorted(results, key=lambda x: x[2], reverse=True)


def find_legacy_causality_between_services(front_service_data, services_data, front_service_metrics, max_lag=2):
    results = []
    for service_name, service_data in services_data.items():
        numeric_service_data = service_data.select_dtypes(include='number')
        merged_data = pd.merge_asof(
            front_service_data.sort_index(),
            numeric_service_data.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
        )
        pair_results = perform_legacy_causality_test(
            merged_data,
            front_service_metrics,
            numeric_service_data.columns,
            max_lag=max_lag,
        )
        results.extend((service_name, front_metric, other_metric, p_value)
                       for front_metric, other_metric, p_value in pair_results)
    return results


def classify_anomaly_type(other_metric):
    if 'Latency' in str(other_metric) or 'NodeNetwork' in str(other_metric) or 'network' in str(other_metric).lower():
        return 'network_delay'
    if 'Cpu' in str(other_metric) or 'cpu' in str(other_metric).lower():
        return 'cpu_contention'
    return str(other_metric)


def build_fault_type_map(ground_truth_df, injection_times):
    mapping = {}
    for ts in injection_times:
        matched = ground_truth_df[
            (ground_truth_df['InjectionTime'] - pd.to_datetime(ts)).abs() <= pd.Timedelta(minutes=1)
        ]
        if matched.empty:
            mapping[ts] = 'unknown'
        else:
            mapping[ts] = str(matched.iloc[0]['InjectType'])
    return mapping


def run_for_date(config, dataset_name, date_str, modality_combo, first_stage_modality, top_percentage, output_dir, fusion_alpha=0.5, fusion_beta=0.5, granger_only=False, window_minutes=10, max_lag=2, scoring='min', method='granger', profiler=None, legacy=False):
    abnormal_data_root = config['abnormal_data']['path']
    metric_folder = os.path.join(abnormal_data_root, date_str, 'metric')
    front_service_path = os.path.join(abnormal_data_root, date_str, 'front_service.csv')

    if not os.path.exists(metric_folder):
        print(f"[WARN] Skip {date_str}: metric folder missing -> {metric_folder}")
        return None

    fault_file = find_fault_file_for_date(date_str, config.get('fault_files', []), abnormal_data_root)
    if not fault_file:
        print(f"[WARN] Skip {date_str}: fault file not found")
        return None

    ground_truth_df = load_ground_truth(fault_file, dataset_name)
    if ground_truth_df.empty:
        print(f"[WARN] Skip {date_str}: empty ground truth")
        return None

    times_file = find_time_file_for_date(date_str, abnormal_data_root)
    injection_times = load_injection_times(times_file, ground_truth_df)
    if not injection_times:
        print(f"[WARN] Skip {date_str}: no injection times")
        return None

    # 加载一阶段结果：包含截断的服务集合和所有服务的MSE分数
    selected_services_by_time, mse_scores_by_time = load_first_stage_results(
        config['anomaly_output_path'],
        first_stage_modality,
        date_str,
        injection_times,
        top_percentage,
    )

    fault_type_map = build_fault_type_map(ground_truth_df, injection_times)

    # Preload long-table metric data once for AIOps granger_outflow
    # Use original dataset path (long table format) instead of preprocessed wide table
    all_longtable_data = None
    if method == 'granger_outflow' and dataset_name == 'aiops':
        raw_data_path = config.get('raw_data_path', '')
        longtable_folder = os.path.join(raw_data_path, date_str, 'metric') if raw_data_path else metric_folder
        all_longtable_data = load_legacy_longtable_services_data(
            longtable_folder, dataset_name, profiler=profiler,
        )

    detailed_rows = []
    output_columns = ['InjectionTime', 'ServiceName', 'FusedScore', 'PValue', 'anomaly_type']
    for injection_time in injection_times:
        case_start = time.perf_counter()
        ts = pd.to_datetime(injection_time)

        if legacy:
            window_start = ts - pd.Timedelta(minutes=5)
            window_end = ts + pd.Timedelta(minutes=5)

            if os.path.exists(front_service_path):
                front_service_data = load_front_service_window(front_service_path, window_start, window_end)
                if front_service_data.empty:
                    continue

                services_data = load_legacy_backend_services_window(
                    metric_folder,
                    window_start,
                    window_end,
                    profiler=profiler,
                )
            else:
                all_services_data = load_legacy_longtable_services_data(
                    metric_folder,
                    dataset_name,
                    profiler=profiler,
                )
                if not all_services_data:
                    continue

                _, front_service_full, backend_full = choose_legacy_front_service(
                    all_services_data,
                    dataset_name,
                    entrypoint_service=config.get('entrypoint_service'),
                )
                if front_service_full.empty:
                    continue

                front_service_data = extract_legacy_front_metrics(front_service_full)
                if front_service_data.empty:
                    continue
                front_service_data = front_service_data.loc[window_start:window_end].dropna(axis=1, how='all')
                if front_service_data.empty:
                    continue

                services_data = {
                    service_name: service_df.loc[window_start:window_end].dropna(axis=1, how='all')
                    for service_name, service_df in backend_full.items()
                }
                services_data = {
                    service_name: service_df
                    for service_name, service_df in services_data.items()
                    if not service_df.empty
                }

            legacy_results = find_legacy_causality_between_services(
                front_service_data,
                services_data,
                list(front_service_data.columns),
                max_lag=2,
            )
            if not legacy_results:
                continue

            if profiler is not None:
                profiler.record_infer_latency((time.perf_counter() - case_start) * 1000.0)

            output_columns = ['InjectionTime', 'ServiceName', 'FrontMetric', 'OtherMetric', 'PValue', 'anomaly_type']
            ranked = sorted(legacy_results, key=lambda x: x[3], reverse=True)[:5]
            for service_name, front_metric, other_metric, p_val in ranked:
                detailed_rows.append((
                    ts,
                    service_name,
                    front_metric,
                    other_metric,
                    p_val,
                    classify_anomaly_type(other_metric),
                ))
            continue

        effective_window_minutes = window_minutes
        effective_max_lag = max_lag
        required_columns = REQUIRED_COLUMNS

        # GAIA/AIOps GT timestamps are UTC+8 but metric data is UTC — shift window by -8h
        metric_ts = ts - pd.Timedelta(hours=8) if dataset_name in ('gaia', 'aiops') else ts
        window_start = metric_ts - pd.Timedelta(minutes=effective_window_minutes)
        window_end = metric_ts + pd.Timedelta(minutes=effective_window_minutes)

        if method == 'granger_outflow':
            if dataset_name == 'aiops' and all_longtable_data is not None:
                # AIOps: long-table format (cmdb_id per row) — use preloaded data
                services_data = {
                    svc: df.loc[window_start:window_end].dropna(axis=1, how='all')
                    for svc, df in all_longtable_data.items()
                }
                services_data = {svc: df for svc, df in services_data.items() if not df.empty}
                if len(services_data) < 2:
                    continue

                # Filter to representative metrics (avoid 59-column explosion)
                representative_metrics = [
                    'container_cpu_usage_seconds', 'container_cpu_load_average_10s',
                    'container_cpu_user_seconds', 'container_memory_usage_MB',
                    'container_memory_working_set_MB', 'container_memory_rss',
                    'container_network_receive_MB', 'container_network_transmit_MB',
                    'container_fs_usage_MB', 'container_fs_writes_MB',
                    'service_mean_response_time', 'service_request_rate', 'service_success_rate',
                ]
                metric_override = os.environ.get('MRCA_AIOPS_OUTFLOW_METRICS', '').strip()
                if metric_override:
                    representative_metrics = [
                        metric.strip() for metric in metric_override.split(',')
                        if metric.strip()
                    ]
                services_data = {
                    svc: df[[c for c in representative_metrics if c in df.columns]]
                    for svc, df in services_data.items()
                }
                services_data = {svc: df for svc, df in services_data.items() if not df.empty and len(df.columns) > 0}

                # Map Stage1 names (paymentservice_0_frequency) → cmdb_id (paymentservice-0).
                # In granger-only mode, do not restrict source services by Stage1.
                stage1_services = set() if granger_only else selected_services_by_time.get(injection_time, set())
                stage1_cmdb_ids = set()
                for svc in stage1_services:
                    base = svc.replace('_frequency', '').replace('_metric', '')
                    cmdb_id = base.replace('_', '-')
                    if cmdb_id in services_data:
                        stage1_cmdb_ids.add(cmdb_id)

                outflow_services_data = services_data
                outflow_source_services = stage1_cmdb_ids if stage1_cmdb_ids else None
                if (
                    not granger_only
                    and stage1_cmdb_ids
                    and os.environ.get('MRCA_OUTFLOW_CANDIDATE_GRAPH', '').strip() == '1'
                ):
                    outflow_services_data = {
                        svc: df for svc, df in services_data.items()
                        if svc in stage1_cmdb_ids
                    }
                    outflow_source_services = None

                granger_pvalues = score_by_causal_outflow(
                    outflow_services_data,
                    max_lag=effective_max_lag,
                    scoring=scoring,
                    source_services=outflow_source_services,
                )
            else:
                # GAIA / wide-table datasets
                services_data = load_backend_services_window(
                    metric_folder,
                    window_start,
                    window_end,
                    selected_services=None,
                    profiler=profiler,
                    required_columns=required_columns,
                )
                if len(services_data) < 2:
                    continue
                stage1_services = set() if granger_only else selected_services_by_time.get(injection_time, set())
                stage1_metric_names = None
                if stage1_services:
                    import re as _re
                    stage1_metric_names = set()
                    for svc in stage1_services:
                        base = _re.sub(r'^(\d+\.){3,}\d+\.', '', svc)
                        base = os.path.splitext(convert_service_name_for_metric_file(base))[0]
                        stage1_metric_names.add(base)
                        stage1_metric_names.add(svc)
                    stage1_metric_names = {s for s in stage1_metric_names if s in services_data}
                granger_pvalues = score_by_causal_outflow(
                    services_data,
                    max_lag=effective_max_lag,
                    scoring=scoring,
                    source_services=stage1_metric_names if stage1_metric_names else None,
                )
        else:
            front_service_data = load_front_service_window(front_service_path, window_start, window_end)
            if front_service_data.empty:
                continue

            if granger_only:
                selected_services = None
            else:
                selected_services = selected_services_by_time.get(injection_time, set())
            services_data = load_backend_services_window(
                metric_folder,
                window_start,
                window_end,
                selected_services,
                profiler=profiler,
                required_columns=required_columns,
            )

            if method == 'correlation':
                granger_pvalues = score_by_correlation(front_service_data, services_data)
            elif method == 'anomaly':
                granger_pvalues = score_by_anomaly(services_data, window_start, window_end)
            else:
                granger_pvalues = find_causality(
                    front_service_data,
                    services_data,
                    max_lag=effective_max_lag,
                    scoring=scoring,
                    legacy=False,
                )

        mse_scores = mse_scores_by_time.get(injection_time, {})

        if not granger_pvalues and not mse_scores:
            continue

        if profiler is not None:
            profiler.record_infer_latency((time.perf_counter() - case_start) * 1000.0)

        if granger_only:
            ranked = sorted(granger_pvalues.items(), key=lambda x: x[1])
            for service_name, p_val in ranked[:5]:
                detailed_rows.append((
                    ts, service_name, p_val, p_val,
                    fault_type_map.get(injection_time, 'unknown')
                ))
        else:
            if (
                dataset_name == 'aiops'
                and os.environ.get('MRCA_AIOPS_SERVICE_LEVEL_FUSION', '').strip() == '1'
            ):
                mse_scores, granger_pvalues = aggregate_scores_to_service_base(
                    mse_scores, granger_pvalues, dataset_name,
                )
            fused_ranking = fuse_scores(mse_scores, granger_pvalues, alpha=fusion_alpha, beta=fusion_beta)

            seen_services = set()
            for service_name, fused_score in fused_ranking[:10]:
                if service_name in seen_services or len(seen_services) >= 5:
                    continue
                seen_services.add(service_name)
                normalized_service_key = normalize_service_key(service_name)
                p_val = float('nan')
                for granger_key, granger_p in granger_pvalues.items():
                    if normalize_service_key(granger_key) == normalized_service_key:
                        p_val = granger_p
                        break
                detailed_rows.append((
                    ts, service_name, fused_score, p_val,
                    fault_type_map.get(injection_time, 'unknown')
                ))

    if not detailed_rows:
        print(f"[WARN] No result rows for {date_str}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, f'result-{date_str}.csv')

    detailed_df = pd.DataFrame(detailed_rows, columns=output_columns)
    detailed_df.to_csv(detailed_path, index=False)

    print(f"[INFO] {date_str} -> saved: {detailed_path}")
    return detailed_path, fault_file


def evaluate_prk(result_files, fault_files, dataset_name, k_values):
    total_cases = 0
    hit_counts = {k: 0 for k in k_values}

    for result_file, fault_file in zip(result_files, fault_files):
        if not os.path.exists(result_file):
            continue

        pred_df = pd.read_csv(result_file)
        if pred_df.empty:
            continue

        gt_df = load_ground_truth(fault_file, dataset_name)
        if gt_df.empty:
            continue

        pred_df['InjectionTime'] = pd.to_datetime(pred_df['InjectionTime'], errors='coerce').dt.floor('s')
        pred_df['ServiceBase'] = pred_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset_name))
        gt_df['InjectionTime'] = pd.to_datetime(gt_df['InjectionTime'], errors='coerce').dt.floor('s')
        gt_df['ServiceBase'] = gt_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset_name))

        for inject_time in sorted(gt_df['InjectionTime'].dropna().unique()):
            gt_services = set(gt_df[gt_df['InjectionTime'] == inject_time]['ServiceBase'].dropna().tolist())
            pred_rows = pred_df[pred_df['InjectionTime'] == inject_time]
            if not gt_services:
                continue

            total_cases += 1
            for k in k_values:
                topk = set(pred_rows.head(k)['ServiceBase'].dropna().tolist())
                if not gt_services.isdisjoint(topk):
                    hit_counts[k] += 1

    return {
        k: (hit_counts[k] / total_cases if total_cases else 0.0, hit_counts[k], total_cases)
        for k in k_values
    }


def main():
    parser = argparse.ArgumentParser(description='Unified stage-2 RCA (with prior fusion)')
    parser.add_argument('--dataset', required=True, choices=['ob', 'tt', 'gaia', 'aiops'])
    parser.add_argument('--modality-combo', default='all+metric',
                        help='all+metric/log+metric/trace+metric or alias tml/ml/tm')
    parser.add_argument('--top-percentage', type=float, default=0.8,
                        help='Top percentage of stage-1 services for Granger (controls computation)')
    parser.add_argument('--fusion-alpha', type=float, default=0.5,
                        help='Weight for stage-1 prior (MSE)')
    parser.add_argument('--fusion-beta', type=float, default=0.5,
                        help='Weight for stage-2 evidence (Granger)')
    parser.add_argument('--granger-only', action='store_true',
                        help='Skip stage-1 filtering, run Granger on all services, rank by p-value only')
    parser.add_argument('--window-minutes', type=int, default=10,
                        help='Time window in minutes for Granger causality (default: 10)')
    parser.add_argument('--max-lag', type=int, default=2,
                        help='Maximum lag for Granger causality test (default: 2)')
    parser.add_argument('--scoring', default='min',
                        choices=['min', 'avg', 'significant_count', 'fisher', 'bonferroni'],
                        help='Method to aggregate pairwise Granger p-values into a service score')
    parser.add_argument('--method', default='granger',
                        choices=['granger', 'correlation', 'anomaly', 'granger_outflow'],
                        help='Stage-2 method: granger (default), correlation, window anomaly magnitude, '
                             'or granger_outflow')
    parser.add_argument('--experiment', type=str, default='rq1', choices=['rq1', 'rq3'])
    parser.add_argument('--variant', type=str, default='base')
    parser.add_argument('--k', default='1,3,5', help='PR@K list, e.g. 1,3,5')
    parser.add_argument('--legacy', action='store_true',
                        help='Enable legacy mode to exactly replicate root_cause_localization.py behavior: '
                             '3 backend metrics only (CpuUsageRate(%), PodClientLatencyP99(s), NodeNetworkReceiveBytes), '
                             '±5min window, max_lag=2, metric-pair level ranking (p-value descending)')

    args = parser.parse_args()

    modality_combo = normalize_modality(args.modality_combo)
    first_stage_modality = get_stage1_modality(modality_combo)
    k_values = tuple(int(x) for x in args.k.split(',') if x.strip())

    config = get_dataset_config(args.dataset, experiment=args.experiment, variant=args.variant)
    output_dir = build_rca_output_dir(config, modality_combo)
    os.makedirs(output_dir, exist_ok=True)

    profiler = RQ2Profiler(
        dataset=args.dataset,
        script_name='rcl_unified',
        stage='infer',
        modality=modality_combo,
        experiment=args.experiment,
        variant=args.variant,
    )

    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Experiment: {args.experiment}, Variant: {args.variant}")
    print(f"[INFO] Modality combo: {modality_combo}, Stage-1 modality: {first_stage_modality}")
    print(f"[INFO] Top percentage for Granger: {args.top_percentage}")
    print(f"[INFO] Fusion weights: alpha={args.fusion_alpha} (prior), beta={args.fusion_beta} (evidence)")
    if args.granger_only:
        if args.legacy:
            print('[INFO] Mode: Granger-only legacy (all services, metric-pair ranking, p-value descending, ±5min, max_lag=2)')
        else:
            print(f"[INFO] Mode: Granger-only (all services, rank by p-value)")
    print(f"[INFO] Output dir: {output_dir}")

    detailed_files = []
    fault_files = []
    profiler.start_phase('infer')
    try:
        for date_str in config['abnormal_data']['dates']:
            result = run_for_date(
                config=config,
                dataset_name=args.dataset,
                date_str=date_str,
                modality_combo=modality_combo,
                first_stage_modality=first_stage_modality,
                top_percentage=args.top_percentage,
                output_dir=output_dir,
                fusion_alpha=args.fusion_alpha,
                fusion_beta=args.fusion_beta,
                granger_only=args.granger_only,
                window_minutes=args.window_minutes,
                max_lag=args.max_lag,
                scoring=args.scoring,
                method=args.method,
                profiler=profiler,
                legacy=args.legacy,
            )
            if result is not None:
                detailed_path, fault_file = result
                detailed_files.append(detailed_path)
                fault_files.append(fault_file)
    finally:
        profiler.end_phase('infer')

    pr = evaluate_prk(detailed_files, fault_files, args.dataset, k_values)
    result_txt = os.path.join(output_dir, 'pr_results.txt')
    with open(result_txt, 'w') as file:
        file.write(f"Service-level PR on dataset {args.dataset}, modality {modality_combo}:\n")
        if args.granger_only:
            if args.legacy:
                file.write('Mode: Granger-only legacy (all services, metric-pair ranking, p-value descending, ±5min, max_lag=2)\n')
            else:
                file.write(f"Mode: Granger-only (all services, p-value ranking)\n")
        else:
            file.write(f"Fusion: alpha={args.fusion_alpha}, beta={args.fusion_beta}\n")
        file.write('---------------------------------------------\n')
        for k in k_values:
            score, hit, total = pr[k]
            line = f"PR@{k}: {score:.2%}  ({hit}/{total})"
            print(line)
            file.write(line + '\n')
        file.write('---------------------------------------------\n')

    print(f"[INFO] PR@K summary saved to: {result_txt}")
    artifact = profiler.write_json()
    print(f"[RQ2] Profiling artifact saved: {artifact}")


if __name__ == '__main__':
    main()
