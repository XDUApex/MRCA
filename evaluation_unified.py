import argparse
import json
import os
import re

import pandas as pd

from config import get_dataset_config, normalize_modality


MODALITY_PLAN = {
    'TML': {'source': 'stage2', 'modality_combo': 'all+metric'},
    'TM': {'source': 'stage2', 'modality_combo': 'trace+metric'},
    'ML': {'source': 'stage2', 'modality_combo': 'log+metric'},
    'TL': {'source': 'stage1', 'stage1_modality': 'all'},
    'L': {'source': 'stage1', 'stage1_modality': 'log'},
    'T': {'source': 'stage1', 'stage1_modality': 'trace'},
}


def normalize_service_name(service_name, dataset_name, keep_instance=False):
    if not isinstance(service_name, str):
        return ''

    name = service_name.replace('.csv', '').replace('_frequency', '').replace('_metric', '')

    # GAIA metric files are named "{IP}.{service}{N}_metric.csv"; strip the IP prefix.
    if dataset_name == 'gaia':
        name = re.sub(r'^(\d+\.){3,}\d+\.', '', name)

    if keep_instance:
        return name.replace('_', '-')

    if dataset_name == 'tt':
        # 统一将下划线转为横线，以兼容预测文件名和GT中的不同格式
        normalized_name = name.replace('_', '-')
        suffix = '-service'
        if suffix in normalized_name:
            return normalized_name[:normalized_name.find(suffix) + len(suffix)]
        return normalized_name

    if dataset_name in ['ob', 'aiops', 'gaia']:
        # AIOps和GAIA一样，去掉数字后缀
        # ob: 先统一分隔符，再取第一部分
        # aiops/gaia: 去掉数字后缀
        if dataset_name == 'ob':
            normalized_name = name.replace('_', '-')
            return normalized_name.split('-')[0]
        else:  # aiops, gaia
            # 先去掉可能残留的分隔符，再去掉所有数字
            result = name.replace('_', '').replace('-', '')
            return ''.join(ch for ch in result if not ch.isdigit())

    return name


def floor_to_second(ts):
    return pd.to_datetime(ts).floor('s')


def maybe_match_time_key(target_ts, available_times):
    target = floor_to_second(target_ts)
    if target in available_times:
        return target

    if not available_times:
        return None

    deltas = [(abs((candidate - target).total_seconds()), candidate) for candidate in available_times]
    deltas.sort(key=lambda x: x[0])
    if deltas[0][0] <= 60:
        return deltas[0][1]
    return None


def load_ground_truth(config, dataset_name):
    records = []

    for fault_file in config.get('fault_files', []):
        if not os.path.exists(fault_file):
            continue

        if fault_file.endswith('.json'):
            with open(fault_file, 'r') as file:
                data = json.load(file)

            for entries in data.values():
                for entry in entries:
                    injection_time = pd.to_datetime(entry.get('inject_time'), errors='coerce')
                    if pd.isna(injection_time):
                        ts_val = entry.get('inject_timestamp')
                        if ts_val is not None:
                            injection_time = pd.to_datetime(pd.to_numeric(ts_val), unit='s', errors='coerce')

                    if pd.isna(injection_time):
                        continue

                    raw_service = str(entry.get('inject_pod', ''))
                    records.append({
                        'InjectionTime': floor_to_second(injection_time),
                        'date': floor_to_second(injection_time).strftime('%Y-%m-%d'),
                        'fault_type': entry.get('inject_type', 'unknown'),
                        'service': normalize_service_name(raw_service, dataset_name),
                        'service_instance': normalize_service_name(raw_service, dataset_name, keep_instance=True),
                    })
            continue

        df = pd.read_csv(fault_file)
        time_col = next((c for c in ['st_time', 'inject_time', 'time', 'timestamp', 'date'] if c in df.columns), None)
        service_col = next((c for c in ['instance', 'inject_pod', 'service', 'service_name'] if c in df.columns), None)
        type_col = next((c for c in ['anomaly_type', 'inject_type', 'fault_type', 'type'] if c in df.columns), None)
        if not time_col or not service_col:
            continue

        df['InjectionTime'] = pd.to_datetime(df[time_col], errors='coerce').map(floor_to_second)
        df['service'] = df[service_col].astype(str).map(lambda s: normalize_service_name(s, dataset_name))
        df['service_instance'] = df[service_col].astype(str).map(lambda s: normalize_service_name(s, dataset_name, keep_instance=True))
        df['fault_type'] = df[type_col] if type_col else 'unknown'
        df['date'] = df['InjectionTime'].dt.strftime('%Y-%m-%d')

        for _, row in df.dropna(subset=['InjectionTime']).iterrows():
            records.append({
                'InjectionTime': row['InjectionTime'],
                'date': row['date'],
                'fault_type': str(row['fault_type']),
                'service': row['service'],
                'service_instance': row['service_instance'],
            })

    if not records:
        return pd.DataFrame(columns=['InjectionTime', 'date', 'fault_type', 'services', 'service_instances'])

    gt_df = pd.DataFrame(records)
    grouped = (
        gt_df.groupby('InjectionTime')
        .agg(
            date=('date', 'first'),
            fault_type=('fault_type', 'first'),
            services=('service', lambda x: sorted(set([v for v in x if v]))),
            service_instances=('service_instance', lambda x: sorted(set([v for v in x if v]))),
        )
        .reset_index()
    )
    return grouped


def parse_ranked_service_file(file_path, dataset_name):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []

    df = pd.read_csv(file_path, header=None)
    if df.empty:
        return []

    df = df.iloc[:, :2]
    df.columns = ['service_file', 'score']
    rankings = [normalize_service_name(str(row['service_file']), dataset_name) for _, row in df.iterrows()]

    deduped = []
    seen = set()
    for svc in rankings:
        if svc and svc not in seen:
            deduped.append(svc)
            seen.add(svc)
    return deduped


def load_stage1_predictions(config, dataset_name, stage1_modality):
    result = {}
    root = config['anomaly_output_path']

    for date_str in config['abnormal_data']['dates']:
        candidate_dirs = [
            os.path.join(root, stage1_modality, date_str),
            os.path.join(root, date_str),
        ]
        date_dir = next((d for d in candidate_dirs if os.path.exists(d)), '')
        if not date_dir:
            continue

        for filename in os.listdir(date_dir):
            if not filename.startswith('ranked_services_') or not filename.endswith('.csv'):
                continue
            match = re.search(r'ranked_services_(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})\.csv', filename)
            if not match:
                continue
            time_key = floor_to_second(pd.to_datetime(
                f"{match.group(1)} {match.group(2)}:{match.group(3)}:{match.group(4)}"
            ))
            result[time_key] = parse_ranked_service_file(os.path.join(date_dir, filename), dataset_name)

    return result


def load_stage2_predictions(config, dataset_name, modality_combo):
    result = {}
    normalized_combo = normalize_modality(modality_combo)
    root = os.path.join(config['rca_output_path'], normalized_combo)

    for date_str in config['abnormal_data']['dates']:
        target_file = os.path.join(root, f'result-{date_str}.csv')
        if not os.path.exists(target_file):
            continue

        df = pd.read_csv(target_file)
        if df.empty or 'InjectionTime' not in df.columns or 'ServiceName' not in df.columns:
            continue

        df['InjectionTime'] = pd.to_datetime(df['InjectionTime'], errors='coerce').map(floor_to_second)

        for injection_time, group in df.groupby('InjectionTime', sort=False):
            services = [normalize_service_name(str(s), dataset_name) for s in group['ServiceName'].tolist()]
            service_instances = [normalize_service_name(str(s), dataset_name, keep_instance=True) for s in group['ServiceName'].tolist()]
            deduped = []
            seen = set()
            for svc in services:
                if svc and svc not in seen:
                    deduped.append(svc)
                    seen.add(svc)
            deduped_instances = []
            seen_instances = set()
            for svc in service_instances:
                if svc and svc not in seen_instances:
                    deduped_instances.append(svc)
                    seen_instances.add(svc)
            result[injection_time] = {
                'services': deduped,
                'service_instances': deduped_instances,
            }

    return result


def evaluate_hits(gt_df, prediction_map, dataset_name, modality_label, stage_name):
    rows = []
    available_times = set(prediction_map.keys())

    for _, row in gt_df.iterrows():
        injection_time = floor_to_second(row['InjectionTime'])
        matched_key = maybe_match_time_key(injection_time, available_times)
        ranked = prediction_map.get(matched_key, {}) if matched_key is not None else {}
        ranked_services = ranked.get('services', [])
        ranked_service_instances = ranked.get('service_instances', [])

        gt_services = set(row['services'])
        gt_service_instances = set(row.get('service_instances', []))
        if stage_name == 'stage2' and gt_service_instances:
            hit_source = ranked_service_instances
            gt_source = gt_service_instances
        else:
            hit_source = ranked_services
            gt_source = gt_services

        rows.append({
            'dataset': dataset_name,
            'date': row['date'],
            'injection_time': injection_time,
            'fault_type': row['fault_type'],
            'modality_combo': modality_label,
            'stage': stage_name,
            'hit@1': int(bool(gt_source & set(hit_source[:1]))),
            'hit@3': int(bool(gt_source & set(hit_source[:3]))),
            'hit@5': int(bool(gt_source & set(hit_source[:5]))),
            'predicted_count': len(hit_source),
            'gt_count': len(gt_source),
            'matched_time': matched_key,
        })

    return rows


def summarize_results(long_df):
    summary_df = (
        long_df.groupby(['dataset', 'stage', 'fault_type', 'modality_combo'])
        .agg(
            cases=('injection_time', 'count'),
            hr_at_1=('hit@1', 'mean'),
            hr_at_3=('hit@3', 'mean'),
            hr_at_5=('hit@5', 'mean'),
        )
        .reset_index()
    )

    overall_df = (
        long_df.groupby(['dataset', 'stage', 'modality_combo'])
        .agg(
            cases=('injection_time', 'count'),
            hr_at_1=('hit@1', 'mean'),
            hr_at_3=('hit@3', 'mean'),
            hr_at_5=('hit@5', 'mean'),
        )
        .reset_index()
    )

    pivot_df = summary_df.pivot_table(
        index=['dataset', 'stage', 'fault_type'],
        columns='modality_combo',
        values=['hr_at_1', 'hr_at_3', 'hr_at_5', 'cases'],
    )
    pivot_df.columns = [f'{metric}_{combo}' for metric, combo in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    return overall_df, summary_df, pivot_df


def run_evaluation(datasets, experiment, variant, output_dir, include_stage1=True, include_stage2=True):
    all_rows = []
    all_distributions = []

    for dataset_name in datasets:
        config = get_dataset_config(dataset_name, experiment=experiment, variant=variant)
        gt_df = load_ground_truth(config, dataset_name)
        if gt_df.empty:
            print(f'[WARN] Skip dataset {dataset_name}: ground truth empty')
            continue

        distribution = gt_df.groupby(['date', 'fault_type']).size().reset_index(name='case_count')
        distribution['dataset'] = dataset_name
        all_distributions.append(distribution)

        for label, plan in MODALITY_PLAN.items():
            if plan['source'] == 'stage1' and include_stage1:
                prediction_map = load_stage1_predictions(config, dataset_name, plan['stage1_modality'])
                all_rows.extend(evaluate_hits(gt_df, prediction_map, dataset_name, label, 'stage1'))

            if plan['source'] == 'stage2' and include_stage2:
                prediction_map = load_stage2_predictions(config, dataset_name, plan['modality_combo'])
                all_rows.extend(evaluate_hits(gt_df, prediction_map, dataset_name, label, 'stage2'))

    if not all_rows:
        raise RuntimeError('No evaluation rows generated. Please check result paths and dataset config.')

    long_df = pd.DataFrame(all_rows)
    overall_df, summary_df, pivot_df = summarize_results(long_df)
    distribution_df = pd.concat(all_distributions, ignore_index=True) if all_distributions else pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    long_path = os.path.join(output_dir, 'mrca_eval_long.csv')
    overall_path = os.path.join(output_dir, 'mrca_eval_overall.csv')
    summary_path = os.path.join(output_dir, 'mrca_eval_fault_type_summary.csv')
    pivot_path = os.path.join(output_dir, 'mrca_eval_fault_type_pivot.csv')
    dist_path = os.path.join(output_dir, 'mrca_fault_type_distribution.csv')

    long_df.to_csv(long_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pivot_df.to_csv(pivot_path, index=False)
    distribution_df.to_csv(dist_path, index=False)

    print(f'[INFO] Long table saved: {long_path}')
    print(f'[INFO] Overall summary saved: {overall_path}')
    print(f'[INFO] Fault-type summary saved: {summary_path}')
    print(f'[INFO] Fault-type pivot saved: {pivot_path}')
    print(f'[INFO] Fault-type distribution saved: {dist_path}')


def parse_datasets(arg_value):
    if not arg_value.strip():
        return ['ob', 'tt', 'gaia', 'aiops']
    return [item.strip() for item in arg_value.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser(description='Unified MRCA evaluator for stage1/stage2 results and fault-type stratified HR@K')
    parser.add_argument('--datasets', type=str, default='ob,tt,gaia,aiops', help='Comma-separated dataset list')
    parser.add_argument('--experiment', type=str, default='rq1', choices=['rq1', 'rq3'])
    parser.add_argument('--variant', type=str, default='base')
    parser.add_argument('--output-dir', type=str, default='reports/evaluation')
    parser.add_argument('--mode', type=str, default='both', choices=['stage1', 'stage2', 'both'])

    args = parser.parse_args()
    datasets = parse_datasets(args.datasets)

    run_evaluation(
        datasets=datasets,
        experiment=args.experiment,
        variant=args.variant,
        output_dir=args.output_dir,
        include_stage1=args.mode in ['stage1', 'both'],
        include_stage2=args.mode in ['stage2', 'both'],
    )


if __name__ == '__main__':
    main()
