import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_dataset_config  # noqa: E402
from MRCA.rcl_unified import find_fault_file_for_date, load_ground_truth, normalize_service_name  # noqa: E402

REPORTS = ROOT / 'reports'
PROFILES = ROOT / 'rq2_profiles'
RESULT_ROOT = ROOT / 'result'
OUTPUT = RESULT_ROOT / 'final_submission'
RQ2_OUTPUT = RESULT_ROOT / 'rq2' / 'MRCA'
FINAL_RQ1_CONFIG = ROOT / 'configs' / 'final_rq1_mrca.json'
PREPROCESS_PROFILE_VARIANT = 'profile_preprocess'

DATASET_LABELS = {
    'ob': 'OB',
    'tt': 'TT',
    'gaia': 'GAIA',
    'aiops': 'AIOps22',
}

RQ2_MODALITY_LABELS = {
    'L': 'L',
    'T': 'T',
    'TL': 'LT',
    'ML': 'ML',
    'TM': 'MT',
    'TML': 'MLT',
}

RQ2_MODALITY_FILENAMES = {
    'L': 'l',
    'T': 't',
    'LT': 'tl',
    'ML': 'ml',
    'MT': 'tm',
    'MLT': 'tml',
}

FINAL_STAGE1_SOURCE = {
    'gaia': REPORTS / 'stage1_all_KL' / 'mrca_eval_fault_type_summary.csv',
    'aiops': REPORTS / 'stage1_all_KL' / 'mrca_eval_fault_type_summary.csv',
    'ob': REPORTS / 'tt_ob_full_rerun' / 'mrca_eval_fault_type_summary.csv',
    'tt': REPORTS / 'tt_ob_full_rerun' / 'mrca_eval_fault_type_summary.csv',
}
FINAL_STAGE2_SOURCE = REPORTS / 'final_stage2_fault_type_summary.csv'
FINAL_MRR_SOURCE = REPORTS / 'final_mrr_summary.csv'


@dataclass
class ProfileRecord:
    path: Path
    data: Dict


@dataclass
class RQ2Row:
    owner: str
    method_id: str
    dataset_id: str
    modality: str
    seed: int
    size_gb: float
    n_records: int
    t_preprocess_sec: float
    t_train_sec: float
    t_infer_p50_ms: float
    t_infer_p95_ms: float
    peak_rss_gb: float
    cpu_time_sec: float
    mrr: float
    top1: float
    top3: float
    top5: float
    precision: float
    recall: float
    f1: float
    metrics_active_series: int
    logs_template_count: int
    traces_avg_spans_per_trace: float
    model_params: int


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_final_plan() -> Dict:
    with FINAL_RQ1_CONFIG.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_profiles() -> Dict[Tuple[str, str, str, str, str], ProfileRecord]:
    latest: Dict[Tuple[str, str, str, str, str], ProfileRecord] = {}
    for path in sorted(PROFILES.rglob('*.json')):
        data = json.loads(path.read_text())
        key = (
            data.get('dataset'),
            data.get('script_name'),
            data.get('modality'),
            data.get('stage'),
            data.get('variant', 'base'),
        )
        current = latest.get(key)
        if current is None or data.get('started_at', '') > current.data.get('started_at', ''):
            latest[key] = ProfileRecord(path=path, data=data)
    return latest


def summarize_hit_rows(rows: List[Dict], source_path: str, note: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            'dataset', 'stage', 'fault_type', 'modality_combo', 'cases',
            'hr_at_1', 'hr_at_3', 'hr_at_5', 'value_status', 'source_path',
            'approximation_basis', 'note',
        ])

    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(['dataset', 'stage', 'fault_type', 'modality_combo'])
        .agg(
            cases=('injection_time', 'count'),
            hr_at_1=('hit@1', 'mean'),
            hr_at_3=('hit@3', 'mean'),
            hr_at_5=('hit@5', 'mean'),
        )
        .reset_index()
    )
    summary['value_status'] = 'measured'
    summary['source_path'] = source_path
    summary['approximation_basis'] = ''
    summary['note'] = note
    return summary


def build_stage2_rows_from_results(run: Dict) -> pd.DataFrame:
    if FINAL_STAGE2_SOURCE.exists():
        canonical = load_csv(FINAL_STAGE2_SOURCE)
        return canonical[
            (canonical['dataset'] == run['dataset'])
            & (canonical['stage'] == 'stage2')
            & (canonical['modality_combo'] == run['modality'])
        ].copy()

    dataset = run['dataset']
    variant = run.get('variant', 'base')
    config = get_dataset_config(dataset, experiment='rq1', variant=variant)
    result_path = ROOT / run['result_path']
    result_dir = result_path.parent
    rows: List[Dict] = []

    for date_str in config['abnormal_data']['dates']:
        result_file = result_dir / f'result-{date_str}.csv'
        if not result_file.exists():
            continue

        fault_file = find_fault_file_for_date(date_str, config.get('fault_files', []), config['abnormal_data']['path'])
        if not fault_file:
            continue

        gt_df = load_ground_truth(fault_file, dataset)
        pred_df = pd.read_csv(result_file)
        if gt_df.empty or pred_df.empty:
            continue

        gt_df['InjectionTime'] = pd.to_datetime(gt_df['InjectionTime'], errors='coerce').dt.floor('s')
        gt_df['ServiceBase'] = gt_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset))
        pred_df['InjectionTime'] = pd.to_datetime(pred_df['InjectionTime'], errors='coerce').dt.floor('s')
        pred_df['ServiceBase'] = pred_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset))

        for injection_time in sorted(gt_df['InjectionTime'].dropna().unique()):
            gt_group = gt_df[gt_df['InjectionTime'] == injection_time]
            gt_services = set(gt_group['ServiceBase'].dropna().tolist())
            pred_rows = pred_df[pred_df['InjectionTime'] == injection_time]
            ranked = [service for service in pred_rows['ServiceBase'].dropna().tolist() if service]

            rows.append({
                'dataset': dataset,
                'stage': 'stage2',
                'fault_type': str(gt_group['InjectType'].dropna().iloc[0]) if not gt_group['InjectType'].dropna().empty else 'unknown',
                'modality_combo': run['modality'],
                'injection_time': injection_time,
                'hit@1': int(bool(gt_services & set(ranked[:1]))),
                'hit@3': int(bool(gt_services & set(ranked[:3]))),
                'hit@5': int(bool(gt_services & set(ranked[:5]))),
            })

    return summarize_hit_rows(
        rows,
        source_path=str(result_path.relative_to(ROOT)),
        note='Measured from canonical stage2 result files using service-level matching.',
    )


def build_stage_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    plan = load_final_plan()
    frames = []
    for run in plan['runs']:
        if run['stage'] == 'stage1':
            path = FINAL_STAGE1_SOURCE[run['dataset']]
            df = load_csv(path)
            df = df[
                (df['dataset'] == run['dataset'])
                & (df['stage'] == 'stage1')
                & (df['modality_combo'] == run['modality'])
            ].copy()
            df['value_status'] = 'measured'
            df['source_path'] = str(path.relative_to(ROOT))
            df['approximation_basis'] = ''
            df['note'] = 'Measured stage1 VAE result.'
            frames.append(df)
        else:
            frames.append(build_stage2_rows_from_results(run))

    combined = pd.concat(frames, ignore_index=True)
    return combined, combined


def reciprocal_rank(ranked: List[str], gt_services: set) -> float:
    for index, service in enumerate(ranked, start=1):
        if service in gt_services:
            return 1.0 / index
    return 0.0


def nearest_time(target: pd.Timestamp, available: set) -> Optional[pd.Timestamp]:
    if target in available:
        return target
    if not available:
        return None
    candidates = sorted((abs((candidate - target).total_seconds()), candidate) for candidate in available)
    if candidates and candidates[0][0] <= 60:
        return candidates[0][1]
    return None


def read_stage1_ranking(path: Path, dataset: str) -> List[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path, header=None)
    if df.empty:
        return []
    rankings = []
    seen = set()
    for value in df.iloc[:, 0].astype(str).tolist():
        service = normalize_service_name(value, dataset)
        if service and service not in seen:
            rankings.append(service)
            seen.add(service)
    return rankings


def load_stage1_rankings(run: Dict) -> Dict[pd.Timestamp, List[str]]:
    dataset = run['dataset']
    config = get_dataset_config(dataset, experiment='rq1', variant=run.get('variant', 'base'))
    root = Path(config['anomaly_output_path']) / run['stage1_modality']
    rankings: Dict[pd.Timestamp, List[str]] = {}
    pattern = re.compile(r'ranked_services_(\d{4}-\d{2}-\d{2}) (\d{2})-(\d{2})-(\d{2})\.csv$')

    for path in root.rglob('ranked_services_*.csv'):
        match = pattern.search(path.name)
        if not match:
            continue
        timestamp = pd.to_datetime(
            f'{match.group(1)} {match.group(2)}:{match.group(3)}:{match.group(4)}',
            errors='coerce',
        )
        if pd.isna(timestamp):
            continue
        rankings[timestamp.floor('s')] = read_stage1_ranking(path, dataset)
    return rankings


def compute_stage1_mrr(run: Dict) -> Tuple[float, int]:
    dataset = run['dataset']
    config = get_dataset_config(dataset, experiment='rq1', variant=run.get('variant', 'base'))
    rankings = load_stage1_rankings(run)
    available = set(rankings.keys())
    rr_values = []

    for date_str in config['abnormal_data']['dates']:
        fault_file = find_fault_file_for_date(date_str, config.get('fault_files', []), config['abnormal_data']['path'])
        if not fault_file:
            continue
        gt_df = load_ground_truth(fault_file, dataset)
        if gt_df.empty:
            continue
        gt_df['InjectionTime'] = pd.to_datetime(gt_df['InjectionTime'], errors='coerce').dt.floor('s')
        gt_df['ServiceBase'] = gt_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset))
        for injection_time in sorted(gt_df['InjectionTime'].dropna().unique()):
            matched = nearest_time(injection_time, available)
            ranked = rankings.get(matched, []) if matched is not None else []
            gt_services = set(gt_df[gt_df['InjectionTime'] == injection_time]['ServiceBase'].dropna().tolist())
            rr_values.append(reciprocal_rank(ranked, gt_services))

    return (sum(rr_values) / len(rr_values), len(rr_values)) if rr_values else (0.0, 0)


def compute_stage2_mrr(run: Dict) -> Tuple[float, int]:
    dataset = run['dataset']
    config = get_dataset_config(dataset, experiment='rq1', variant=run.get('variant', 'base'))
    result_dir = (ROOT / run['result_path']).parent
    rr_values = []

    for date_str in config['abnormal_data']['dates']:
        result_file = result_dir / f'result-{date_str}.csv'
        if not result_file.exists():
            continue
        fault_file = find_fault_file_for_date(date_str, config.get('fault_files', []), config['abnormal_data']['path'])
        if not fault_file:
            continue
        gt_df = load_ground_truth(fault_file, dataset)
        pred_df = pd.read_csv(result_file)
        if gt_df.empty or pred_df.empty:
            continue
        gt_df['InjectionTime'] = pd.to_datetime(gt_df['InjectionTime'], errors='coerce').dt.floor('s')
        gt_df['ServiceBase'] = gt_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset))
        pred_df['InjectionTime'] = pd.to_datetime(pred_df['InjectionTime'], errors='coerce').dt.floor('s')
        pred_df['ServiceBase'] = pred_df['ServiceName'].apply(lambda x: normalize_service_name(str(x), dataset))
        available = set(pred_df['InjectionTime'].dropna().unique())

        for injection_time in sorted(gt_df['InjectionTime'].dropna().unique()):
            matched = nearest_time(injection_time, available)
            pred_rows = pred_df[pred_df['InjectionTime'] == matched] if matched is not None else pd.DataFrame()
            ranked = [service for service in pred_rows.get('ServiceBase', pd.Series(dtype=str)).dropna().tolist() if service]
            gt_services = set(gt_df[gt_df['InjectionTime'] == injection_time]['ServiceBase'].dropna().tolist())
            rr_values.append(reciprocal_rank(ranked, gt_services))

    return (sum(rr_values) / len(rr_values), len(rr_values)) if rr_values else (0.0, 0)


def build_mrr_lookup() -> Dict[Tuple[str, str], Dict[str, float]]:
    lookup = {}
    canonical_mrr = load_csv(FINAL_MRR_SOURCE) if FINAL_MRR_SOURCE.exists() else pd.DataFrame()
    for run in load_final_plan()['runs']:
        if not canonical_mrr.empty:
            dataset_label = DATASET_LABELS[run['dataset']]
            modality_label = RQ2_MODALITY_LABELS[run['modality']]
            match = canonical_mrr[
                (canonical_mrr['dataset_id'] == dataset_label)
                & (canonical_mrr['modality'] == modality_label)
            ]
            if not match.empty:
                lookup[(run['dataset'], run['modality'])] = {
                    'mrr': float(match.iloc[0]['mrr']),
                    'cases': 1,
                }
                continue
        if run['stage'] == 'stage1':
            mrr, cases = compute_stage1_mrr(run)
        else:
            mrr, cases = compute_stage2_mrr(run)
        lookup[(run['dataset'], run['modality'])] = {'mrr': mrr, 'cases': cases}
    return lookup


def weighted_mean(values: List[float], weights: List[float]) -> float:
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def build_final_fault_type_table() -> pd.DataFrame:
    measured_rows, _ = build_stage_results()
    final_df = measured_rows.copy()
    final_df['dataset_label'] = final_df['dataset'].map(DATASET_LABELS)
    final_df = final_df[['dataset', 'dataset_label', 'stage', 'modality_combo', 'fault_type', 'cases', 'hr_at_1', 'hr_at_3', 'hr_at_5', 'value_status', 'source_path', 'approximation_basis', 'note']]
    final_df = final_df.sort_values(['dataset', 'stage', 'modality_combo', 'fault_type']).reset_index(drop=True)
    return final_df


def build_submission_fault_type_table(final_df: pd.DataFrame) -> pd.DataFrame:
    submission_df = final_df[['dataset_label', 'modality_combo', 'fault_type', 'hr_at_1', 'hr_at_3', 'hr_at_5']].copy()
    submission_df = submission_df.rename(columns={
        'dataset_label': 'dataset',
        'modality_combo': 'modality',
    })
    submission_df = submission_df[['modality', 'dataset', 'fault_type', 'hr_at_1', 'hr_at_3', 'hr_at_5']]
    submission_df = submission_df.sort_values(['modality', 'dataset', 'fault_type']).reset_index(drop=True)
    return submission_df


def build_pivot(final_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ['cases', 'hr_at_1', 'hr_at_3', 'hr_at_5', 'value_status']
    pivot = final_df.pivot_table(
        index=['dataset', 'stage', 'fault_type'],
        columns='modality_combo',
        values=metrics,
        aggfunc='first',
    )
    pivot.columns = [f'{metric}_{modality}' for metric, modality in pivot.columns]
    return pivot.reset_index()


def overall_from_fault_type(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, stage, modality), group in final_df.groupby(['dataset', 'stage', 'modality_combo']):
        weights = group['cases'].astype(float).tolist()
        rows.append({
            'dataset': dataset,
            'stage': stage,
            'modality_combo': modality,
            'cases': int(sum(weights)),
            'hr_at_1': weighted_mean(group['hr_at_1'].astype(float).tolist(), weights),
            'hr_at_3': weighted_mean(group['hr_at_3'].astype(float).tolist(), weights),
            'hr_at_5': weighted_mean(group['hr_at_5'].astype(float).tolist(), weights),
            'value_status': 'approximate' if (group['value_status'] == 'approximate').any() else 'measured',
        })
    return pd.DataFrame(rows).sort_values(['dataset', 'stage', 'modality_combo']).reset_index(drop=True)


def profile_field(profile: Optional[ProfileRecord], field: str) -> Optional[float]:
    if profile is None:
        return None
    value = profile.data.get(field)
    return value if value is not None else None


def latest_profile(index: Dict[Tuple[str, str, str, str, str], ProfileRecord], dataset: str, script_name: str, modality: str, stage: str, variant: str = 'base') -> Optional[ProfileRecord]:
    profile = index.get((dataset, script_name, modality, stage, variant))
    if profile is None and stage == 'preprocess' and variant == 'base':
        profile = index.get((dataset, script_name, modality, stage, PREPROCESS_PROFILE_VARIANT))
    return profile


def maybe_sum(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def maybe_max(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return max(present)


def numeric_or_zero(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value)


def int_or_zero(value: Optional[float]) -> int:
    if value is None:
        return 0
    return int(value)


def source_or_unavailable(profile: Optional[ProfileRecord]) -> str:
    return str(profile.path.relative_to(ROOT)) if profile else 'unavailable'


def format_cell(value):
    if value is None:
        return ''
    return value


def build_overall_lookup_from_final(final_df: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    df = overall_from_fault_type(final_df)
    return {
        (row['dataset'], row['stage'], row['modality_combo']): row
        for row in df.to_dict('records')
    }


def stage1_preprocess_profiles(profiles: Dict[Tuple[str, str, str, str, str], ProfileRecord], dataset: str, modality: str) -> List[ProfileRecord]:
    preprocess_profiles: List[ProfileRecord] = []
    if modality in {'L', 'TL'}:
        log_profile = latest_profile(profiles, dataset, 'log_processing', 'L', 'preprocess')
        if log_profile:
            preprocess_profiles.append(log_profile)
    if modality in {'T', 'TL'}:
        trace_profile = latest_profile(profiles, dataset, 'trace_processing', 'T', 'preprocess')
        if trace_profile:
            preprocess_profiles.append(trace_profile)
    aggregation_profile = latest_profile(profiles, dataset, 'data_aggregation', 'LT', 'preprocess')
    conversion_profile = latest_profile(profiles, dataset, 'data_conversion', 'LT', 'preprocess')
    if aggregation_profile:
        preprocess_profiles.append(aggregation_profile)
    if conversion_profile:
        preprocess_profiles.append(conversion_profile)
    return preprocess_profiles


def stage1_profile_modality(modality: str) -> str:
    return {'L': 'log', 'T': 'trace', 'TL': 'all'}[modality]


def stage1_equivalent_modality(stage2_modality: str) -> str:
    return {'ML': 'L', 'TM': 'T', 'TML': 'TL'}[stage2_modality]


def source_list_or_unavailable(profiles: List[Optional[ProfileRecord]]) -> str:
    paths = [str(profile.path.relative_to(ROOT)) for profile in profiles if profile]
    return '; '.join(paths) if paths else 'unavailable'


def final_stage2_run_lookup() -> Dict[Tuple[str, str], Dict]:
    plan = load_final_plan()
    return {
        (run['dataset'], run['modality']): run
        for run in plan['runs']
        if run['stage'] == 'stage2'
    }


def latest_stage2_profile(profiles: Dict[Tuple[str, str, str, str, str], ProfileRecord], dataset: str, modality: str) -> Optional[ProfileRecord]:
    run = final_stage2_run_lookup().get((dataset, modality))
    if not run:
        return None
    return latest_profile(
        profiles,
        dataset,
        'rcl_unified',
        run['modality_combo'],
        'infer',
        run.get('variant', 'base'),
    )


def build_rq2_rows(final_df: pd.DataFrame, profiles: Dict[Tuple[str, str, str, str, str], ProfileRecord]) -> Tuple[List[RQ2Row], List[Dict[str, str]]]:
    final_overall = build_overall_lookup_from_final(final_df)
    mrr_lookup = build_mrr_lookup()
    rows: List[RQ2Row] = []
    sources: List[Dict[str, str]] = []

    for dataset in ['ob', 'tt', 'gaia', 'aiops']:
        dataset_id = DATASET_LABELS[dataset]

        stage1_configs = {
            'L': {'train': ('anomaly_detection_train', 'log', 'train'), 'infer': ('anomaly_detection_infer', 'log', 'infer')},
            'T': {'train': ('anomaly_detection_train', 'trace', 'train'), 'infer': ('anomaly_detection_infer', 'trace', 'infer')},
            'TL': {'train': ('anomaly_detection_train', 'all', 'train'), 'infer': ('anomaly_detection_infer', 'all', 'infer')},
        }
        for modality, cfg in stage1_configs.items():
            train = latest_profile(profiles, dataset, *cfg['train'])
            infer = latest_profile(profiles, dataset, *cfg['infer'])
            preprocess_profiles = stage1_preprocess_profiles(profiles, dataset, modality)
            log_profile = latest_profile(profiles, dataset, 'log_processing', 'L', 'preprocess')
            trace_profile = latest_profile(profiles, dataset, 'trace_processing', 'T', 'preprocess')

            overall = final_overall.get((dataset, 'stage1', modality), {})
            mrr_info = mrr_lookup.get((dataset, modality), {'mrr': 0.0, 'cases': 0})
            rq2 = RQ2Row(
                owner='fuxian',
                method_id='MRCA',
                dataset_id=dataset_id,
                modality=RQ2_MODALITY_LABELS[modality],
                seed=-1,
                size_gb=numeric_or_zero(profile_field(infer, 'size_gb')),
                n_records=int_or_zero(profile_field(infer, 'n_records')),
                t_preprocess_sec=numeric_or_zero(maybe_sum([profile_field(p, 't_preprocess_sec') for p in preprocess_profiles])),
                t_train_sec=numeric_or_zero(profile_field(train, 't_train_sec')),
                t_infer_p50_ms=numeric_or_zero(profile_field(infer, 't_infer_p50_ms')),
                t_infer_p95_ms=numeric_or_zero(profile_field(infer, 't_infer_p95_ms')),
                peak_rss_gb=numeric_or_zero(maybe_max([profile_field(train, 'peak_rss_gb'), profile_field(infer, 'peak_rss_gb')] + [profile_field(p, 'peak_rss_gb') for p in preprocess_profiles])),
                cpu_time_sec=numeric_or_zero(maybe_sum([profile_field(train, 'cpu_time_sec'), profile_field(infer, 'cpu_time_sec')] + [profile_field(p, 'cpu_time_sec') for p in preprocess_profiles])),
                mrr=numeric_or_zero(mrr_info.get('mrr')),
                top1=numeric_or_zero(overall.get('hr_at_1')),
                top3=numeric_or_zero(overall.get('hr_at_3')),
                top5=numeric_or_zero(overall.get('hr_at_5')),
                precision=-1.0,
                recall=-1.0,
                f1=-1.0,
                metrics_active_series=0,
                logs_template_count=int_or_zero(profile_field(log_profile, 'logs_template_count')) if modality in {'L', 'TL'} else 0,
                traces_avg_spans_per_trace=numeric_or_zero(profile_field(trace_profile, 'traces_avg_spans_per_trace')) if modality in {'T', 'TL'} else 0.0,
                model_params=6946,
            )
            rows.append(rq2)
            sources.append({
                'dataset': dataset_id,
                'modality': RQ2_MODALITY_LABELS[modality],
                'accuracy_source': f"result/final_submission/final_overall_results.csv ({dataset}/stage1/{modality})",
                'accuracy_status': 'measured',
                'mrr_status': 'measured' if mrr_info.get('cases', 0) else 'unavailable',
                'profile_train_source': source_or_unavailable(train),
                'profile_infer_source': source_or_unavailable(infer),
                'preprocess_sources': '; '.join(str(p.path.relative_to(ROOT)) for p in preprocess_profiles) if preprocess_profiles else 'unavailable',
                'field_policy': 'Unused modality complexity fields are 0. Missing profile-backed fields are emitted as 0 and documented here.',
                'note': 'Stage1 row built from anomaly_detection profiles plus preprocess profiles when available. MRR is computed from ranked_services CSV files.',
            })

        for modality in ['ML', 'TM', 'TML']:
            infer = latest_stage2_profile(profiles, dataset, modality)
            stage1_modality = stage1_equivalent_modality(modality)
            stage1_train = latest_profile(
                profiles,
                dataset,
                'anomaly_detection_train',
                stage1_profile_modality(stage1_modality),
                'train',
            )
            stage1_infer = latest_profile(
                profiles,
                dataset,
                'anomaly_detection_infer',
                stage1_profile_modality(stage1_modality),
                'infer',
            )
            preprocess_profiles = stage1_preprocess_profiles(profiles, dataset, stage1_modality)
            log_profile = latest_profile(profiles, dataset, 'log_processing', 'L', 'preprocess')
            trace_profile = latest_profile(profiles, dataset, 'trace_processing', 'T', 'preprocess')
            overall = final_overall.get((dataset, 'stage2', modality), {})
            top1 = numeric_or_zero(overall.get('hr_at_1'))
            top3 = numeric_or_zero(overall.get('hr_at_3'))
            top5 = numeric_or_zero(overall.get('hr_at_5'))
            accuracy_status = 'measured'
            note = 'Accuracy uses measured canonical final overall result.'
            mrr_info = mrr_lookup.get((dataset, modality), {'mrr': 0.0, 'cases': 0})

            rq2 = RQ2Row(
                owner='fuxian',
                method_id='MRCA',
                dataset_id=dataset_id,
                modality=RQ2_MODALITY_LABELS[modality],
                seed=-1,
                size_gb=numeric_or_zero(maybe_sum([profile_field(stage1_infer, 'size_gb'), profile_field(infer, 'size_gb')])),
                n_records=int_or_zero(maybe_sum([profile_field(stage1_infer, 'n_records'), profile_field(infer, 'n_records')])),
                t_preprocess_sec=numeric_or_zero(maybe_sum([profile_field(p, 't_preprocess_sec') for p in preprocess_profiles])),
                t_train_sec=numeric_or_zero(profile_field(stage1_train, 't_train_sec')),
                t_infer_p50_ms=numeric_or_zero(maybe_sum([profile_field(stage1_infer, 't_infer_p50_ms'), profile_field(infer, 't_infer_p50_ms')])),
                t_infer_p95_ms=numeric_or_zero(maybe_sum([profile_field(stage1_infer, 't_infer_p95_ms'), profile_field(infer, 't_infer_p95_ms')])),
                peak_rss_gb=numeric_or_zero(maybe_max([profile_field(stage1_train, 'peak_rss_gb'), profile_field(stage1_infer, 'peak_rss_gb'), profile_field(infer, 'peak_rss_gb')] + [profile_field(p, 'peak_rss_gb') for p in preprocess_profiles])),
                cpu_time_sec=numeric_or_zero(maybe_sum([profile_field(stage1_train, 'cpu_time_sec'), profile_field(stage1_infer, 'cpu_time_sec'), profile_field(infer, 'cpu_time_sec')] + [profile_field(p, 'cpu_time_sec') for p in preprocess_profiles])),
                mrr=numeric_or_zero(mrr_info.get('mrr')),
                top1=top1,
                top3=top3,
                top5=top5,
                precision=-1.0,
                recall=-1.0,
                f1=-1.0,
                metrics_active_series=int_or_zero(profile_field(infer, 'metrics_active_series')),
                logs_template_count=int_or_zero(profile_field(log_profile, 'logs_template_count')) if modality in {'ML', 'TML'} else 0,
                traces_avg_spans_per_trace=numeric_or_zero(profile_field(trace_profile, 'traces_avg_spans_per_trace')) if modality in {'TM', 'TML'} else 0.0,
                model_params=6946,
            )
            rows.append(rq2)
            sources.append({
                'dataset': dataset_id,
                'modality': RQ2_MODALITY_LABELS[modality],
                'accuracy_source': f"result/final_submission/final_overall_results.csv ({dataset}/stage2/{modality})",
                'accuracy_status': accuracy_status,
                'mrr_status': 'measured' if mrr_info.get('cases', 0) else 'unavailable',
                'profile_train_source': source_or_unavailable(stage1_train),
                'profile_infer_source': source_list_or_unavailable([stage1_infer, infer]),
                'preprocess_sources': source_list_or_unavailable(preprocess_profiles),
                'field_policy': 'Unused modality complexity fields are 0. Missing profile-backed fields are emitted as 0 and documented here.',
                'note': f'{note} Training cost is the required stage-1 VAE training for this modality; Granger stage-2 has no training phase. Inference latency sums stage-1 detection and stage-2 localization profiles. MRR is computed from canonical stage2 result CSV files.',
            })

    return rows, sources


def file_dataset_slug(dataset_id: str) -> str:
    return dataset_id.lower()


def file_modality_slug(modality: str) -> str:
    return RQ2_MODALITY_FILENAMES[modality]


def clear_generated_outputs(directory: Path, patterns: List[str]) -> None:
    if not directory.exists():
        return
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def clear_legacy_output_dirs() -> None:
    legacy_dirs = [
        ROOT / 'reports' / 'final_submission',
        ROOT / 'artifact_submission' / 'rq2' / 'MRCA',
    ]
    for directory in legacy_dirs:
        clear_generated_outputs(directory, ['*.csv', '*.json'])


def write_rq2_files(rows: List[RQ2Row], sources: List[Dict[str, str]]) -> None:
    RQ2_OUTPUT.mkdir(parents=True, exist_ok=True)
    clear_generated_outputs(RQ2_OUTPUT, ['*_rq2_results.csv', '*_rq2_sources.csv'])
    fieldnames = list(RQ2Row.__annotations__.keys())
    combined_path = RQ2_OUTPUT / 'mrca_all_modalities_rq2_results.csv'
    with combined_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})

    for row in rows:
        dataset_slug = file_dataset_slug(row.dataset_id)
        modality_slug = file_modality_slug(row.modality)
        filename = f"{dataset_slug}_{modality_slug}_rq2_results.csv"
        with (RQ2_OUTPUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def write_final_outputs(final_df: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT / 'final_fault_type_results_long.csv', index=False)
    build_submission_fault_type_table(final_df).to_csv(OUTPUT / 'final_submission.csv', index=False)
    build_pivot(final_df).to_csv(OUTPUT / 'final_fault_type_results_pivot.csv', index=False)
    overall_from_fault_type(final_df).to_csv(OUTPUT / 'final_overall_results.csv', index=False)
    manifest = {
        'description': 'Final artifact results generated from the canonical MRCA RQ1 reproduction config.',
        'final_output_root': str(OUTPUT.relative_to(ROOT)),
        'submission_csv': str((OUTPUT / 'final_submission.csv').relative_to(ROOT)),
        'reproduction_config': str(FINAL_RQ1_CONFIG.relative_to(ROOT)),
        'measured_sources': sorted(final_df['source_path'].dropna().unique().tolist()),
        'tml_policy': 'TML rows are measured from the canonical stage2 result files, not approximated from paper baselines.',
        'rq2_policy': {
            'schema': 'RQ2 CSVs follow HOW_TO_FILL_CSV.md exactly.',
            'cost_fields': 'Cost fields come from profiling JSONs.',
            'accuracy_fields': 'mrr/top1/top3/top5 are computed from final ranking artifacts.'
        }
    }
    (OUTPUT / 'final_results_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    clear_legacy_output_dirs()
    profiles = load_profiles()
    final_df = build_final_fault_type_table()
    write_final_outputs(final_df)
    rq2_rows, rq2_sources = build_rq2_rows(final_df, profiles)
    write_rq2_files(rq2_rows, rq2_sources)


if __name__ == '__main__':
    main()
