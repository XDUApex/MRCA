import copy
import os
from datetime import datetime, timedelta


def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return dates


EXPERIMENT_LAYOUT = {
    'rq1': {
        'processed_root': 'rq1_processed_data',
        'model_root': 'rq1_models',
        'anomaly_root': 'anomaly_detection/rq1_anomaly_score',
        'rca_root': 'rq1_RCA',
    },
    'rq3': {
        'processed_root': 'processed_data',
        'model_root': 'models',
        'anomaly_root': 'anomaly_detection/anomaly_score',
        'rca_root': 'RCA',
    },
}


MODALITY_ALIASES = {
    'all': 'all',
    'tml': 'all+metric',
    'all+metric': 'all+metric',
    'log': 'log',
    'l': 'log',
    'ml': 'log+metric',
    'log+metric': 'log+metric',
    'trace': 'trace',
    't': 'trace',
    'tm': 'trace+metric',
    'trace+metric': 'trace+metric',
}


STAGE1_MODALITY = {
    'all': 'all',
    'log': 'log',
    'trace': 'trace',
    'all+metric': 'all',
    'log+metric': 'log',
    'trace+metric': 'trace',
}


def normalize_modality(modality):
    key = str(modality).strip().lower()
    if key not in MODALITY_ALIASES:
        raise ValueError(f"Unknown modality: {modality}. Supported: {sorted(MODALITY_ALIASES.keys())}")
    return MODALITY_ALIASES[key]


def get_stage1_modality(modality):
    normalized = normalize_modality(modality)
    return STAGE1_MODALITY[normalized]


def _safe_modality_for_filename(modality):
    return normalize_modality(modality).replace('+', '_plus_')


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _resolve_storage(experiment, variant):
    if experiment not in EXPERIMENT_LAYOUT:
        raise ValueError(f"Unknown experiment: {experiment}. Supported: {list(EXPERIMENT_LAYOUT.keys())}")

    storage = copy.deepcopy(EXPERIMENT_LAYOUT[experiment])
    variant = (variant or 'base').strip()
    storage['variant'] = variant
    return storage


# 数据集配置
DATASET_CONFIG = {
    'ob': {
        'name': 'OB Dataset',
        'normal_data': {
            'path': 'raw_data/normal_data',
            'dates': ['2022-08-22', '2022-08-23']
        },
        'abnormal_data': {
            'path': 'raw_data/abnormal',
            'dates': ['2022-08-22', '2022-08-23']
        },
        'fault_files': [
            'raw_data/abnormal/2022-08-22/2022-08-22-fault_list.json',
            'raw_data/abnormal/2022-08-23/2022-08-23-fault_list.json',
        ],
        'dataset_mapping': {
            '2022-08-22': 'OB Dataset 1',
            '2022-08-23': 'OB Dataset 2'
        },
        'training_params': {
            'epochs': 7500,
            'threshold': 0.01,
            'learning_rate': 1e-3
        },
        'evaluation_params': {
            'k_values': [1, 3, 5, 7, 9, 10],
            'show_individual_results': True,
            'max_individual_display': 5
        },
        'rq3_overrides': {
            'abnormal_data': {
                'path': os.environ.get('MRCA_OB_RQ3_ABNORMAL_PATH', '/home/fuxian/DataSet/RQ3/ob_scene3_storage')
            },
            'fault_files': [
                os.environ.get('MRCA_OB_RQ3_GT_2022_08_22', '/home/fuxian/DataSet/RQ3/ob_scene1_network/2022-08-22/groundtruth.csv'),
                os.environ.get('MRCA_OB_RQ3_GT_2022_08_23', '/home/fuxian/DataSet/RQ3/ob_scene1_network/2022-08-23/groundtruth.csv'),
            ],
        },
    },
    'tt': {
        'name': 'TT Dataset',
        'normal_data': {
            'path': 'raw_data/normal_data',
            'dates': ['2023-01-29', '2023-01-30']
        },
        'abnormal_data': {
            'path': 'raw_data/abnormal',
            'dates': ['2023-01-29', '2023-01-30']
        },
        'fault_files': [
            'raw_data/abnormal/2023-01-29/2023-01-29-fault_list.json',
            'raw_data/abnormal/2023-01-30/2023-01-30-fault_list.json',
        ],
        'dataset_mapping': {
            '2023-01-29': 'TT Dataset 1',
            '2023-01-30': 'TT Dataset 2'
        },
        'training_params': {
            'epochs': 8000,
            'threshold': 0.005,
            'learning_rate': 5e-4
        },
        'evaluation_params': {
            'k_values': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
            'show_individual_results': True,
            'max_individual_display': 3
        },
        'rq3_overrides': {
            'abnormal_data': {
                'path': os.environ.get('MRCA_TT_RQ3_ABNORMAL_PATH', '/home/fuxian/DataSet/RQ3/tt_scene3_storage')
            },
            'fault_files': [
                os.environ.get('MRCA_TT_RQ3_GT_2023_01_29', '/home/fuxian/DataSet/RQ3/tt_scene1_network/2023-01-29/groundtruth.csv'),
                os.environ.get('MRCA_TT_RQ3_GT_2023_01_30', '/home/fuxian/DataSet/RQ3/tt_scene1_network/2023-01-30/groundtruth.csv'),
            ],
        },
    },
    'gaia': {
        'name': 'GAIA Dataset',
        'raw_data_path': os.environ.get('MRCA_GAIA_RAW_PATH', '/home/fuxian/DataSet/RQ3/new_GAIA_scene3_storage'),
        'abnormal_data': {
            'path': os.environ.get('MRCA_GAIA_ABNORMAL_PATH', '/home/fuxian/MRCA/raw_data/abnormal'),
        },
        'date_ranges': {
            'train': {'start': '2021-07-01', 'end': '2021-07-03'},
            'detection': {'start': '2021-07-04', 'end': '2021-07-31'},
        },
        'fault_files_pattern': os.environ.get('MRCA_GAIA_GT_PATTERN', '/home/fuxian/DataSet/new_GAIA/{date}/groundtruth.csv'),
        'dataset_mapping_pattern': 'GAIA Dataset {date}',
        'entrypoint_service': 'logservice1',
        'training_params': {
            'epochs': 10000,
            'threshold': 0.008,
            'learning_rate': 2e-3
        },
        'evaluation_params': {
            'k_values': [1, 3, 5, 7, 9, 10],
            'show_individual_results': False,
            'max_individual_display': 5
        }
    },
    'aiops': {
        'name': 'AIOps Dataset',
        'raw_data_path': os.environ.get('MRCA_AIOPS_RAW_PATH', '/home/fuxian/DataSet/NewDataset/aiops'),
        'normal_data': {
            'path': 'raw_data/normal_data',
            'dates': ['2022-05-01', '2022-05-03', '2022-05-05', '2022-05-07', '2022-05-09']
        },
        'abnormal_data': {
            'path': os.environ.get('MRCA_AIOPS_ABNORMAL_PATH', '/home/fuxian/MRCA/raw_data/abnormal'),
            'dates': ['2022-05-01', '2022-05-03', '2022-05-05', '2022-05-07', '2022-05-09']
        },
        'fault_files': [
            os.environ.get('MRCA_AIOPS_GT_2022_05_01', '/home/fuxian/DataSet/NewDataset/aiops/2022-05-01/groundtruth.csv'),
            os.environ.get('MRCA_AIOPS_GT_2022_05_03', '/home/fuxian/DataSet/NewDataset/aiops/2022-05-03/groundtruth.csv'),
            os.environ.get('MRCA_AIOPS_GT_2022_05_05', '/home/fuxian/DataSet/NewDataset/aiops/2022-05-05/groundtruth.csv'),
            os.environ.get('MRCA_AIOPS_GT_2022_05_07', '/home/fuxian/DataSet/NewDataset/aiops/2022-05-07/groundtruth.csv'),
            os.environ.get('MRCA_AIOPS_GT_2022_05_09', '/home/fuxian/DataSet/NewDataset/aiops/2022-05-09/groundtruth.csv'),
        ],
        'dataset_mapping_pattern': 'AIOps Dataset {date}',
        'training_params': {
            'epochs': 6000,
            'threshold': 0.015,
            'learning_rate': 1e-3
        },
        'evaluation_params': {
            'k_values': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
            'show_individual_results': True,
            'max_individual_display': 0
        },
        'all_base_services': [
            'adservice',
            'cartservice',
            'checkoutservice',
            'currencyservice',
            'emailservice',
            'frontend',
            'paymentservice',
            'productcatalogservice',
            'recommendationservice',
            'shippingservice'
        ]
    }
}


def _apply_date_range_if_needed(config):
    if 'date_ranges' not in config:
        return config

    if isinstance(config['date_ranges']['train'], list):
        train_dates = config['date_ranges']['train']
        detection_dates = config['date_ranges']['detection']
    else:
        train_dates = generate_date_range(
            config['date_ranges']['train']['start'],
            config['date_ranges']['train']['end']
        )
        detection_dates = generate_date_range(
            config['date_ranges']['detection']['start'],
            config['date_ranges']['detection']['end']
        )

    config['normal_data'] = {
        'path': config['raw_data_path'],
        'dates': train_dates
    }
    # Preserve explicitly set abnormal_data.path; only use raw_data_path as fallback.
    existing_abnormal_path = config.get('abnormal_data', {}).get('path', config['raw_data_path'])
    config['abnormal_data'] = {
        'path': existing_abnormal_path,
        'dates': detection_dates
    }

    if 'fault_files_pattern' in config:
        config['fault_files'] = [config['fault_files_pattern'].format(date=d) for d in detection_dates]

    if 'dataset_mapping_pattern' in config:
        config['dataset_mapping'] = {d: config['dataset_mapping_pattern'].format(date=d) for d in detection_dates}

    return config


def get_dataset_output_name(dataset_name, variant='base'):
    variant = (variant or 'base').strip()
    if variant == 'base':
        return dataset_name
    return f"{dataset_name}-{variant}"


def build_model_path(config, modality='all'):
    safe_modality = _safe_modality_for_filename(get_stage1_modality(modality))
    return os.path.join(config['model_dir'], f'vae_model_{safe_modality}.pth')


def build_rca_output_dir(config, modality_combo):
    normalized = normalize_modality(modality_combo)
    return os.path.join(config['rca_output_path'], normalized)


def get_dataset_config(dataset_name, experiment='rq1', variant='base', overrides=None):
    """获取指定数据集配置，并按实验类型和场景构造规范化路径。"""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIG.keys())}")

    config = copy.deepcopy(DATASET_CONFIG[dataset_name])
    experiment = experiment.strip().lower()

    if experiment == 'rq3' and 'rq3_overrides' in config:
        _deep_update(config, config['rq3_overrides'])

    config = _apply_date_range_if_needed(config)

    if overrides:
        _deep_update(config, overrides)

    storage = _resolve_storage(experiment, variant)
    dataset_output_name = get_dataset_output_name(dataset_name, variant=variant)

    processed_data_path = os.path.join(storage['processed_root'], f'{dataset_name}_data')
    if storage['variant'] != 'base':
        processed_data_path = os.path.join(processed_data_path, storage['variant'])

    model_dir = os.path.join(storage['model_root'], dataset_output_name)
    anomaly_output_path = os.path.join(storage['anomaly_root'], dataset_output_name)
    rca_output_path = os.path.join(storage['rca_root'], dataset_output_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(anomaly_output_path, exist_ok=True)
    os.makedirs(rca_output_path, exist_ok=True)

    config['experiment'] = experiment
    config['variant'] = storage['variant']
    config['dataset_output_name'] = dataset_output_name
    config['storage'] = storage

    config['processed_data_path'] = processed_data_path
    config['model_dir'] = model_dir
    config['model_path'] = os.path.join(model_dir, 'vae_model.pth')
    config['anomaly_output_path'] = anomaly_output_path
    config['rca_output_path'] = rca_output_path

    return config