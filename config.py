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

# 数据集配置
DATASET_CONFIG = {
    'ob': {
        'name': 'OB Dataset',
        'processed_data_path': 'processed_data/ob_data',
        'anomaly_output_path': 'anomaly_detection/anomaly_score/ob',
        'model_path': 'models/ob/vae_model.pth',
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
            'raw_data/abnormal/2022-08-23/2022-08-23-fault_list.json'
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
        }
    },
    'tt': {
        'name': 'TT Dataset',
        'processed_data_path': 'processed_data/tt_data',
        'anomaly_output_path': 'anomaly_detection/anomaly_score/tt',
        'model_path': 'models/tt/vae_model.pth',
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
            'raw_data/abnormal/2023-01-30/2023-01-30-fault_list.json'
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
        }
    },
    'gaia': {
        'name': 'GAIA Dataset',
        'raw_data_path': '/home/fuxian/DataSet/new_GAIA',
        'processed_data_path': 'processed_data/gaia_data',
        'anomaly_output_path': 'anomaly_detection/anomaly_score/gaia',
        'model_path': 'models/gaia/vae_model.pth',
        'date_ranges': {
            'train': {'start': '2021-07-04', 'end': '2021-07-05'},
            'detection': {'start': '2021-07-04', 'end': '2021-07-31'},
        },
        'fault_files_pattern': '/home/fuxian/DataSet/new_GAIA/{date}/groundtruth.csv',
        'dataset_mapping_pattern': 'GAIA Dataset {date}',
        'training_params': {
            'epochs': 10000,
            'threshold': 0.008,
            'learning_rate': 2e-3
        },
        'evaluation_params': {
            'k_values': [1, 3, 5, 7, 9],
            'show_individual_results': False,
            'max_individual_display': 5
        }
    },
    'aiops': {
        'name': 'AIOps Dataset',
        'raw_data_path': '/home/fuxian/DataSet/NewDataset/aiops',
        'processed_data_path': 'processed_data/aiops_data',
        'anomaly_output_path': 'anomaly_detection/anomaly_score/aiops',
        'model_path': 'models/aiops/vae_model.pth',
        'date_ranges': {
            'train': ['2022-05-01', '2022-05-03'],
            'detection': ['2022-05-01', '2022-05-03', '2022-05-05', '2022-05-07', '2022-05-09'],
        },
        'fault_files_pattern': '/home/fuxian/DataSet/NewDataset/aiops/fault_data/{date}-fault_list.json',
        'dataset_mapping_pattern': 'AIOps Dataset {date}',
        'training_params': {
            'epochs': 6000,
            'threshold': 0.015,
            'learning_rate': 1e-3
        },
        'evaluation_params': {
            'k_values': [1, 3, 5, 7, 9],
            'show_individual_results': True,
            'max_individual_display': 3
        }
    }
}

def get_dataset_config(dataset_name):
    """获取指定数据集的配置"""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIG.keys())}")
    
    config = DATASET_CONFIG[dataset_name].copy()
    
    if 'date_ranges' in config:
        # 1. 根据 date_ranges 生成训练日期 (等同于 normal) 和检测日期 (等同于 abnormal)
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
        
        # 2. 动态创建 normal_data 字典，以匹配 ob/tt 的结构
        config['normal_data'] = {
            'path': config['raw_data_path'],
            'dates': train_dates
        }
        
        # 3. 动态创建 abnormal_data 字典，以匹配 ob/tt 的结构
        config['abnormal_data'] = {
            'path': config['raw_data_path'],
            'dates': detection_dates
        }
        
        # 4. 原有的 fault_files 和 dataset_mapping 逻辑现在基于 detection_dates
        if 'fault_files_pattern' in config:
            fault_files = []
            for date in detection_dates:  # 使用检测日期来生成故障文件列表
                fault_file = config['fault_files_pattern'].format(date=date)
                fault_files.append(fault_file)
            config['fault_files'] = fault_files
        
        if 'dataset_mapping_pattern' in config:
            dataset_mapping = {}
            for date in detection_dates:  # 使用检测日期来生成数据集映射
                dataset_mapping[date] = config['dataset_mapping_pattern'].format(date=date)
            config['dataset_mapping'] = dataset_mapping
    
    model_dir = os.path.dirname(config['model_path'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    return config