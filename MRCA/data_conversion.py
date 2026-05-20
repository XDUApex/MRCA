import os
import pandas as pd
from config import get_dataset_config
from MRCA.rq2_profiling import RQ2Profiler

def convert_data(dataset_name, experiment='rq1', variant='base'):
    """根据数据集配置将聚合数据转换为统一的2维格式，并覆盖原始文件"""
    config = get_dataset_config(dataset_name, experiment=experiment, variant=variant)

    profiler = RQ2Profiler(
        dataset=dataset_name,
        script_name='data_conversion',
        stage='preprocess',
        modality='LT',
        experiment=experiment,
        variant=variant,
    )

    with profiler.phase('preprocess'):
        process_data(config['processed_data_path'], config['normal_data']['dates'], 'normal', profiler=profiler)
        process_data(config['processed_data_path'], config['abnormal_data']['dates'], 'abnormal', profiler=profiler)

    artifact = profiler.write_json()
    print(f"[RQ2] Profiling artifact saved: {artifact}")

def process_data(base_path, dates, data_type, profiler=None):
    """处理指定类型的数据（normal 或 abnormal）"""
    for date in dates:
        input_folder = os.path.join(base_path, data_type, date, 'aggregation')
        if not os.path.exists(input_folder):
            print(f"Warning: Aggregation folder for {data_type} data on date {date} does not exist. Skipping...")
            continue

        for filename in os.listdir(input_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(input_folder, filename)

                data = pd.read_csv(file_path)
                if profiler is not None:
                    profiler.add_input_file(file_path, len(data))

                if 'Timestamp' in data.columns:
                    log_template_max = data.iloc[:, 1:-1].max(axis=1)
                    trace_latency = data.iloc[:, -1]
                else:
                    print(f"Warning: {filename} does not contain required columns. Skipping...")
                    continue

                converted_data = pd.DataFrame({
                    'Timestamp': data['Timestamp'],
                    'LogTemplateSum': log_template_max,
                    'TraceLatency': trace_latency
                })

                if profiler is not None:
                    profiler.add_records(len(converted_data))

                converted_data.to_csv(file_path, index=False)
                print(f"Converted data saved to {file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert aggregated data to 2D format for specified dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., ob, tt, gaia, aiops)")
    parser.add_argument('--experiment', type=str, default='rq1', choices=['rq1', 'rq3'],
                        help='Experiment namespace for path layout')
    parser.add_argument('--variant', type=str, default='base',
                        help='Scenario suffix to avoid overwriting results, e.g. base/1/2/3')
    args = parser.parse_args()

    print(f"Experiment: {args.experiment}, Variant: {args.variant}")
    convert_data(args.dataset, experiment=args.experiment, variant=args.variant)