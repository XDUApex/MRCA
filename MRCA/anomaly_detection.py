import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pytz
import json
import argparse
from config import build_model_path, get_dataset_config, get_stage1_modality, normalize_modality
import shutil
import time

from MRCA.rq2_profiling import RQ2Profiler


def parse_timestamp_to_utc(timestamp_series):
    """Robustly parse heterogeneous numeric/string timestamps into UTC datetimes."""
    numeric = pd.to_numeric(timestamp_series, errors='coerce')
    numeric_ratio = numeric.notna().mean() if len(numeric) else 0

    if numeric_ratio > 0.9:
        max_abs = numeric.abs().max()

        if max_abs < 1e8:
            # Some aggregated files store epoch seconds scaled down by 1000.
            numeric = numeric * 1000
            return pd.to_datetime(numeric, unit='s', utc=True, errors='coerce')
        if max_abs < 1e11:
            return pd.to_datetime(numeric, unit='s', utc=True, errors='coerce')
        if max_abs < 1e14:
            return pd.to_datetime(numeric, unit='ms', utc=True, errors='coerce')
        if max_abs < 1e17:
            return pd.to_datetime(numeric, unit='us', utc=True, errors='coerce')
        return pd.to_datetime(numeric, unit='ns', utc=True, errors='coerce')

    parsed = pd.to_datetime(timestamp_series, errors='coerce', utc=True)
    return parsed

def load_injection_times(fault_file_path):
    """
    智能加载故障注入时间。
    能处理 .json (ob/tt) 和 .csv (gaia/aiops) 格式。
    返回一个时间字符串列表 (格式: 'YYYY-MM-DD HH:MM:SS')。
    """
    injection_times = []
    
    if not os.path.exists(fault_file_path):
        print(f"Warning: Fault file not found: {fault_file_path}")
        return injection_times

    try:
        if fault_file_path.endswith('.json'):
            with open(fault_file_path, 'r') as f:
                fault_data = json.load(f)
            for hour_faults in fault_data.values():
                for fault in hour_faults:
                    injection_times.append(fault['inject_time'])
        
        elif fault_file_path.endswith('.csv'):
            df = pd.read_csv(fault_file_path)
            if 'st_time' in df.columns:
                time_series = pd.to_datetime(df['st_time'], errors='coerce').dropna().dt.strftime('%Y-%m-%d %H:%M:%S')
                injection_times = time_series.unique().tolist()
            else:
                print(f"Warning: 'st_time' column not found in {fault_file_path}")

    except Exception as e:
        print(f"Error processing fault file {fault_file_path}: {e}")
        
    return injection_times

def load_and_prepare_data(df, modality):
    """
    从DataFrame中提取特征，并根据指定的模态进行消融处理，最后进行缩放。
    这是实现模态消融的核心函数。
    :param df: 输入的DataFrame，必须包含 'LogTemplateSum' 和 'TraceLatency' 列。
    :param modality: 模态选择 ('all', 'log', 'trace')。
    :return: 缩放后的Numpy数组。
    """
    # 保证列存在
    if 'LogTemplateSum' not in df.columns or 'TraceLatency' not in df.columns:
        raise ValueError("DataFrame must contain 'LogTemplateSum' and 'TraceLatency' columns.")
        
    data = df[['LogTemplateSum', 'TraceLatency']].values.astype(np.float32)

    if modality == 'log':
        # 仅使用log模态，将trace特征(第二列, index=1)置零
        print("Applying 'log' modality: zeroing out trace features.")
        data[:, 1] = 0
    elif modality == 'trace':
        # 仅使用trace模态，将log特征(第一列, index=0)置零
        print("Applying 'trace' modality: zeroing out log features.")
        data[:, 0] = 0
    # 如果 modality == 'all'，则不进行任何操作
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.sigma = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        return self.decoder(z), mu, sigma

def vae_loss_function(reconstructed, x, mu, sigma, beta=0.5):
    """
    VAE损失函数 = 重构损失 + β * KL散度
    - 重构损失：MSE
    - KL散度：-0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    - beta: KL散度的权重系数，控制KL项的影响
    """
    recon_loss = torch.nn.functional.mse_loss(reconstructed, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2) + 1e-8) - mu.pow(2) - sigma.pow(2))
    return recon_loss + beta * kl_div, recon_loss, kl_div

def train_vae_multiple_days(base_input_folder, model_path, train_dates, epochs, learning_rate, modality, profiler=None):
    """使用多天数据训练VAE模型（带KL散度）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=2, output_size=2, latent_size=16, hidden_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    all_training_data = []
    for date_str in train_dates:
        input_folder = os.path.join(base_input_folder, date_str, 'aggregation')
        if os.path.exists(input_folder):
            for filename in os.listdir(input_folder):
                if filename.endswith('.csv'):
                    file_path = os.path.join(input_folder, filename)
                    df = pd.read_csv(file_path)
                    if profiler is not None:
                        profiler.add_input_file(file_path, len(df))

                    if len(df.columns) == 2:
                        df.columns = ['Timestamp', 'LogTemplateSum']
                        df['TraceLatency'] = 0
                    elif len(df.columns) >= 3:
                         df.columns = ['Timestamp', 'LogTemplateSum', 'TraceLatency'] + [f'extra_{i}' for i in range(len(df.columns) - 3)]

                    try:
                        scaled_data = load_and_prepare_data(df, modality)
                        all_training_data.append(scaled_data)
                    except ValueError as e:
                        print(f"Skipping file {filename} due to error: {e}")

    if not all_training_data:
        raise ValueError("No training data found in the specified folder.")

    combined_data = torch.tensor(np.vstack(all_training_data), dtype=torch.float32).to(device)
    if profiler is not None:
        profiler.add_records(int(combined_data.shape[0]))
    print(f"Training with {len(combined_data)} samples using '{modality}' modality (with KL divergence, beta=0.5).")

    if profiler is not None:
        profiler.start_phase('train')
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, mu, sigma = model(combined_data)
            loss, recon_loss, kl_div = vae_loss_function(reconstructed, combined_data, mu, sigma, beta=0.5)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Total Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, KL: {kl_div.item():.6f}")
    finally:
        if profiler is not None:
            profiler.end_phase('train')

    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def detect_anomalies_multiple_days(base_input_folder, base_output_folder, detection_dates, fault_files, dataset_name, model_path, threshold, modality, profiler=None):
    """检测多天异常，按日期和模态组织输出"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=2, output_size=2, latent_size=16, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    modality_specific_output_folder = os.path.join(base_output_folder, modality)
    print(f"Results will be saved in: {modality_specific_output_folder}")

    for date_str in detection_dates:
        print(f"Detecting anomalies for {date_str} using '{modality}' modality.")

        fault_file = None
        for ff in fault_files:
            if date_str in ff.replace("\\", "/"):
                fault_file = ff
                break

        if not fault_file or not os.path.exists(fault_file):
            print(f"Warning: No valid fault file found for {date_str}. Skipping...")
            continue

        injection_times = load_injection_times(fault_file)

        if not injection_times:
            print(f"No injection times found for {date_str} from {fault_file}. Skipping...")
            continue

        input_folder = os.path.join(base_input_folder, date_str, 'aggregation')
        output_folder = os.path.join(modality_specific_output_folder, date_str)

        if os.path.exists(output_folder):
            print(f"Cleaning up old anomaly detection results in: {output_folder}")
            shutil.rmtree(output_folder)

        if os.path.exists(input_folder):
            detect_anomalies(input_folder, output_folder, injection_times, dataset_name, model_path, threshold, modality, profiler=profiler)
        else:
            print(f"Warning: Input folder not found for {date_str}")

def detect_anomalies(input_folder, output_folder, injection_times, dataset_name, model_path, threshold, modality, profiler=None):
    """检测单天异常"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=2, output_size=2, latent_size=16, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    local_tz = pytz.timezone('Asia/Shanghai') if dataset_name == 'aiops' else None

    if profiler is not None:
        profiler.start_phase('infer')
    try:
        for target_time_str in injection_times:
            mse_scores = {}

            naive_time = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
            if local_tz:
                target_time_utc = local_tz.localize(naive_time).astimezone(pytz.utc)
            else:
                target_time_utc = pytz.utc.localize(naive_time)

            start_time = target_time_utc - timedelta(minutes=5)
            end_time = target_time_utc + timedelta(minutes=5)

            for filename in os.listdir(input_folder):
                if filename.endswith('.csv'):
                    log_path = os.path.join(input_folder, filename)
                    df = pd.read_csv(log_path)
                    if profiler is not None:
                        profiler.add_input_file(log_path, len(df))

                    if len(df.columns) == 2:
                        df.columns = ['Timestamp', 'LogTemplateSum']
                        df['TraceLatency'] = 0
                    elif len(df.columns) >= 3:
                        df.columns = ['Timestamp', 'LogTemplateSum', 'TraceLatency'] + [f'extra_{i}' for i in range(len(df.columns) - 3)]

                    if 'Timestamp' not in df.columns:
                        continue

                    df['Timestamp'] = parse_timestamp_to_utc(df['Timestamp'])
                    df = df.dropna(subset=['Timestamp'])
                    filtered_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

                    if filtered_df.empty:
                        continue

                    try:
                        scaled_data = load_and_prepare_data(filtered_df, modality)
                    except ValueError as e:
                        print(f"Skipping file {filename} in detection due to error: {e}")
                        continue

                    if profiler is not None:
                        profiler.add_records(len(filtered_df))

                    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)

                    start_latency = time.perf_counter()
                    with torch.no_grad():
                        reconstructed, _, _ = model(data_tensor)
                        mse_loss = torch.nn.functional.mse_loss(reconstructed, data_tensor, reduction='none').mean(dim=1)
                        mse_scores[filename] = mse_loss.mean().item()
                    if profiler is not None:
                        profiler.record_infer_latency((time.perf_counter() - start_latency) * 1000.0)

            sorted_services = sorted(mse_scores.items(), key=lambda x: x[1], reverse=True)

            safe_time_str = target_time_str.replace(":", "-")
            result_file = os.path.join(output_folder, f'ranked_services_{safe_time_str}.csv')

            with open(result_file, 'w') as f:
                for service, mse in sorted_services:
                    f.write(f"{service},{mse}\n")
    finally:
        if profiler is not None:
            profiler.end_phase('infer')

# def main():
#     parser = argparse.ArgumentParser(description='Anomaly detection for specified dataset')
#     parser.add_argument('--dataset', type=str, required=True, help='Dataset name (ob, tt, gaia, aiops)')
#     parser.add_argument('--modality', type=str, default='all', choices=['all', 'log', 'trace'],
#                         help="Modality to use for training and detection: 'all' (log+trace), 'log' only, or 'trace' only.")
    
#     args = parser.parse_args()
    
#     config = get_dataset_config(args.dataset)
    
#     print(f"Processing {config['name']} dataset...")
#     print(f"Selected modality: {args.modality}")
    
#     print("Step 1: Training VAE model with normal data...")
#     train_vae_multiple_days(
#         base_input_folder=os.path.join(config['processed_data_path'], 'normal'),
#         model_path=config['model_path'],
#         train_dates=config['normal_data']['dates'],
#         epochs=config['training_params']['epochs'],
#         learning_rate=config['training_params']['learning_rate'],
#         modality=args.modality
#     )
    
#     print("Step 2: Detecting anomalies in abnormal data...")
#     detect_anomalies_multiple_days(
#         base_input_folder=os.path.join(config['processed_data_path'], 'abnormal'),
#         base_output_folder=config['anomaly_output_path'],
#         detection_dates=config['abnormal_data']['dates'],
#         fault_files=config['fault_files'],
#         dataset_name=args.dataset,
#         model_path=config['model_path'],
#         threshold=config['training_params']['threshold'],
#         modality=args.modality
#     )

def main():
    parser = argparse.ArgumentParser(description='Anomaly detection for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (ob, tt, gaia, aiops)')
    parser.add_argument('--modality', type=str, default='all', 
                        choices=['all', 'log', 'trace', 'log+metric', 'trace+metric', 'tml', 'tm', 'ml', 'l', 't'],
                        help="Modality to use: all/log/trace/log+metric/trace+metric, or aliases tml/tm/ml/l/t")
    parser.add_argument('--experiment', type=str, default='rq1', choices=['rq1', 'rq3'],
                        help='Experiment namespace for path layout')
    parser.add_argument('--variant', type=str, default='base',
                        help='Scenario suffix to avoid overwriting results, e.g. base/1/2/3')
    
    args = parser.parse_args()
    normalized_modality = normalize_modality(args.modality)
    first_stage_modality = get_stage1_modality(normalized_modality)

    config = get_dataset_config(args.dataset, experiment=args.experiment, variant=args.variant)
    model_path = build_model_path(config, first_stage_modality)

    train_profiler = RQ2Profiler(
        dataset=args.dataset,
        script_name='anomaly_detection_train',
        stage='train',
        modality=first_stage_modality,
        experiment=args.experiment,
        variant=args.variant,
    )
    infer_profiler = RQ2Profiler(
        dataset=args.dataset,
        script_name='anomaly_detection_infer',
        stage='infer',
        modality=first_stage_modality,
        experiment=args.experiment,
        variant=args.variant,
    )

    print(f"Processing {config['name']} dataset...")
    print(f"Experiment: {args.experiment}, Variant: {args.variant}")
    print(f"Selected modality: {normalized_modality}")
    print(f"Stage-1 modality: {first_stage_modality}")
    print(f"Model path: {model_path}")
    
    print("Step 1: Training VAE model with normal data...")
    train_vae_multiple_days(
        base_input_folder=os.path.join(config['processed_data_path'], 'normal'),
        model_path=model_path,
        train_dates=config['normal_data']['dates'],
        epochs=config['training_params']['epochs'],
        learning_rate=config['training_params']['learning_rate'],
        modality=first_stage_modality,
        profiler=train_profiler,
    )
    train_artifact = train_profiler.write_json()
    print(f"[RQ2] Training profiling artifact saved: {train_artifact}")

    print("Step 2: Detecting anomalies in abnormal data...")
    detect_anomalies_multiple_days(
        base_input_folder=os.path.join(config['processed_data_path'], 'abnormal'),
        base_output_folder=config['anomaly_output_path'],
        detection_dates=config['abnormal_data']['dates'],
        fault_files=config['fault_files'],
        dataset_name=args.dataset,
        model_path=model_path,
        threshold=config['training_params']['threshold'],
        modality=first_stage_modality,
        profiler=infer_profiler,
    )
    infer_artifact = infer_profiler.write_json()
    print(f"[RQ2] Inference profiling artifact saved: {infer_artifact}")
    
    # 输出第一阶段完成的信息，为第二阶段做准备
    print(f"First stage (anomaly detection) completed using '{first_stage_modality}' modality.")
    print(f"Results saved in: {os.path.join(config['anomaly_output_path'], first_stage_modality)}")
    if normalized_modality in ['all+metric', 'log+metric', 'trace+metric']:
        print(f"Next: Run root cause localization to complete the '{normalized_modality}' experiment.")

if __name__ == "__main__":
    main()