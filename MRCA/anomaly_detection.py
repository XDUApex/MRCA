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
from config import get_dataset_config

def load_injection_times(fault_file_path):
    """
    智能加载故障注入时间。
    能处理 .json (ob/tt) 和 .csv (gaia) 格式。
    返回一个时间字符串列表 (格式: 'YYYY-MM-DD HH:MM:SS')。
    """
    injection_times = []
    
    if not os.path.exists(fault_file_path):
        print(f"Warning: Fault file not found: {fault_file_path}")
        return injection_times

    try:
        if fault_file_path.endswith('.json'):
            # 处理 ob/tt 的 JSON 文件
            with open(fault_file_path, 'r') as f:
                fault_data = json.load(f)
            for hour_faults in fault_data.values():
                for fault in hour_faults:
                    injection_times.append(fault['inject_time'])
        
        elif fault_file_path.endswith('.csv'):
            # 处理 gaia 的 CSV 文件
            df = pd.read_csv(fault_file_path)
            # 假设故障开始时间列为 'st_time'
            if 'st_time' in df.columns:
                # 转换时间格式为 'YYYY-MM-DD HH:MM:SS'
                # errors='coerce' 会将无法转换的值变为 NaT, dropna() 再移除它们
                time_series = pd.to_datetime(df['st_time'], errors='coerce').dropna().dt.strftime('%Y-%m-%d %H:%M:%S')
                injection_times = time_series.unique().tolist() # 使用unique避免重复
            else:
                print(f"Warning: 'st_time' column not found in {fault_file_path}")

    except Exception as e:
        print(f"Error processing fault file {fault_file_path}: {e}")
        
    return injection_times

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

def train_vae_multiple_days(base_input_folder, model_path, train_dates, epochs, learning_rate):
    """使用多天数据训练VAE模型"""
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
                    data = df[['LogTemplateSum', 'TraceLatency']].values
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(data)
                    all_training_data.append(scaled_data)

    if not all_training_data:
        raise ValueError("No training data found in the specified folder.")
    
    combined_data = torch.tensor(np.vstack(all_training_data), dtype=torch.float32).to(device)
    print(f"Training with {len(combined_data)} samples")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, _, _ = model(combined_data)
        loss = torch.nn.functional.mse_loss(reconstructed, combined_data)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def detect_anomalies_multiple_days(base_input_folder, base_output_folder, detection_dates, fault_files, model_path, threshold):
    """检测多天异常，按日期组织输出"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=2, output_size=2, latent_size=16, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    for date_str in detection_dates:
        print(f"Detecting anomalies for {date_str}")
        
        fault_file = None
        for ff in fault_files:
            if date_str in ff:
                fault_file = ff
                break
        
        if not fault_file:
            print(f"No fault file found for {date_str}")
            continue
            
        # 使用新的通用函数加载故障注入时间
        injection_times = load_injection_times(fault_file)
        
        if not injection_times:
            print(f"No injection times found for {date_str} from {fault_file}")
            continue
        
        # 检测异常 - 为每个日期创建单独的输出文件夹
        input_folder = os.path.join(base_input_folder, date_str, 'aggregation')
        output_folder = os.path.join(base_output_folder, date_str)  # 按日期组织输出
        
        if os.path.exists(input_folder):
            detect_anomalies(input_folder, output_folder, injection_times, model_path, threshold)
        else:
            print(f"Warning: Input folder not found for {date_str}")

def detect_anomalies(input_folder, output_folder, injection_times, model_path, threshold):
    """检测单天异常"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_size=2, output_size=2, latent_size=16, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    for target_time_str in injection_times:
        mse_scores = {}
        target_time = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
        target_time = pytz.utc.localize(target_time)
        start_time = target_time - timedelta(minutes=5)
        end_time = target_time + timedelta(minutes=5)

        for filename in os.listdir(input_folder):
            if filename.endswith('.csv'):
                log_path = os.path.join(input_folder, filename)
                df = pd.read_csv(log_path)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)
                filtered_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

                if filtered_df.empty:
                    continue

                data = filtered_df[['LogTemplateSum', 'TraceLatency']].values
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                data_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)

                with torch.no_grad():
                    reconstructed, _, _ = model(data_tensor)
                    mse_loss = torch.nn.functional.mse_loss(reconstructed, data_tensor, reduction='none').mean(dim=1)
                    mse_scores[filename] = mse_loss.mean().item()

        sorted_services = sorted(mse_scores.items(), key=lambda x: x[1], reverse=True)
        result_file = os.path.join(output_folder, f'ranked_services_{target_time_str.replace(":", "-")}.csv')
        with open(result_file, 'w') as f:
            for service, mse in sorted_services:
                f.write(f"{service},{mse}\n")

def main():
    parser = argparse.ArgumentParser(description='Anomaly detection for specified dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (ob, tt)')
    args = parser.parse_args()
    
    config = get_dataset_config(args.dataset)
    
    print(f"Processing {config['name']} dataset...")
    
    print("Step 1: Training VAE model with normal data...")
    train_vae_multiple_days(
        base_input_folder=os.path.join(config['processed_data_path'], 'normal'),
        model_path=config['model_path'],
        train_dates=config['normal_data']['dates'],
        epochs=config['training_params']['epochs'],
        learning_rate=config['training_params']['learning_rate']
    )
    
    print("Step 2: Detecting anomalies in abnormal data...")
    detect_anomalies_multiple_days(
        base_input_folder=os.path.join(config['processed_data_path'], 'abnormal'),
        base_output_folder=config['anomaly_output_path'],
        detection_dates=config['abnormal_data']['dates'],
        fault_files=config['fault_files'],
        model_path=config['model_path'],
        threshold=config['training_params']['threshold']
    )

if __name__ == "__main__":
    main()