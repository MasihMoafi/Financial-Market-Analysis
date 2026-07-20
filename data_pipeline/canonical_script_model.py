import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import precision_score
import logging
import os
import argparse
from pathlib import Path
import math
import joblib
from datetime import datetime
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_environment(seed: int) -> torch.device:
    """Sets random seeds for reproducibility and returns the appropriate device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return torch.device("cuda:0")
    logging.warning("No GPU available. Using CPU.")
    return torch.device("cpu")

def load_and_prepare_data(file_path: str, args: argparse.Namespace) -> tuple:
    """Loads, preprocesses, and splits the data into training and validation sets."""
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    if args.date_col in df.columns:
        dt_series = pd.to_datetime(df[args.date_col])
        df['hour'] = dt_series.dt.hour
        df['dayofweek'] = dt_series.dt.dayofweek
    else:
        raise ValueError(f"Date column '{args.date_col}' not found.")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(args.target_classes)
    y = label_encoder.transform(df[args.target_col])
    
    X_df = df[args.all_features].copy()
    split_idx = int(len(X_df) * (1.0 - args.val_size))
    
    X_train_df, X_val_df = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    scaler = RobustScaler()
    X_train_df.loc[:, args.numerical_features] = scaler.fit_transform(X_train_df[args.numerical_features])
    X_val_df.loc[:, args.numerical_features] = scaler.transform(X_val_df[args.numerical_features])
    
    return X_train_df.values, y_train, X_val_df.values, y_val, scaler, label_encoder

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> tuple:
    """Creates overlapping sequences from the time series data using stride tricks for efficiency."""
    n_samples = X.shape[0]
    shape = (n_samples - seq_length + 1, seq_length, X.shape[1])
    strides = (X.strides[0], X.strides[0], X.strides[1])
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides), y[seq_length - 1:]

class ForexDataset(Dataset):
    """Custom PyTorch Dataset for loading forex sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].size(1)]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GatedTransformer(nn.Module):
    """A dual-tower transformer with a gating mechanism."""
    def __init__(self, numerical_dim: int, d_model: int, n_heads: int, num_layers: int, d_ff: int,
                 num_classes: int, seq_len_T: int, dropout: float,
                 hour_emb_dim: int, day_emb_dim: int):
        super().__init__()
        self.d_model = d_model
        self.numerical_dim = numerical_dim
        self.hour_embedding = nn.Embedding(24, hour_emb_dim)
        self.day_embedding = nn.Embedding(7, day_emb_dim)
        time_emb_total_dim = hour_emb_dim + day_emb_dim
        d_model_for_numerical = self.d_model - time_emb_total_dim
        self.numerical_proj = nn.Linear(self.numerical_dim, d_model_for_numerical)
        self.step_wise_pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len_T + 100)
        step_encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True, norm_first=True)
        self.step_wise_transformer_encoder = nn.TransformerEncoder(step_encoder_layer, num_layers, nn.LayerNorm(d_model))
        input_dim_C = numerical_dim + 2
        self.channel_wise_input_proj = nn.Linear(seq_len_T, d_model)
        self.channel_wise_pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_dim_C + 100)
        channel_encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True, norm_first=True)
        self.channel_wise_transformer_encoder = nn.TransformerEncoder(channel_encoder_layer, num_layers, nn.LayerNorm(d_model))
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_numerical = x[..., :self.numerical_dim]
        x_time_int = x[..., self.numerical_dim:].long()
        x_hour, x_day = x_time_int[..., 0], x_time_int[..., 1]
        hour_emb = self.hour_embedding(x_hour)
        day_emb = self.day_embedding(x_day)
        time_embeddings = torch.cat([hour_emb, day_emb], dim=-1)
        numerical_projected = self.numerical_proj(x_numerical)
        x_projected = torch.cat([numerical_projected, time_embeddings], dim=-1)
        x_step = self.step_wise_pos_encoder(x_projected)
        h_step = self.step_wise_transformer_encoder(x_step)
        h_step_pooled = h_step.mean(dim=1)
        x_channel_permuted = x.permute(0, 2, 1)
        x_channel = self.channel_wise_input_proj(x_channel_permuted)
        x_channel = self.channel_wise_pos_encoder(x_channel)
        h_chan = self.channel_wise_transformer_encoder(x_channel)
        h_chan_pooled = h_chan.mean(dim=1)
        concat_features = torch.cat((h_step_pooled, h_chan_pooled), dim=1)
        gate_vals = torch.sigmoid(self.gate_linear(concat_features))
        final_repr = gate_vals * h_step_pooled + (1 - gate_vals) * h_chan_pooled
        return self.output_layer(final_repr)

class CostSensitiveLoss(nn.Module):
    """Applies a cost matrix as a regularization term to a base loss function."""
    def __init__(self, base_loss: nn.Module, cost_matrix: torch.Tensor):
        super().__init__()
        self.base_loss = base_loss
        self.register_buffer('cost_matrix', cost_matrix)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, current_lambda: float) -> torch.Tensor:
        base_loss_value = self.base_loss(inputs, targets)
        probs = torch.softmax(inputs, dim=1)
        sample_costs = self.cost_matrix[targets]
        expected_cost = torch.sum(probs * sample_costs, dim=1)
        reg_term = torch.mean(expected_cost)
        return base_loss_value + current_lambda * reg_term

class AdaptiveLambda:
    """Dynamically adjusts the lambda hyperparameter for the cost-sensitive loss."""
    def __init__(self, initial_lambda: float, patience: int, increase_factor: float, decrease_factor: float):
        self.current_lambda = initial_lambda
        self.patience = patience
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.best_metric = -1.0
        self.epochs_no_improve = 0

    def update(self, val_metric: float) -> float:
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            self.epochs_no_improve = 0
            self.current_lambda *= self.decrease_factor
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.current_lambda *= self.increase_factor
                self.epochs_no_improve = 0
        self.current_lambda = max(0.5, min(self.current_lambda, 20.0))
        return self.current_lambda

def evaluate_model(model, data_loader, criterion, device, label_encoder, args, current_lambda):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss, total_samples = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(enabled=args.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets, current_lambda)
            total_loss += loss.item() * inputs.size(0)
            total_samples += targets.size(0)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total_samples
    labels_order = list(range(len(label_encoder.classes_)))
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0, labels=labels_order)
    buy_idx = label_encoder.transform(['buy'])[0]
    sell_idx = label_encoder.transform(['sell'])[0]
    buy_precision = precision[buy_idx]
    sell_precision = precision[sell_idx]
    composite_score = (buy_precision + sell_precision) / 2
    return avg_loss, composite_score, buy_precision, sell_precision

def train_engine(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, scaler, label_encoder):
    """The main training loop with all professional practices integrated."""
    best_composite_score = -1.0
    output_dir = Path(args.output_dir)
    adaptive_lambda_scheduler = AdaptiveLambda(args.cost_initial_lambda, args.cost_lambda_patience, 1.2, 0.9)
    scaler_amp = GradScaler(enabled=args.use_amp)

    for epoch in range(args.epochs):
        model.train()
        current_lambda = adaptive_lambda_scheduler.current_lambda
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast(enabled=args.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets, current_lambda)
                loss = loss / args.grad_accum_steps
            scaler_amp.scale(loss).backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        val_loss, composite_score, bp, sp = evaluate_model(model, val_loader, criterion, device, label_encoder, args, current_lambda)
        adaptive_lambda_scheduler.update(composite_score)
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"E {epoch+1}/{args.epochs} | Val Loss: {val_loss:.4f} | Comp Score: {composite_score:.4f} (B:{bp:.2f}, S:{sp:.2f}) | Lambda: {current_lambda:.2f} | LR: {current_lr:.2e}")

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            logging.info(f"--> New best model. Saving checkpoint to {output_dir / 'checkpoint_best.pt'}")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_composite_score': best_composite_score,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'args': args
            }
            torch.save(checkpoint, output_dir / 'checkpoint_best.pt')
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"prod_run_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    device = setup_environment(args.seed)
    X_train, y_train, X_val, y_val, scaler, label_encoder = load_and_prepare_data(args.train_file, args)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, args.sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, args.sequence_length)
    train_dataset = ForexDataset(X_train_seq, y_train_seq)
    val_dataset = ForexDataset(X_val_seq, y_val_seq)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    model = GatedTransformer(
        numerical_dim=len(args.numerical_features), d_model=args.d_model, n_heads=args.n_heads,
        num_layers=args.num_layers, d_ff=args.d_ff, num_classes=len(label_encoder.classes_),
        seq_len_T=args.sequence_length, dropout=args.dropout, hour_emb_dim=args.hour_emb_dim, day_emb_dim=args.day_emb_dim
    ).to(device)

    if args.use_torch_compile:
        if hasattr(torch, 'compile'):
            logging.info("Applying torch.compile() to the model for performance.")
            model = torch.compile(model)
        else:
            logging.warning("torch.compile() is not available in this PyTorch version. Skipping.")

    base_loss = nn.CrossEntropyLoss()
    cost_matrix = torch.tensor(args.cost_matrix, dtype=torch.float32).to(device)
    criterion = CostSensitiveLoss(base_loss=base_loss, cost_matrix=cost_matrix)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.epochs * len(train_loader))
    
    logging.info("Starting model training...")
    train_engine(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, scaler, label_encoder)
    logging.info(f"--- Training finished. Best artifacts are in: {Path(args.output_dir).resolve()} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='processed_data_canonical/train.csv')
    parser.add_argument('--output_dir', type=str, default="output_production")
    parser.add_argument('--date_col', type=str, default='Date')
    parser.add_argument('--target_col', type=str, default='signal')
    parser.add_argument('--target_classes', nargs='+', default=['buy', 'sell', 'keep'])
    parser.add_argument('--numerical_features', nargs='+', default=['open_return', 'high_return', 'low_return', 'close_return', 'Body', 'High-Low'])
    parser.add_argument('--time_features', nargs='+', default=['hour', 'dayofweek'])
    parser.add_argument('--all_features', nargs='+', default=['open_return', 'high_return', 'low_return', 'close_return', 'Body', 'High-Low', 'hour', 'dayofweek'])
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--sequence_length', type=int, default=240)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--hour_emb_dim', type=int, default=16)
    parser.add_argument('--day_emb_dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no-use_amp', dest='use_amp', action='store_false')
    parser.add_argument('--use_torch_compile', action='store_true', default=True)
    parser.add_argument('--no-use_torch_compile', dest='use_torch_compile', action='store_false')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--cost_matrix', nargs='+', default=[[0.0, 1.5, 2.0], [1.0, 0.0, 1.0], [2.0, 1.5, 0.0]])
    parser.add_argument('--cost_initial_lambda', type=float, default=5.0)
    parser.add_argument('--cost_lambda_patience', type=int, default=3)
    args = parser.parse_args()
    main(args)