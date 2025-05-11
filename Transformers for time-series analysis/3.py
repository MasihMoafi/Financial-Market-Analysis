#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""       
Transformer Model Training Script for Forex Data

This script loads forex data, preprocesses it, creates sequences,
trains a Transformer model using PyTorch with optimizations for GPU usage,
and evaluates the model. It incorporates best practices like proper data splitting,
scaling, efficient sequence creation, mixed-precision training, and early stopping.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
# SMOTE is CPU-only and can be slow with large datasets
from imblearn.over_sampling import SMOTE # SMOTE is imported but not used by default
import time
import logging
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
# import shutil # shutil is imported but not used in this version
import h5py
import math
from packaging import version

# Configure deterministic behavior for reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    # PyTorch 1.8+ API
    torch.use_deterministic_algorithms(True, warn_only=True) # warn_only=True can be helpful
except AttributeError:
    # PyTorch 1.7 and older API
    torch.backends.cudnn.deterministic = True
    logging.warning("Using older PyTorch version. Enabling deterministic mode via torch.backends.cudnn.deterministic.")

# Always set these regardless of PyTorch version
torch.backends.cudnn.deterministic = True # Ensure this is set
torch.backends.cudnn.benchmark = False # Disable benchmark for determinism
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
# import random # 'random' module is not explicitly used elsewhere, so this is optional
# random.seed(42) 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- GPU Monitoring ---
def log_gpu_usage(device: torch.device):
    """Logs current GPU memory usage."""
    if device.type == 'cuda':
        try:
            gpu_properties = torch.cuda.get_device_properties(device)
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            total_memory = gpu_properties.total_memory / 1024**3
            utilization = (memory_allocated / total_memory) * 100 if total_memory > 0 else 0

            logging.info(f"GPU: {gpu_properties.name}")
            logging.info(f"Total Memory: {total_memory:.2f} GB")
            logging.info(f"Allocated Memory: {memory_allocated:.2f} GB")
            logging.info(f"Reserved Memory: {memory_reserved:.2f} GB")
            logging.info(f"Utilization (Allocated/Total): {utilization:.2f}%")
        except Exception as e:
            logging.error(f"Could not get GPU details: {e}")
    else:
        logging.warning("CUDA not available. Using CPU. No GPU usage to log.")

# --- Precision & Performance Tweaks for Ampere/Hopper GPUs ---
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_supports_tf32 = any(arch in gpu_name for arch in ['a100', 'a10', 'a30', 'a40', 'a6000', 'h100', 'h800', 'l40'])
    if gpu_supports_tf32:
        logging.info(f"Detected TensorFloat32-capable GPU: {gpu_name}. Enabling TF32.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logging.info(f"GPU {gpu_name} does not explicitly support TF32 acceleration.")
    
    try: # Set float32 matmul precision hint (PyTorch >=1.12)
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

# Improve PyTorch CUDA fragmentation handling
# Declare torch_version_str globally or ensure it's passed if needed in other scopes
torch_version_str = torch.__version__ 
try:
    current_torch_version = version.parse(torch_version_str) 
    if current_torch_version >= version.parse('2.1.0'):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
    else:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
except Exception as e: 
    logging.warning(f"Could not set PYTORCH_CUDA_ALLOC_CONF due to version parsing or other issue: {e}")
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128') # Fallback

# --- Data Handling ---
def create_test_set(file_path: str, months_to_cut: int = 6, test_file_path_str: Optional[str] = None) -> Tuple[str, str]:
    """
    Splits the data chronologically, cutting off the last N months for test set.
    Saves train and test splits to new files to avoid overwriting the original.
    """
    logging.info(f"Creating test set by splitting last {months_to_cut} months of data from {file_path}...")
    
    original_file_path = Path(file_path)
    if test_file_path_str is None:
        test_file_p = original_file_path.parent / f"{original_file_path.stem}_test.csv"
    else:
        test_file_p = Path(test_file_path_str)
    
    train_split_file_p = original_file_path.parent / f"{original_file_path.stem}_train_split.csv"

    try:
        df = pd.read_csv(original_file_path)
        logging.info(f"Loaded data: {df.shape} rows from {original_file_path}")
        
        if df.empty:
            raise ValueError(f"Data file {original_file_path} is empty, cannot create test set")
            
        date_col_candidates = ['date', 'Date', 'datetime', 'Datetime', 'time', 'Time', 'timestamp', 'Timestamp']
        date_col = next((col for col in date_col_candidates if col in df.columns), None)
        if date_col is None: 
            date_col = df.columns[0]
            logging.warning(f"No standard datetime column found. Using first column '{date_col}' as date reference.")
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                logging.info(f"Converted '{date_col}' to datetime.")
            except Exception as e:
                logging.error(f"Failed to convert column '{date_col}' to datetime: {e}. Please ensure it's a valid date format.")
                raise
        
        df = df.sort_values(by=date_col)
        latest_date = df[date_col].max()
        cutoff_date = latest_date - pd.DateOffset(months=months_to_cut)
        
        logging.info(f"Data ranges from {df[date_col].min()} to {latest_date}")
        logging.info(f"Test set cutoff date: {cutoff_date}")
        
        train_data = df[df[date_col] < cutoff_date].copy()
        test_data = df[df[date_col] >= cutoff_date].copy()
        
        if train_data.empty or test_data.empty:
            logging.warning(f"Temporal split resulted in empty train ({train_data.shape[0]}) or test ({test_data.shape[0]}) set. Defaulting to 80/20 chronological-like split of the original data.")
            df_original_for_fallback = pd.read_csv(original_file_path) 
            # Ensure fallback split is also somewhat chronological if shuffle=False
            split_idx_fallback = int(len(df_original_for_fallback) * 0.8)
            train_data = df_original_for_fallback.iloc[:split_idx_fallback].copy()
            test_data = df_original_for_fallback.iloc[split_idx_fallback:].copy()


        train_data.to_csv(train_split_file_p, index=False) 
        test_data.to_csv(test_file_p, index=False)
        
        logging.info(f"Split complete: Training set ({train_data.shape[0]} rows) saved to {train_split_file_p}")
        logging.info(f"Split complete: Test set ({test_data.shape[0]} rows) saved to {test_file_p}")
        
        return str(train_split_file_p), str(test_file_p)
        
    except Exception as e:
        logging.error(f"Error creating test set: {e}", exc_info=True)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully: {df.shape}")
        if df.empty:
            raise ValueError(f"Loaded dataframe from {file_path} is empty.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        raise

def preprocess_and_split_data_temporal(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler, LabelEncoder]:
    """
    Preprocesses data: scales numerical features, encodes target,
    and splits into train and validation sets using temporal splitting.
    """
    logging.info("Preprocessing and splitting data temporally...")

    label_encoder = LabelEncoder()
    # Fit LabelEncoder only on the unique values present in the target column of the provided df
    # to avoid errors if args.target_classes contains classes not in the data.
    try:
        unique_targets_in_df = df[args.target_col].unique()
        label_encoder.fit(unique_targets_in_df)
        # Check if all args.target_classes are covered by the fit encoder. This is more of a sanity check.
        # The critical part is that the encoder can transform the actual data.
        unseen_classes = [cls for cls in args.target_classes if cls not in label_encoder.classes_]
        if unseen_classes:
            logging.warning(f"Target classes specified in args ({args.target_classes}) include classes not found in the actual data's target column '{args.target_col}': {unseen_classes}. LabelEncoder fitted on actual data labels: {list(label_encoder.classes_)}")

        y = label_encoder.transform(df[args.target_col])
    except KeyError:
        logging.error(f"Target column '{args.target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        raise
    except ValueError as e: # This can happen if transform encounters labels not seen during fit
        logging.error(f"Error encoding target variable '{args.target_col}': {e}. This might happen if new, unseen labels are present that were not in the initial fit set. Actual unique labels in data: {df[args.target_col].unique().tolist()}")
        raise
    logging.info(f"Target classes encoded. Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


    missing_features = [f for f in args.all_features if f not in df.columns]
    if missing_features:
        logging.error(f"Missing feature columns specified in args.all_features: {missing_features}. Available columns: {df.columns.tolist()}")
        raise ValueError("Missing required feature columns based on arguments.")
    
    X_df = df[args.all_features].copy() 

    train_size_ratio = 1.0 - args.val_size
    if not (0 < train_size_ratio < 1):
        raise ValueError(f"Invalid train size ratio calculated: {train_size_ratio} (from val_size: {args.val_size}). val_size must be > 0 and < 1.")
    
    split_idx = int(len(X_df) * train_size_ratio)
    # Ensure at least one sample in validation if val_size > 0
    if split_idx == 0 or split_idx >= len(X_df) - (1 if args.val_size > 0 else 0) : 
        raise ValueError(f"Temporal split resulted in an empty training or validation set (or val set too small). Split index: {split_idx}, Data length: {len(X_df)}. Adjust val_size or check data length.")

    X_train_df = X_df.iloc[:split_idx].copy()
    X_val_df = X_df.iloc[split_idx:].copy()
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    logging.info(f"Temporal data split: Train={len(X_train_df)} ({train_size_ratio:.1%}), Val={len(X_val_df)} ({args.val_size:.1%})")
    
    scaler = RobustScaler()
    X_train_scaled_df = X_train_df.copy()
    X_val_scaled_df = X_val_df.copy()

    numerical_features_present = [f for f in args.numerical_features if f in X_train_df.columns]
    if not numerical_features_present:
        logging.warning("No numerical features (from args.numerical_features) were found in the training data. Skipping scaling.")
    else:
        logging.info(f"Scaling numerical features: {numerical_features_present}")
        X_train_scaled_df.loc[:, numerical_features_present] = scaler.fit_transform(X_train_df[numerical_features_present])
        if not X_val_df.empty: 
            X_val_scaled_df.loc[:, numerical_features_present] = scaler.transform(X_val_df[numerical_features_present])
        logging.info("Numerical features scaled using RobustScaler (fit on train set).")
    
    X_train_np = X_train_scaled_df.values
    X_val_np = X_val_scaled_df.values

    return X_train_np, y_train, X_val_np, y_val, scaler, label_encoder

def preprocess_test_data(test_df: pd.DataFrame, scaler: RobustScaler, label_encoder: LabelEncoder, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess test data using fitted scaler and encoder."""
    logging.info("Preprocessing test data...")
    if test_df.empty:
        logging.warning("Test DataFrame is empty. Returning empty arrays.")
        num_features = len(args.all_features) if args.all_features else 0
        return np.empty((0, num_features)), np.array([], dtype=int)
    
    try:
        y_test = label_encoder.transform(test_df[args.target_col])
    except KeyError:
        logging.error(f"Target column '{args.target_col}' not found in test DataFrame. Available columns: {test_df.columns.tolist()}")
        raise
    except ValueError as e:
        logging.error(f"Error encoding test target variable '{args.target_col}': {e}. Check if all target values were seen by the LabelEncoder during its fit on training data. Unique labels in test data: {test_df[args.target_col].unique().tolist()}")
        raise
    
    missing_features = [f for f in args.all_features if f not in test_df.columns]
    if missing_features:
        logging.error(f"Missing feature columns in test DataFrame: {missing_features}. Required by args.all_features. Available: {test_df.columns.tolist()}")
        raise ValueError("Missing required feature columns in test data.")
    X_test_df_features = test_df[args.all_features].copy()
    
    numerical_features_present = [f for f in args.numerical_features if f in X_test_df_features.columns]
    if numerical_features_present:
        X_test_df_features.loc[:, numerical_features_present] = scaler.transform(X_test_df_features[numerical_features_present])
        logging.info(f"Test data numerical features scaled: {numerical_features_present}")
    else:
        logging.warning("No numerical features found in test data to scale (or args.numerical_features is empty).")

    X_test_np = X_test_df_features.values
    
    logging.info(f"Test data preprocessed: X={X_test_np.shape}, y={y_test.shape}")
    return X_test_np, y_test

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences from input data and targets."""
    num_input_features = X.shape[1] if X.ndim == 2 and X.shape[0] > 0 else (X.shape[2] if X.ndim == 3 and X.shape[0] > 0 else 0)

    if X.size == 0 or y.size == 0:
        logging.warning(f"Input X or y is empty for sequence creation. X shape: {X.shape}, y shape: {y.shape}. Returning empty sequences.")
        return np.empty((0, seq_length, num_input_features)), np.empty((0,))

    logging.info(f"Creating sequences with length {seq_length} from X shape {X.shape} and y shape {y.shape}...")
    n_samples, n_features = X.shape

    if n_samples < seq_length:
        logging.warning(f"Dataset length ({n_samples}) < sequence length ({seq_length}). Cannot create sequences. Returning empty sequences.")
        return np.empty((0, seq_length, n_features)), np.empty((0,))

    if not X.flags['C_CONTIGUOUS']:
        X = np.ascontiguousarray(X)

    shape = (n_samples - seq_length + 1, seq_length, n_features)
    strides = (X.strides[0], X.strides[0], X.strides[1]) 
    
    try:
        sequences_X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    except ValueError as e:
        logging.error(f"Error creating sequences with as_strided: {e}. X.shape: {X.shape}, target shape: {shape}, strides: {strides}")
        raise
        
    sequences_y = y[seq_length - 1:]

    if sequences_X.shape[0] != sequences_y.shape[0]:
        raise RuntimeError(f"Mismatch in sequence creation: X sequences ({sequences_X.shape[0]}) != y targets ({sequences_y.shape[0]})")

    logging.info(f"Sequences created: X={sequences_X.shape}, y={sequences_y.shape}")
    return sequences_X.copy(), sequences_y.copy()

class ForexDataset(Dataset):
    """PyTorch Dataset for Forex sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"Input X must be 3-dimensional (samples, seq_length, features), got {X.ndim}D, shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Input y must be 1-dimensional (samples,), got {y.ndim}D, shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples.")
        
        if X.shape[0] == 0: 
            logging.warning("Initializing ForexDataset with zero samples.")
            # Define shape for empty tensor correctly based on expected X structure
            seq_len_dim = X.shape[1] if X.ndim == 3 else 0 
            features_dim = X.shape[2] if X.ndim == 3 else 0
            self.X = torch.empty((0, seq_len_dim, features_dim), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.long) 
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    """Injects positional information into input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0: raise ValueError("d_model must be positive.")
        if not (0 <= dropout <= 1): raise ValueError("dropout must be between 0 and 1.")
        if max_len <= 0: raise ValueError("max_len must be positive.")
        
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        
        # Correct handling for div_term, ensuring it works for odd/even d_model
        div_term_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_term_indices * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0: # Even d_model
            pe[:, 1::2] = torch.cos(position * div_term)
        else: # Odd d_model
            if d_model > 1: # Ensure there's space for cosine terms
                 pe[:, 1::2] = torch.cos(position * div_term[:d_model//2]) # Use correct length for div_term

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length ({seq_len}) exceeds PositionalEncoding max_len ({self.pe.size(1)}). Increase max_len.")
        if x.size(2) != self.pe.size(2):
             raise ValueError(f"Input feature dimension ({x.size(2)}) does not match PositionalEncoding d_model ({self.pe.size(2)}).")
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer."""
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4, k: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_experts < 1: raise ValueError("num_experts must be at least 1.")
        if k < 1 or k > num_experts: raise ValueError(f"k must be between 1 and num_experts (inclusive), got k={k}, num_experts={num_experts}.")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 2), 
                nn.GELU(),
                nn.Linear(input_dim * 2, output_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self._init_parameters()
    
    def _init_parameters(self):
        for expert in self.experts:
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        gate_logits = self.gate(x)
        top_k_values, top_k_indices = torch.topk(gate_logits, self.k, dim=1)
        top_k_weights = torch.softmax(top_k_values, dim=1)
        
        final_output = torch.zeros((batch_size, self.output_dim), dtype=x.dtype, device=x.device)
        
        # This expert routing can be slow due to Python loops.
        # For higher performance, one might explore scatter operations if k is small.
        for i in range(batch_size):
            sample_output = torch.zeros(self.output_dim, dtype=x.dtype, device=x.device)
            for j in range(self.k):
                expert_idx = top_k_indices[i, j].item()
                expert_weight = top_k_weights[i, j]
                expert_output = self.experts[expert_idx](x[i]) # Process one sample at a time
                sample_output += expert_weight * expert_output
            final_output[i] = sample_output
            
        return final_output

class MaxGPUTransformer(nn.Module):
    """Transformer Encoder model for time series classification."""
    def __init__(self, input_dim: int, d_model: int, n_heads: int, num_layers: int, d_ff: int, 
                 num_classes: int, dropout: float = 0.1, max_seq_len: int = 5000,
                 num_experts: int = 4, topk_experts: int = 2, use_moe: bool = True):
        super().__init__()
        if any(param <= 0 for param in [input_dim, d_model, n_heads, num_layers, d_ff, num_classes]):
            raise ValueError("All dimension parameters and num_classes must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not (0 <= dropout <= 1):
            raise ValueError("dropout must be between 0 and 1.")
        
        self.d_model = d_model
        self.use_moe = use_moe
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        
        if use_moe:
            self.mixture_of_experts = MixtureOfExperts(input_dim=d_model, output_dim=d_model, 
                                                       num_experts=num_experts, k=topk_experts, dropout=dropout)
        
        self.output_layer = nn.Linear(d_model, num_classes)
        self._init_parameters()

    def _init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None:
            # Ensure mask is boolean for proper inversion and float for multiplication
            valid_padding_mask = ~src_key_padding_mask 
            expanded_mask = valid_padding_mask.unsqueeze(-1).float()
            masked_x = x * expanded_mask
            sum_x = torch.sum(masked_x, dim=1)
            num_valid_tokens = torch.sum(expanded_mask, dim=1)
            num_valid_tokens = torch.clamp(num_valid_tokens, min=1e-9) # Avoid division by zero with a small epsilon
            x_pooled = sum_x / num_valid_tokens
        else:
            x_pooled = torch.mean(x, dim=1)
        
        if self.use_moe:
            x_pooled = self.mixture_of_experts(x_pooled)
        
        output = self.output_layer(x_pooled)
        return output

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Creates a learning rate scheduler with warmup and cosine decay."""
    if total_steps == 0 : 
        logging.warning("Total steps for LR scheduler is 0. Scheduler will not be effective.")
        def lr_lambda_no_op(_current_step): return 1.0 
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_no_op)

    if total_steps <= warmup_steps:
        logging.warning(f"Total steps ({total_steps}) is less than or equal to warmup steps ({warmup_steps}). Warmup will be a ramp up over all steps.")
        def lr_lambda_short(current_step): # current_step is 0-indexed by LambdaLR
            return float(current_step + 1) / float(max(1, total_steps)) 
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_short)

    def lr_lambda(current_step): # current_step is 0-indexed
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps)) 
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Training Loop ---
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], criterion: nn.Module,
                optimizer: optim.Optimizer, scheduler: Any, device: torch.device, epochs: int, patience: int,
                output_dir: Path, grad_accum_steps: int, args: argparse.Namespace) -> nn.Module:
    """Trains the model with validation, early stopping, and mixed precision.
       Saves a single overwriting checkpoint periodically and the best model at the end."""
    best_val_f1_macro = -1.0 
    epochs_no_improve = 0
    best_model_state_in_memory = None 
    
    total_start_time = time.time()
    use_amp = device.type == 'cuda'
    
    # GradScaler initialization
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logging.info(f"Starting training for {epochs} epochs with patience {patience} (monitoring validation Macro F1)...")
    logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logging.info(f"AMP enabled: {use_amp}")
    logging.info(f"Periodic checkpoint (latest_checkpoint.pt) will be saved every {args.checkpoint_interval} epochs to {output_dir}.")
    
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, val_f1_hist, lr_hist = [], [], [], [], [], []
    best_val_epoch = 0
            
    metrics_path = output_dir / "training_metrics.csv"
    latest_checkpoint_path = output_dir / "latest_checkpoint.pt" 

    for epoch in range(epochs):
        epoch_start_time = time.time()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        model.train()
        train_loss_acum = 0.0 # Accumulator for loss over an epoch
        correct_train_acum = 0
        total_train_samples = 0
        
        # Reset optimizer gradients at the start of each epoch if not accumulating across epochs
        # For standard grad_accum_steps, zero_grad is done after optimizer.step()
        # optimizer.zero_grad() # Typically done within the accumulation loop

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss_per_sample = criterion(outputs, targets) 
                loss_for_step = loss_per_sample / grad_accum_steps if grad_accum_steps > 0 else loss_per_sample
            
            scaler.scale(loss_for_step).backward()
            
            train_loss_acum += loss_per_sample.item() * inputs.size(0) # Accumulate total loss for epoch
            _, predicted = outputs.max(1)
            total_train_samples += targets.size(0)
            correct_train_acum += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler and isinstance(scheduler, optim.lr_scheduler.LambdaLR):
                    scheduler.step() # LambdaLR steps per optimizer step
            
            if (batch_idx + 1) % args.log_interval == 0:
                # Log average loss over micro-batches processed so far in the epoch
                current_avg_loss_so_far = train_loss_acum / total_train_samples if total_train_samples > 0 else 0
                current_train_acc_so_far = 100. * correct_train_acum / total_train_samples if total_train_samples > 0 else 0
                logging.info(f"Epoch {epoch+1}/{epochs} | Batch {(batch_idx+1)}/{len(train_loader)} | Avg Loss (epoch): {current_avg_loss_so_far:.4f} | Avg Acc (epoch): {current_train_acc_so_far:.2f}%")
                sys.stdout.flush()

        avg_train_loss_epoch = train_loss_acum / total_train_samples if total_train_samples > 0 else float('nan')
        train_acc_epoch = 100. * correct_train_acum / total_train_samples if total_train_samples > 0 else 0.0
        train_loss_hist.append(avg_train_loss_epoch)
        train_acc_hist.append(train_acc_epoch)

        current_val_f1_macro = 0.0
        avg_val_loss = float('nan')
        val_acc = float('nan')

        if val_loader and len(val_loader) > 0: # Ensure val_loader is not None and not empty
            model.eval()
            val_loss_epoch_total, correct_val_epoch, total_val_epoch = 0.0, 0, 0
            all_val_preds, all_val_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    val_loss_epoch_total += loss.item() * inputs.size(0) # Sum of losses for all samples
                    _, predicted = outputs.max(1)
                    total_val_epoch += targets.size(0)
                    correct_val_epoch += predicted.eq(targets).sum().item()
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_targets.extend(targets.cpu().numpy())
            
            avg_val_loss = val_loss_epoch_total / total_val_epoch if total_val_epoch > 0 else float('nan')
            val_acc = 100. * correct_val_epoch / total_val_epoch if total_val_epoch > 0 else 0.0
            if total_val_epoch > 0:
                current_val_f1_macro = f1_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
            else:
                current_val_f1_macro = 0.0
        
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(val_acc)
        val_f1_hist.append(current_val_f1_macro)
        lr_hist.append(optimizer.param_groups[0]['lr'])
        
        log_msg = (
            f"Epoch {epoch+1}/{epochs} Summary | Time: {time.time() - epoch_start_time:.2f}s | LR: {lr_hist[-1]:.6e} | "
            f"Train Loss: {avg_train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.2f}%"
        )
        if val_loader and len(val_loader) > 0:
            log_msg += f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro F1: {current_val_f1_macro:.4f}"
        logging.info("-" * 80) 
        logging.info(log_msg)
        logging.info("-" * 80) 
        sys.stdout.flush()
        
        current_metrics_df = pd.DataFrame({
            'epoch': range(1, len(train_loss_hist) + 1),
            'train_loss': train_loss_hist, 'val_loss': val_loss_hist,
            'train_acc': train_acc_hist, 'val_acc': val_acc_hist,
            'val_f1_macro': val_f1_hist, 'lr': lr_hist
        })
        current_metrics_df.to_csv(metrics_path, index=False)

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_f1_macro': best_val_f1_macro, 
                'args': vars(args) # Save args as dict
            }
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, latest_checkpoint_path) # This overwrites the file
            logging.info(f"Saved latest checkpoint to {latest_checkpoint_path} (epoch {epoch + 1})")

        if val_loader and len(val_loader) > 0:
            if current_val_f1_macro > best_val_f1_macro: 
                best_val_f1_macro = current_val_f1_macro
                best_val_epoch = epoch + 1
                epochs_no_improve = 0
                best_model_state_in_memory = model.state_dict() 
                logging.info(f"---> New best validation F1 Macro: {best_val_f1_macro:.4f} at epoch {best_val_epoch} (state stored in memory).")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}. Best epoch was {best_val_epoch} with Val Macro F1: {best_val_f1_macro:.4f}.")
                    break 
            
            if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_val_f1_macro)

    if best_model_state_in_memory: 
        model.load_state_dict(best_model_state_in_memory)
        logging.info(f"Loaded best model state from epoch {best_val_epoch} (Val Macro F1: {best_val_f1_macro:.4f}) for final model.")
    else: 
        logging.info("Using model state from the last trained epoch for final model (no better state found or no validation).")
    
    if os.path.exists(metrics_path): # Check if metrics file exists before plotting
        final_metrics_df = pd.read_csv(metrics_path)
        plt.figure(figsize=(12, 8))
        plt.subplot(2,2,1); plt.plot(final_metrics_df['epoch'], final_metrics_df['train_loss'], label='Train Loss'); plt.plot(final_metrics_df['epoch'], final_metrics_df['val_loss'], label='Val Loss'); plt.legend(); plt.title("Loss")
        plt.subplot(2,2,2); plt.plot(final_metrics_df['epoch'], final_metrics_df['train_acc'], label='Train Acc'); plt.plot(final_metrics_df['epoch'], final_metrics_df['val_acc'], label='Val Acc'); plt.legend(); plt.title("Accuracy")
        plt.subplot(2,2,3); plt.plot(final_metrics_df['epoch'], final_metrics_df['lr']); plt.ylabel('Learning Rate'); plt.xlabel('Epoch'); plt.title("Learning Rate")
        plt.subplot(2,2,4); plt.plot(final_metrics_df['epoch'], final_metrics_df['val_f1_macro']); plt.ylabel('Validation Macro F1'); plt.xlabel('Epoch'); plt.title('Validation Macro F1')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_summary_plots.png'); plt.close()
    else:
        logging.warning(f"Metrics file {metrics_path} not found. Skipping plot generation.")
    
    return model

# --- Evaluation ---
def evaluate_model(model: nn.Module, data_loader: Optional[DataLoader], criterion: nn.Module, device: torch.device, dataset_name: str = "Test") -> Tuple[float, float]:
    if not data_loader or (hasattr(data_loader, 'dataset') and len(data_loader.dataset) == 0):
        logging.warning(f"{dataset_name} loader is empty or None. Skipping evaluation.")
        return float('nan'), float('nan')
    
    model.eval()
    total_loss_sum, correct, total_samples = 0.0, 0, 0
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss_sum += loss.item() * inputs.size(0) # Weighted average for loss
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss_sum / total_samples if total_samples > 0 else float('nan')
    accuracy = 100. * correct / total_samples if total_samples > 0 else 0.0
    logging.info(f"{dataset_name} Results -> Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% ({correct}/{total_samples})")
    return avg_loss, accuracy

def evaluate_model_enhanced(model: nn.Module, data_loader: Optional[DataLoader], criterion: nn.Module, 
                           device: torch.device, class_names: List[str], dataset_name: str = "Test") -> Tuple[float, float, float, float, np.ndarray, str]:
    if not data_loader or (hasattr(data_loader, 'dataset') and len(data_loader.dataset) == 0):
        logging.warning(f"{dataset_name} loader is empty or None. Skipping enhanced evaluation.")
        return float('nan'), 0.0, 0.0, 0.0, np.array([]), "No data for classification report."
    
    model.eval()
    total_loss_sum = 0.0
    all_predictions, all_targets = [], []
    use_amp = device.type == 'cuda'
    total_samples_eval = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss_sum += loss.item() * inputs.size(0)
            total_samples_eval += inputs.size(0)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions_np = np.array(all_predictions)
    all_targets_np = np.array(all_targets)

    if len(all_targets_np) == 0:
        return float('nan'), 0.0, 0.0, 0.0, np.array([]), "Targets array is empty."

    avg_loss = total_loss_sum / total_samples_eval if total_samples_eval > 0 else float('nan')
    accuracy = 100. * (all_predictions_np == all_targets_np).sum() / len(all_targets_np) if len(all_targets_np) > 0 else 0.0
    
    labels_for_metrics = np.arange(len(class_names))

    f1_macro = f1_score(all_targets_np, all_predictions_np, average='macro', zero_division=0, labels=labels_for_metrics)
    f1_weighted = f1_score(all_targets_np, all_predictions_np, average='weighted', zero_division=0, labels=labels_for_metrics)
    cm = confusion_matrix(all_targets_np, all_predictions_np, labels=labels_for_metrics)
    
    # Ensure target_names matches the number of labels used for CM and report
    # If some classes are not in predictions/targets, report might exclude them by default
    # Providing labels and target_names explicitly ensures consistency.
    class_report = classification_report(all_targets_np, all_predictions_np, target_names=class_names, zero_division=0, labels=labels_for_metrics)
    
    return avg_loss, accuracy, f1_macro, f1_weighted, cm, class_report

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    if cm.size == 0:
        logging.warning("Confusion matrix is empty, skipping plot.")
        return
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names)*0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:cm.shape[1]], yticklabels=class_names[:cm.shape[0]])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path); plt.close()

# --- Utility: Dynamically scale batch-size to available GPU memory ---
def auto_adjust_batch_size(seq_length: int, input_dim: int, initial_batch: int, device: torch.device, utilisation_target: float = 0.70, d_model_ref: Optional[int] = None) -> int:
    if device.type != 'cuda' or input_dim == 0 or seq_length == 0:
        return initial_batch

    log_gpu_usage(device)
    total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    bytes_per_float = 4 
    effective_feature_dim = d_model_ref if d_model_ref is not None else input_dim
    
    # Simplified heuristic, may need adjustment based on specific model architecture
    per_sample_bytes_estimate = (seq_length * effective_feature_dim * bytes_per_float * 10) + \
                                (seq_length * seq_length * bytes_per_float * 2) # Attention part slightly higher
    per_sample_bytes_estimate = max(per_sample_bytes_estimate, 1)

    if per_sample_bytes_estimate == 0: return initial_batch

    effective_utilisation = min(utilisation_target, 0.85)
    available_mem_for_batch = total_mem_bytes * effective_utilisation
    max_samples = int(available_mem_for_batch // per_sample_bytes_estimate)
    max_samples = max(max_samples, 1)
    adjusted_batch = min(initial_batch, max_samples)
    
    if adjusted_batch > 1:
        adjusted_batch = 2**int(math.log2(adjusted_batch))
    adjusted_batch = max(adjusted_batch, 1)

    if adjusted_batch < initial_batch:
        logging.info(f"[AutoBatch] Initial: {initial_batch}. VRAM for batch: {available_mem_for_batch/1024**3:.2f}GB. Est. per sample bytes: {per_sample_bytes_estimate}. Max samples: {max_samples}.")
        logging.info(f"[AutoBatch] Reducing batch from {initial_batch} to {adjusted_batch} (~{effective_utilisation*100:.0f}% VRAM target).")
    else:
        logging.info(f"[AutoBatch] Initial batch size {initial_batch} seems feasible. Using it.")
    return adjusted_batch

# --- Main Execution ---
def main(args: argparse.Namespace):
    logging.info("Starting Transformer Training Script")
    logging.info(f"Script arguments: {vars(args)}")
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        log_gpu_usage(device)

    try:
        # If you want to use create_test_set to split your data:
        # 1. Add an argument, e.g., --initial_dataset_file
        # 2. Call: train_file_path, test_file_path = create_test_set(args.initial_dataset_file, months_to_cut=6)
        # 3. Then use train_file_path and test_file_path below.
        # For now, assuming args.train_file and args.test_file are already prepared.
        train_df = load_data(args.train_file)
        test_df = load_data(args.test_file)
        
        X_train_np, y_train, X_val_np, y_val, scaler, label_encoder = preprocess_and_split_data_temporal(train_df, args)
        X_test_np, y_test = preprocess_test_data(test_df, scaler, label_encoder, args)

    except Exception as e:
        logging.error(f"Data loading or preprocessing failed: {e}", exc_info=True)
        return

    num_classes = len(label_encoder.classes_) # Number of classes based on fitted LabelEncoder
    input_dim = len(args.all_features) 
    logging.info(f"Input dimension (number of features): {input_dim}")
    logging.info(f"Number of classes: {num_classes} (from LabelEncoder: {list(label_encoder.classes_)})")


    if args.auto_batch_size:
        original_batch_size = args.batch_size
        args.batch_size = auto_adjust_batch_size(args.sequence_length, input_dim, args.batch_size, device, 
                                                 utilisation_target=0.7, d_model_ref=args.d_model)
        if args.batch_size < original_batch_size // 2 and original_batch_size > 1:
             logging.warning(f"Auto-adjusted batch size ({args.batch_size}) is significantly smaller than requested ({original_batch_size}).")

    class_weights = None
    if len(y_train) > 0:
        class_indices, class_counts = np.unique(y_train, return_counts=True)
        # Ensure class_indices from y_train match what label_encoder expects (0 to N-1)
        # And that all num_classes are represented for weight calculation.
        expected_indices = np.arange(num_classes)
        current_class_counts = np.zeros(num_classes, dtype=int)
        for idx, count in zip(class_indices, class_counts):
            if idx < num_classes : # Ensure index is within expected range
                 current_class_counts[idx] = count
        
        if np.all(current_class_counts > 0): # All classes must have samples
            weights_val = len(y_train) / (num_classes * current_class_counts)
            class_weights = torch.tensor(weights_val, dtype=torch.float32).to(device)
            logging.info(f"Calculated class weights: {class_weights.cpu().numpy().round(4)}")
            y_train_counts_map = {label_encoder.inverse_transform([i])[0]: current_class_counts[i] for i in expected_indices}
            logging.info(f"Based on training class counts: {y_train_counts_map}")
        else:
            logging.warning(f"Training data missing some classes or classes have 0 samples (Counts: {current_class_counts} for {num_classes} classes). Using unweighted loss.")
    else:
        logging.warning("Training data (y_train) is empty. Using unweighted loss.")
    
    X_train_np_for_seq, y_train_for_seq = X_train_np, y_train

    X_train_seq, y_train_seq = create_sequences(X_train_np_for_seq, y_train_for_seq, args.sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_np, y_val, args.sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_np, y_test, args.sequence_length)

    if X_train_seq.size == 0:
        logging.error("Training sequence creation resulted in empty data. Cannot proceed.")
        return

    train_dataset = ForexDataset(X_train_seq, y_train_seq)
    val_dataset = ForexDataset(X_val_seq, y_val_seq) if X_val_seq.size > 0 else None
    test_dataset = ForexDataset(X_test_seq, y_test_seq) if X_test_seq.size > 0 else None

    num_workers = min(os.cpu_count() or 1, args.num_workers) 
    logging.info(f"Using {num_workers} workers for DataLoaders.")

    g = torch.Generator()
    if args.random_state is not None: g.manual_seed(args.random_state)

    # Use the globally parsed torch_version
    global current_torch_version 
    use_persistent_workers = num_workers > 0 and current_torch_version >= version.parse('1.8.0')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == 'cuda', num_workers=num_workers, drop_last=True, generator=g,
        persistent_workers=use_persistent_workers, 
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == 'cuda', num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None
    ) if val_dataset and len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == 'cuda', num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None
    ) if test_dataset and len(test_dataset) > 0 else None

    model = MaxGPUTransformer(
        input_dim=input_dim, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers,
        d_ff=args.d_ff, num_classes=num_classes, dropout=args.dropout, 
        max_seq_len=max(5000, args.sequence_length + 100), # Ample max_seq_len
        num_experts=args.num_experts, topk_experts=args.topk_experts, use_moe=args.use_moe
    ).to(device)

    if args.compile_model and hasattr(torch, 'compile') and device.type == 'cuda' and current_torch_version >= version.parse('2.0.0'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logging.info("Model compiled successfully with torch.compile(mode='reduce-overhead').")
        except Exception as e:
            logging.warning(f"torch.compile() failed – proceeding without compilation. Error: {e}", exc_info=True)
    
    if args.gradient_checkpointing and device.type == 'cuda':
        if hasattr(model.transformer_encoder, 'layers'):
            for i, layer_module in enumerate(model.transformer_encoder.layers):
                if hasattr(torch.utils.checkpoint, 'checkpoint_wrapper'):
                    try:
                        # This replaces the layer with a checkpointed version
                        model.transformer_encoder.layers[i] = torch.utils.checkpoint.checkpoint_wrapper(layer_module)
                    except Exception as e:
                        logging.warning(f"Failed to apply torch.utils.checkpoint.checkpoint_wrapper to layer {i}: {e}")
            logging.info("Attempted to enable gradient checkpointing on Transformer encoder layers using checkpoint_wrapper.")
        elif hasattr(model.transformer_encoder, 'gradient_checkpointing_enable'): # Some models might have a built-in method
             model.transformer_encoder.gradient_checkpointing_enable()
             logging.info("Enabled gradient checkpointing via model.transformer_encoder.gradient_checkpointing_enable().")
        else:
            logging.warning("Gradient checkpointing requested, but couldn't apply to model.transformer_encoder automatically.")


    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1 if num_classes > 1 else 0.0)
    logging.info(f"Using CrossEntropyLoss with weights: {class_weights is not None}, label_smoothing: {criterion.label_smoothing}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)
    
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    # Calculate total_effective_steps more robustly
    if len(train_loader) > 0 and args.grad_accum_steps > 0:
        steps_per_epoch = len(train_loader) // args.grad_accum_steps
        if len(train_loader) % args.grad_accum_steps != 0: # Account for partial last accumulation step
            steps_per_epoch +=1
        total_effective_steps = steps_per_epoch * args.epochs
    else:
        total_effective_steps = 0
        
    warmup_steps = int(total_effective_steps * args.warmup_ratio) if total_effective_steps > 0 else 0

    if args.use_warmup and warmup_steps > 0 and total_effective_steps > 0:
        scheduler = get_lr_scheduler(optimizer, warmup_steps, total_effective_steps)
        logging.info(f"Using Cosine decay scheduler with {warmup_steps} warmup steps over {total_effective_steps} total effective steps.")
    elif val_loader and len(val_loader) > 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, threshold=0.001, verbose=True)
        logging.info(f"Using ReduceLROnPlateau scheduler, monitoring validation F1 Macro (mode='max').")
    else:
        scheduler = None
        logging.info("No LR scheduler will be used (no warmup, or no validation set for ReduceLROnPlateau).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, args.early_stopping_patience, output_dir, args.grad_accum_steps, args)

    if val_loader and len(val_loader) > 0:
        evaluate_model_enhanced(model, val_loader, criterion, device, list(label_encoder.classes_), "Validation (on Final Model)")
    
    if test_loader and len(test_loader) > 0:
        test_loss, test_acc, test_f1, test_f1_weighted, cm, class_report = evaluate_model_enhanced(model, test_loader, criterion, device, list(label_encoder.classes_), "Test (on Final Model)")
        logging.info(f"FINAL TEST RESULTS -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}% | F1 Macro: {test_f1:.4f} | F1 Weighted: {test_f1_weighted:.4f}")
        logging.info(f"FINAL Classification Report (Test):\n{class_report}")
        if cm.size > 0:
            plot_confusion_matrix(cm, list(label_encoder.classes_), output_dir / "confusion_matrix_final_test.png")

    model_save_path = output_dir / "transformer_model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model state_dict saved to {model_save_path}")

    model_save_path_h5 = output_dir / "transformer_model_final.h5"
    try:
        with h5py.File(model_save_path_h5, 'w') as f:
            state_dict = model.state_dict()
            for key, value in state_dict.items():
                f.create_dataset(key, data=value.cpu().numpy())
        logging.info(f"Final model state_dict also saved to HDF5 format: {model_save_path_h5}")
    except Exception as e:
        logging.error(f"Failed to save model state_dict to HDF5: {e}", exc_info=True)

    if 'scaler' in locals() and 'label_encoder' in locals() : # Ensure they exist
        import joblib
        joblib.dump(scaler, output_dir / "scaler.joblib")
        joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
        logging.info("Scaler and LabelEncoder saved.")
    else:
        logging.warning("Scaler or LabelEncoder not found in local scope, skipping saving them.")
        
    logging.info(f"All results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model on Forex data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_file', type=str, default='train.csv', help='Path to training CSV.')
    parser.add_argument('--test_file', type=str, default='test.csv', help='Path to test CSV.')
    parser.add_argument('--numerical_features', nargs='+', default=['Open', 'High', 'Low', 'Close', 'Body', 'High-Low'], help='Numerical feature columns.')
    parser.add_argument('--binary_features', nargs='+', default=["Is_Doji", "Is_Spike", "Is_Long_Shadow", "gap"], help='Binary feature columns.')
    parser.add_argument('--time_features', nargs='+', default=['time_sin', 'time_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos'], help='Pre-calculated time feature columns.')
    parser.add_argument('--target_col', type=str, default='signal', help='Target column name.')
    parser.add_argument('--target_classes', nargs='+', default=['buy', 'sell', 'keep'], help='Target classes for LabelEncoder fitting if data is empty or for reference.')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion (used by legacy random split, ignored by temporal_split).') # Kept for reference if legacy split is used
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set proportion (temporal split from end of training data). Must be >0 and <1.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for seeding.')
    
    parser.add_argument('--d_model', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of encoder layers.')
    parser.add_argument('--d_ff', type=int, default=0, help='Feed-forward dimension. If 0, defaults to 4*d_model. Original default was 2048 for d_model 512.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    
    # BooleanOptionalAction changed to store_true/store_false for wider Python compatibility
    parser.add_argument('--use_moe', action='store_true', default=True, help='Use Mixture of Experts layer (default: True).')
    parser.add_argument('--no_use_moe', action='store_false', dest='use_moe', help='Disable Mixture of Experts layer.')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts in MoE.')
    parser.add_argument('--topk_experts', type=int, default=2, help='Top-k experts for MoE.')
    
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size per step (before grad_accum).')
    parser.add_argument('--grad_accum_steps', type=int, default=8, help='Gradient accumulation steps.')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty).')
    
    parser.add_argument('--use_warmup', action='store_true', default=True, help='Use learning rate warmup (default: True).')
    parser.add_argument('--no_use_warmup', action='store_false', dest='use_warmup', help='Disable learning rate warmup.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio of total effective steps.')
    parser.add_argument('--lr_patience', type=int, default=5, help='LR scheduler (ReduceLROnPlateau) patience in epochs.')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience in epochs (default: 3).') # CHANGED DEFAULT TO 3
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval in micro-batches.')
    
    parser.add_argument('--output_dir', type=str, default='transformer_a100_results', help='Output directory for results.')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers. Will be capped by os.cpu_count().')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if CUDA is available.')
    
    parser.add_argument('--auto_batch_size', action='store_true', default=True, help='Automatically try to scale batch size to fit GPU memory (default: True).')
    parser.add_argument('--no_auto_batch_size', action='store_false', dest='auto_batch_size', help='Disable auto batch sizing.')
    
    parser.add_argument('--compile_model', action='store_true', default=True, help='Use torch.compile for model acceleration (PyTorch >=2.0) (default: True).')
    parser.add_argument('--no_compile_model', action='store_false', dest='compile_model', help='Disable torch.compile.')
    
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False, help='Enable gradient checkpointing (default: False).')
    parser.add_argument('--no_gradient_checkpointing', action='store_false', dest='gradient_checkpointing', help='Disable gradient checkpointing.')

    parser.add_argument('--checkpoint_interval', type=int, default=5, 
                        help='Save an overwriting checkpoint (latest_checkpoint.pt) every N epochs (default: 5).')

    if 'ipykernel' in sys.argv[0] or 'IPython' in sys.modules:
        logging.info("IPython/Jupyter environment detected. Using default args from parser. Modify 'args' object in notebook for custom values.")
        args = parser.parse_args([]) 
        # Example overrides for notebook:
        # args.train_file = 'your_notebook_train.csv'
        # args.early_stopping_patience = 3 
    else:
        logging.info("Running as standard Python script.")
        args = parser.parse_args()

    # Combine Feature Lists
    args.all_features = list(set(args.numerical_features + args.binary_features + args.time_features))
    temp_numerical_features = list(args.numerical_features) 
    for tf_cat in [args.time_features, args.binary_features]: # Ensure binary features are also considered for numerical scaling if specified
        for tf in tf_cat:
            if tf in args.all_features and tf not in temp_numerical_features:
                 # Decision: only add time_features to numerical_features, binary features are usually not scaled.
                 # If binary features (0/1) need to be scaled, they should be explicitly in numerical_features list.
                 if tf_cat is args.time_features:
                    temp_numerical_features.append(tf)
    args.numerical_features = list(set(temp_numerical_features))


    logging.info(f"Final feature list (all_features): {sorted(args.all_features)}")
    logging.info(f"Numerical features for scaling: {sorted(args.numerical_features)}")

    if args.d_ff == 0: 
        args.d_ff = 4 * args.d_model
        logging.info(f"d_ff was 0, calculated to 4*d_model = {args.d_ff}")
    # Removed the specific d_ff == 2048 check for more general behavior based on d_ff=0
    else:
        logging.info(f"Using provided d_ff: {args.d_ff}")

    try:
        if not (0 < args.val_size < 1):
            raise ValueError("--val_size must be between 0 (exclusive) and 1 (exclusive) for temporal split.")
        if args.d_model <= 0 or args.n_heads <=0 or args.num_layers <=0 :
            raise ValueError("d_model, n_heads, num_layers must be positive.")
        if args.d_model % args.n_heads != 0:
            raise ValueError(f"d_model ({args.d_model}) must be divisible by n_heads ({args.n_heads}).")
        if not (0 <= args.dropout <= 1):
            raise ValueError("dropout must be between 0 and 1.")
        if args.use_moe:
            if args.num_experts < 1: raise ValueError("num_experts must be at least 1 if use_moe is True.")
            if args.topk_experts < 1 or args.topk_experts > args.num_experts:
                raise ValueError(f"topk_experts must be between 1 and num_experts ({args.num_experts}) if use_moe is True.")
        if args.grad_accum_steps < 1: raise ValueError("grad_accum_steps must be at least 1.")
        if args.checkpoint_interval < 1 : raise ValueError("checkpoint_interval must be at least 1.")
        if args.epochs <1 : raise ValueError("epochs must be at least 1.")
        if args.early_stopping_patience <1 : raise ValueError("early_stopping_patience must be at least 1.")

    except ValueError as e:
        if 'ipykernel' in sys.argv[0] or 'IPython' in sys.modules: 
            logging.error(f"Argument validation failed: {e}", exc_info=True)
        else: 
            parser.error(str(e))
        sys.exit(1)

    main(args)

# In[ ]:
