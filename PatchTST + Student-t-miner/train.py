"""
Offline training script for PatchTST models
"""
import logging
import os
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import PatchTSTModel
from data_fetcher import BenchmarksFetcher, parse_benchmarks_data
from features import extract_hf_features, extract_lf_features
from config import HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series training"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        asset_ids: np.ndarray,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.asset_ids = torch.LongTensor(asset_ids)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.asset_ids[idx]


def student_t_nll(mu: torch.Tensor, log_sigma: torch.Tensor, nu: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Student-t negative log-likelihood
    
    Args:
        mu: [B, H] mean
        log_sigma: [B, H] log scale
        nu: [B] degrees of freedom
        targets: [B, H] observed returns
    Returns:
        NLL loss
    """
    sigma = torch.exp(log_sigma)
    
    # Standardize
    z = (targets - mu) / sigma
    
    # Log probability
    from torch.distributions import StudentT
    
    # Expand nu to match shape
    nu_expanded = nu.unsqueeze(1).expand(-1, targets.size(1))  # [B, H]
    
    dist = StudentT(df=nu_expanded, loc=mu, scale=sigma)
    log_prob = dist.log_prob(targets)
    
    # Negative log-likelihood
    nll = -log_prob.mean()
    
    return nll


def prepare_training_data(
    assets: List[str],
    start_date: datetime,
    end_date: datetime,
    resolution: int,
    is_hf: bool,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data from historical Benchmarks data
    
    Returns:
        (features, targets, asset_ids)
    """
    fetcher = BenchmarksFetcher()
    
    all_features = []
    all_targets = []
    all_asset_ids = []
    
    asset_to_id = {asset: i for i, asset in enumerate(sorted(assets))}
    
    for asset in assets:
        logger.info(f"Fetching data for {asset}")
        
        from_ts = int(start_date.timestamp())
        to_ts = int(end_date.timestamp())
        
        data = fetcher.fetch_history(asset, resolution, from_ts, to_ts)
        if data is None:
            logger.warning(f"No data for {asset}")
            continue
        
        timestamps, prices = parse_benchmarks_data(data)
        if len(prices) < horizon + 100:
            logger.warning(f"Insufficient data for {asset}")
            continue
        
        # Extract features
        if is_hf:
            features = extract_hf_features(timestamps, prices)
        else:
            features = extract_lf_features(timestamps, prices, asset)
        
        if len(features) < horizon + 1:
            continue
        
        # Create targets (future returns)
        targets = []
        for i in range(len(features) - horizon):
            # Compute cumulative log returns over horizon
            start_price = prices[i]
            end_price = prices[i + horizon]
            if start_price > 0 and end_price > 0:
                cum_return = np.log(end_price / start_price)
                targets.append(cum_return)
            else:
                targets.append(0.0)
        
        targets = np.array(targets)
        
        # Align features and targets
        min_len = min(len(features) - horizon, len(targets))
        features = features[:min_len]
        targets = targets[:min_len]
        
        # Expand targets to horizon length (for now, use same value)
        # In practice, you might want to predict step-by-step returns
        targets_expanded = np.tile(targets[:, np.newaxis], (1, horizon))
        
        asset_ids = np.full(len(features), asset_to_id[asset])
        
        all_features.append(features)
        all_targets.append(targets_expanded)
        all_asset_ids.append(asset_ids)
        
        logger.info(f"{asset}: {len(features)} samples")
    
    if not all_features:
        raise ValueError("No training data collected")
    
    # Concatenate
    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    asset_ids = np.concatenate(all_asset_ids, axis=0)
    
    logger.info(f"Total training samples: {len(features)}")
    
    return features, targets, asset_ids


def train_model(
    model: PatchTSTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str = "cpu",
) -> PatchTSTModel:
    """Train model"""
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, targets, asset_ids in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            asset_ids = asset_ids.to(device)
            
            optimizer.zero_grad()
            
            mu, log_sigma, nu = model(features, asset_ids)
            
            loss = student_t_nll(mu, log_sigma, nu, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets, asset_ids in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                asset_ids = asset_ids.to(device)
                
                mu, log_sigma, nu = model(features, asset_ids)
                loss = student_t_nll(mu, log_sigma, nu, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
    
    return model


def compute_normalization_stats(features: np.ndarray) -> dict:
    """Compute mean and std for normalization"""
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    return {"mean": mean.squeeze(0).tolist(), "std": std.squeeze(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description="Train PatchTST model")
    parser.add_argument("--model", choices=["HF", "LF-crypto", "LF-equity"], required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-days", type=int, default=180)  # 6 months
    args = parser.parse_args()
    
    # Determine parameters
    if args.model == "HF":
        assets = sorted(list(HF_ASSETS))
        resolution = 1
        horizon = 60
        num_features = 15
        num_assets = 4
        d_model = 192
        num_layers = 6
        is_hf = True
    elif args.model == "LF-crypto":
        assets = sorted(list(LF_CRYPTO_ASSETS))
        resolution = 5
        horizon = 288
        num_features = 14
        num_assets = 4
        d_model = 256
        num_layers = 8
        is_hf = False
    else:  # LF-equity
        assets = sorted(list(LF_EQUITY_ASSETS))
        resolution = 5
        horizon = 288
        num_features = 17
        num_assets = 5
        d_model = 256
        num_layers = 8
        is_hf = False
    
    # Prepare data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.train_days)
    
    logger.info(f"Preparing training data from {start_date} to {end_date}")
    features, targets, asset_ids = prepare_training_data(
        assets, start_date, end_date, resolution, is_hf, horizon
    )
    
    # Compute normalization stats
    logger.info("Computing normalization statistics")
    norm_stats = compute_normalization_stats(features)
    
    # Normalize features
    mean = np.array(norm_stats["mean"])
    std = np.array(norm_stats["std"])
    features_norm = (features - mean) / (std + 1e-8)
    
    # Split train/val
    split_idx = int(0.9 * len(features))
    train_features = features_norm[:split_idx]
    train_targets = targets[:split_idx]
    train_asset_ids = asset_ids[:split_idx]
    
    val_features = features_norm[split_idx:]
    val_targets = targets[split_idx:]
    val_asset_ids = asset_ids[split_idx:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_features, train_targets, train_asset_ids)
    val_dataset = TimeSeriesDataset(val_features, val_targets, val_asset_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = PatchTSTModel(
        num_features=num_features,
        d_model=d_model,
        num_layers=num_layers,
        num_assets=num_assets,
        horizon=horizon,
    )
    
    # Train
    logger.info("Starting training")
    model = train_model(model, train_loader, val_loader, args.epochs)
    
    # Save model
    staging_dir = os.path.join(args.artifacts_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    model_file = os.path.join(staging_dir, f"{args.model.lower()}_model.pt")
    torch.save(model.state_dict(), model_file)
    logger.info(f"Model saved to {model_file}")
    
    # Save normalization stats
    stats_file = os.path.join(staging_dir, "normalization_stats.json")
    stats_data = {args.model: norm_stats}
    
    # Load existing if present
    if os.path.exists(stats_file):
        import json
        with open(stats_file, 'r') as f:
            existing = json.load(f)
        existing[args.model] = norm_stats
        stats_data = existing
    
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    logger.info(f"Normalization stats saved to {stats_file}")
    
    # Publish (atomic move)
    current_dir = os.path.join(args.artifacts_dir, "current")
    os.makedirs(current_dir, exist_ok=True)
    
    current_model_file = os.path.join(current_dir, f"{args.model.lower()}_model.pt")
    if os.path.exists(model_file):
        os.rename(model_file, current_model_file)
        logger.info(f"Published model to {current_model_file}")
    
    current_stats_file = os.path.join(current_dir, "normalization_stats.json")
    if os.path.exists(stats_file):
        # Merge with existing
        if os.path.exists(current_stats_file):
            with open(current_stats_file, 'r') as f:
                existing = json.load(f)
            existing.update(stats_data)
            stats_data = existing
        with open(current_stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        logger.info(f"Published stats to {current_stats_file}")


if __name__ == "__main__":
    main()
