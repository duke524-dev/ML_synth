"""
Offline calibration script for sigma_scale adjustment
"""
import logging
import os
import json
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict
import numpy as np

from .config import (
    HF_CALIBRATION_HORIZONS, LF_CALIBRATION_HORIZONS,
    HF_CALIBRATION_WEIGHTS, LF_CALIBRATION_WEIGHTS,
    HF_CALIBRATION_ALPHA, LF_CALIBRATION_ALPHA,
    SIGMA_SCALE_MIN, SIGMA_SCALE_MAX,
    HF_MAX_MOVE, LF_MAX_MOVE,
    HF_MIN_COUNT, LF_MIN_COUNT,
)
from .forecast_engine import ForecastEngine
from .data_fetcher import BenchmarksFetcher, parse_benchmarks_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_realized_dispersion(
    timestamps: list, prices: list, horizon_steps: int, resolution_minutes: int
) -> float:
    """
    Compute realized dispersion of cumulative returns at horizon
    
    Args:
        timestamps: List of timestamps
        prices: List of prices
        horizon_steps: Number of steps ahead
        resolution_minutes: Resolution in minutes
    Returns:
        Standard deviation of cumulative returns
    """
    if len(prices) < horizon_steps + 1:
        return np.nan
    
    returns = []
    for i in range(len(prices) - horizon_steps):
        start_price = prices[i]
        end_price = prices[i + horizon_steps]
        if start_price > 0 and end_price > 0:
            cum_return = np.log(end_price / start_price)
            returns.append(cum_return)
    
    if len(returns) < 10:
        return np.nan
    
    return np.std(returns)


def compute_predicted_dispersion(
    engine: ForecastEngine, asset: str, start_time: datetime,
    time_increment: int, horizon_steps: int, num_samples: int = 200
) -> float:
    """
    Compute predicted dispersion by sampling from model
    
    Args:
        engine: ForecastEngine instance
        start_time: Start time for prediction
        horizon_steps: Number of steps ahead
        num_samples: Number of samples to draw
    Returns:
        Standard deviation of cumulative returns
    """
    time_length = horizon_steps * time_increment
    
    # Generate multiple paths
    start_time_str = start_time.isoformat()
    paths_list = []
    
    for _ in range(num_samples // 100 + 1):  # Batch in groups of 100
        try:
            result = engine.generate_paths(
                asset=asset,
                start_time=start_time_str,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=min(100, num_samples - len(paths_list)),
            )
            
            if result and len(result) > 2:
                # Extract paths (skip timestamp and increment)
                paths = result[2:]
                for path in paths:
                    if len(path) > horizon_steps:
                        paths_list.append(path)
        except Exception as e:
            logger.warning(f"Error generating paths: {e}")
            continue
        
        if len(paths_list) >= num_samples:
            break
    
    if len(paths_list) < 10:
        return np.nan
    
    # Compute cumulative returns
    returns = []
    for path in paths_list:
        if len(path) > horizon_steps and path[0] > 0:
            start_price = path[0]
            end_price = path[horizon_steps]
            if end_price > 0:
                cum_return = np.log(end_price / start_price)
                returns.append(cum_return)
    
    if len(returns) < 10:
        return np.nan
    
    return np.std(returns)


def calibrate_model(
    engine: ForecastEngine,
    model_key: str,
    assets: list,
    artifacts_dir: str,
    is_hf: bool,
):
    """
    Calibrate sigma_scale for a model
    
    Args:
        engine: ForecastEngine instance
        model_key: "HF", "LF-crypto", or "LF-equity"
        assets: List of assets to calibrate
        artifacts_dir: Directory for artifacts
        is_hf: Whether this is HF model
    """
    logger.info(f"Starting calibration for {model_key}")
    
    # Load current calibration
    calib_file = os.path.join(artifacts_dir, "current", "calibration.json")
    current_scales = {}
    if os.path.exists(calib_file):
        with open(calib_file, 'r') as f:
            current_scales = json.load(f)
    
    if model_key not in current_scales:
        current_scales[model_key] = {asset: 1.0 for asset in assets}
    
    # Determine parameters
    if is_hf:
        calibration_window = timedelta(days=3)
        horizons = HF_CALIBRATION_HORIZONS
        weights = HF_CALIBRATION_WEIGHTS
        alpha = HF_CALIBRATION_ALPHA
        max_move = HF_MAX_MOVE
        min_count = HF_MIN_COUNT
        resolution = 1
        time_increment = 60
        sample_interval_min = 10  # Every 10-15 minutes
        sample_interval_max = 15
    else:
        calibration_window = timedelta(days=10)
        horizons = LF_CALIBRATION_HORIZONS
        weights = LF_CALIBRATION_WEIGHTS
        alpha = LF_CALIBRATION_ALPHA
        max_move = LF_MAX_MOVE
        min_count = LF_MIN_COUNT
        resolution = 5
        time_increment = 300
        sample_interval_min = 30  # Every 30-60 minutes
        sample_interval_max = 60
    
    # Sample timestamps across calibration window
    end_time = datetime.now(timezone.utc)
    start_time = end_time - calibration_window
    
    # Generate sample times
    sample_times = []
    current = start_time
    while current < end_time:
        sample_times.append(current)
        interval = np.random.uniform(sample_interval_min, sample_interval_max)
        current += timedelta(minutes=interval)
    
    logger.info(f"Sampling {len(sample_times)} timestamps")
    
    # Fetch historical data
    fetcher = BenchmarksFetcher()
    
    # Calibrate each asset
    new_scales = {}
    
    for asset in assets:
        logger.info(f"Calibrating {asset}")
        
        ratios_per_horizon = {h: [] for h in horizons}
        
        for sample_time in sample_times:
            # Fetch historical data around this time
            from_ts = int((sample_time - timedelta(days=1)).timestamp())
            to_ts = int((sample_time + timedelta(hours=25)).timestamp())
            
            data = fetcher.fetch_history(asset, resolution, from_ts, to_ts)
            if data is None:
                continue
            
            timestamps, prices = parse_benchmarks_data(data)
            if len(prices) < max(horizons) + 100:
                continue
            
            # Find index closest to sample_time
            sample_ts = int(sample_time.timestamp())
            closest_idx = min(
                range(len(timestamps)),
                key=lambda i: abs(timestamps[i] - sample_ts)
            )
            
            if closest_idx + max(horizons) >= len(prices):
                continue
            
            # Compute realized dispersion for each horizon
            for horizon in horizons:
                if closest_idx + horizon >= len(prices):
                    continue
                
                # Realized
                realized_std = compute_realized_dispersion(
                    timestamps[closest_idx:closest_idx + horizon + 1],
                    prices[closest_idx:closest_idx + horizon + 1],
                    horizon, resolution
                )
                
                if np.isnan(realized_std) or realized_std <= 0:
                    continue
                
                # Predicted
                predicted_std = compute_predicted_dispersion(
                    engine, asset, sample_time, time_increment, horizon
                )
                
                if np.isnan(predicted_std) or predicted_std <= 0:
                    continue
                
                # Ratio
                ratio = realized_std / predicted_std
                ratios_per_horizon[horizon].append(ratio)
        
        # Combine ratios using weights
        if not any(ratios_per_horizon.values()):
            logger.warning(f"No valid ratios for {asset}, keeping current scale")
            new_scales[asset] = current_scales[model_key].get(asset, 1.0)
            continue
        
        # Weighted average in log space
        log_ratios = []
        total_weight = 0.0
        
        for horizon, ratios in ratios_per_horizon.items():
            if not ratios:
                continue
            weight = weights.get(horizon, 0.0)
            if weight > 0:
                median_ratio = np.median(ratios)
                if median_ratio > 0:
                    log_ratios.append(np.log(median_ratio) * weight)
                    total_weight += weight
        
        if total_weight == 0:
            logger.warning(f"No valid weighted ratios for {asset}")
            new_scales[asset] = current_scales[model_key].get(asset, 1.0)
            continue
        
        # Average in log space
        log_ratio_avg = sum(log_ratios) / total_weight
        new_scale = np.exp(log_ratio_avg)
        
        # EMA update
        current_scale = current_scales[model_key].get(asset, 1.0)
        log_current = np.log(current_scale)
        log_new = alpha * log_ratio_avg + (1 - alpha) * log_current
        new_scale = np.exp(log_new)
        
        # Clamp and limit move
        new_scale = max(SIGMA_SCALE_MIN, min(SIGMA_SCALE_MAX, new_scale))
        
        # Limit per-run move
        max_scale = current_scale * (1 + max_move)
        min_scale = current_scale * (1 - max_move)
        new_scale = max(min_scale, min(max_scale, new_scale))
        
        # Check count
        total_count = sum(len(ratios) for ratios in ratios_per_horizon.values())
        if total_count < min_count:
            logger.warning(
                f"Insufficient samples for {asset} ({total_count} < {min_count}), "
                "keeping current scale"
            )
            new_scales[asset] = current_scale
        else:
            new_scales[asset] = new_scale
            logger.info(
                f"{asset}: {current_scale:.4f} -> {new_scale:.4f} "
                f"(count={total_count})"
            )
    
    # Update calibration
    current_scales[model_key] = new_scales
    
    # Write to staging
    staging_file = os.path.join(artifacts_dir, "staging", "calibration.json")
    os.makedirs(os.path.dirname(staging_file), exist_ok=True)
    with open(staging_file, 'w') as f:
        json.dump(current_scales, f, indent=2)
    
    logger.info(f"Calibration written to staging: {staging_file}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate sigma scales")
    parser.add_argument("--model", choices=["HF", "LF-crypto", "LF-equity"], required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    
    # Initialize engine
    engine = ForecastEngine(artifacts_dir=args.artifacts_dir)
    
    # Determine assets
    if args.model == "HF":
        assets = ["BTC", "ETH", "SOL", "XAU"]
        is_hf = True
    elif args.model == "LF-crypto":
        assets = ["BTC", "ETH", "SOL", "XAU"]
        is_hf = False
    else:  # LF-equity
        assets = ["SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
        is_hf = False
    
    calibrate_model(engine, args.model, assets, args.artifacts_dir, is_hf)
    
    # Publish (atomic move)
    staging_file = os.path.join(args.artifacts_dir, "staging", "calibration.json")
    current_file = os.path.join(args.artifacts_dir, "current", "calibration.json")
    
    if os.path.exists(staging_file):
        os.makedirs(os.path.dirname(current_file), exist_ok=True)
        os.rename(staging_file, current_file)
        logger.info(f"Published calibration to {current_file}")


if __name__ == "__main__":
    main()
