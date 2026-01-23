#!/usr/bin/env python3
"""
Debug script to check what predictions are being generated
"""
import sys
import os
import numpy as np
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_wrapper import generate_simulations, _get_forecast_engine, update_states_from_price_data
from synth.validator.price_data_provider import PriceDataProvider
from synth.validator import prompt_config

# Test with BTC, LOW_FREQUENCY
asset = "BTC"
prompt_cfg = prompt_config.LOW_FREQUENCY

# Get a test time
test_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
start_iso = test_time.isoformat()

# Fetch real prices for comparison
provider = PriceDataProvider()
from synth.db.models import ValidatorRequest
vreq = ValidatorRequest(
    asset=asset,
    start_time=test_time,
    time_increment=60,
    time_length=24 * 3600,
)
real_prices = provider.fetch_data(vreq)

# Update state with previous day's data (warm-up)
prev_day = test_time.replace(day=14)
prev_vreq = ValidatorRequest(
    asset=asset,
    start_time=prev_day,
    time_increment=60,
    time_length=24 * 3600,
)
prev_prices = provider.fetch_data(prev_vreq)

print(f"Warming up state for {asset}...")
update_states_from_price_data(
    asset=asset,
    prices=prev_prices,
    base_start=prev_day,
    target_time=test_time
)

# Update state up to test time
print(f"Updating state to {start_iso}...")
update_states_from_price_data(
    asset=asset,
    prices=real_prices,
    base_start=test_time,
    target_time=test_time
)

# Generate predictions
print(f"\nGenerating predictions for {asset}...")
print(f"  start_time: {start_iso}")
print(f"  time_increment: {prompt_cfg.time_increment}s")
print(f"  time_length: {prompt_cfg.time_length}s")
print(f"  num_simulations: {prompt_cfg.num_simulations}")

try:
    predictions = generate_simulations(
        asset=asset,
        start_time=start_iso,
        time_increment=prompt_cfg.time_increment,
        time_length=prompt_cfg.time_length,
        num_simulations=prompt_cfg.num_simulations,
    )
    
    # Extract paths
    predictions_list = list(predictions)
    start_ts = predictions_list[0]
    time_inc = predictions_list[1]
    paths = predictions_list[2:]
    
    print(f"\nPredictions generated:")
    print(f"  start_ts: {start_ts}")
    print(f"  time_increment: {time_inc}")
    print(f"  num_paths: {len(paths)}")
    print(f"  path_length: {len(paths[0]) if paths else 0}")
    
    if paths:
        # Check first path
        first_path = paths[0]
        print(f"\nFirst path (first 10 values):")
        for i, price in enumerate(first_path[:10]):
            print(f"  t={i}: {price:.2f}")
        
        print(f"\nFirst path (last 10 values):")
        for i, price in enumerate(first_path[-10:]):
            print(f"  t={len(first_path)-10+i}: {price:.2f}")
        
        # Statistics
        all_prices = np.array([p for path in paths for p in path])
        print(f"\nPrediction statistics:")
        print(f"  min: {all_prices.min():.2f}")
        print(f"  max: {all_prices.max():.2f}")
        print(f"  mean: {all_prices.mean():.2f}")
        print(f"  median: {np.median(all_prices):.2f}")
        print(f"  std: {all_prices.std():.2f}")
        
        # Check if paths are constant (fallback)
        path_stds = [np.std(path) for path in paths[:10]]
        print(f"\nPath variability (std of first 10 paths):")
        for i, std in enumerate(path_stds):
            print(f"  path {i}: std={std:.2f}")
        
        if max(path_stds) < 0.01:
            print("\n⚠️  WARNING: Paths appear to be constant (fallback paths being used)")
        
        # Compare with real prices
        print(f"\nReal prices (first 10, downsampled to {prompt_cfg.time_increment}s):")
        factor = prompt_cfg.time_increment // 60
        real_downsampled = real_prices[::factor][:10]
        for i, price in enumerate(real_downsampled):
            print(f"  t={i}: {price:.2f}")
        
        # Check engine state
        engine = _get_forecast_engine()
        predictor = engine.predictors.get((asset, False))  # LF
        if predictor:
            print(f"\nPredictor state:")
            print(f"  asset: {predictor.asset}")
            print(f"  is_hf: {predictor.is_hf}")
            print(f"  leads: {predictor.leads}")
            print(f"  models loaded: {len(predictor.trainer.models)} models")
            for lead, model in predictor.trainer.models.items():
                print(f"    lead {lead}s: ✓")
        else:
            print("\n⚠️  WARNING: No predictor found for (BTC, False)")
        
except Exception as e:
    print(f"\n❌ Error generating predictions: {e}")
    import traceback
    traceback.print_exc()
