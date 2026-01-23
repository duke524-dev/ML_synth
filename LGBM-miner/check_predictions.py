#!/usr/bin/env python3
"""
Simple diagnostic to check prediction values
"""
import sys
import os
import numpy as np
import pickle
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forecast_engine import ForecastEngine
from config import TOKEN_MAP, HF_ASSETS

# Initialize engine
state_dir = os.path.join(os.path.dirname(__file__), "state")
model_dir = os.path.join(os.path.dirname(__file__), "models")
engine = ForecastEngine(state_dir=state_dir, model_dir=model_dir)

# Check a model file directly
asset = "BTC"
is_hf = asset in HF_ASSETS
model_key = (asset, is_hf)

print(f"Checking {asset} ({'HF' if is_hf else 'LF'})...")
print(f"Model key: {model_key}")

# Check if predictor exists
predictor = engine.predictors.get(model_key)
if predictor:
    print(f"✓ Predictor exists")
    print(f"  Leads: {predictor.leads}")
    print(f"  Models loaded: {len(predictor.trainer.models)}")
    
    # Check each model
    for lead in predictor.leads:
        model = predictor.trainer.models.get(lead)
        if model:
            print(f"  ✓ Model for lead {lead}s: loaded")
            # Try to get feature names
            features = predictor.trainer.feature_names.get(lead, [])
            print(f"    Features: {len(features)} features")
            if features:
                print(f"    First 5 features: {features[:5]}")
        else:
            print(f"  ✗ Model for lead {lead}s: MISSING")
else:
    print(f"✗ No predictor found for {model_key}")

# Check if we can get anchor price
try:
    anchor_price = engine._get_anchor_price(asset)
    print(f"\nAnchor price for {asset}: ${anchor_price:,.2f}")
except Exception as e:
    print(f"\nError getting anchor price: {e}")

# Check data manager
data_manager = engine.data_managers.get(asset)
if data_manager:
    bars_1m = data_manager.get_1m_bars()
    print(f"\nData manager for {asset}:")
    print(f"  1m bars: {len(bars_1m)}")
    if bars_1m:
        print(f"  Latest bar: timestamp={bars_1m[-1].timestamp}, close=${bars_1m[-1].close:,.2f}")
else:
    print(f"\n✗ No data manager for {asset}")

# Try to generate a simple prediction
print(f"\n--- Testing prediction generation ---")
test_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
start_iso = test_time.isoformat()

try:
    # Check if we have enough data
    if data_manager and len(bars_1m) >= 50:
        print(f"Generating test prediction...")
        result = engine.generate_paths(
            asset=asset,
            start_time=start_iso,
            time_increment=300,  # 5 minutes
            time_length=3600,  # 1 hour
            num_simulations=10,  # Just 10 for testing
        )
        
        if result:
            predictions_list = list(result)
            start_ts = predictions_list[0]
            time_inc = predictions_list[1]
            paths = predictions_list[2:]
            
            print(f"\nPrediction result:")
            print(f"  start_ts: {start_ts}")
            print(f"  time_increment: {time_inc}")
            print(f"  num_paths: {len(paths)}")
            if paths:
                print(f"  path_length: {len(paths[0])}")
                
                # Check first path
                first_path = np.array(paths[0])
                print(f"\nFirst path statistics:")
                print(f"  min: ${first_path.min():,.2f}")
                print(f"  max: ${first_path.max():,.2f}")
                print(f"  mean: ${first_path.mean():,.2f}")
                print(f"  std: ${first_path.std():,.2f}")
                print(f"  first value: ${first_path[0]:,.2f}")
                print(f"  last value: ${first_path[-1]:,.2f}")
                
                # Check if paths are constant
                path_stds = [np.std(np.array(p)) for p in paths[:5]]
                print(f"\nPath variability (std of first 5 paths):")
                for i, std in enumerate(path_stds):
                    print(f"  path {i}: std=${std:,.2f}")
                
                if max(path_stds) < 0.01:
                    print("\n⚠️  WARNING: Paths appear to be constant!")
                
                # Check if values are reasonable for BTC
                if first_path.mean() < 1000 or first_path.mean() > 200000:
                    print(f"\n⚠️  WARNING: Mean price ${first_path.mean():,.2f} seems unreasonable for BTC!")
    else:
        print(f"⚠️  Not enough data: {len(bars_1m) if data_manager else 0} bars (need at least 50)")
        
except Exception as e:
    print(f"\n❌ Error generating prediction: {e}")
    import traceback
    traceback.print_exc()
