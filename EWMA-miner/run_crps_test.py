#!/usr/bin/env python3
"""
Test runner for EWMA miner with offline CRPS test
Patches synth.miner.simulations to use EWMA miner instead of baseline
"""
import sys
import os
import shutil

# Add EWMA-miner to path
ewma_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ewma_dir)

# Add project root to path (for offline_crps_simple.py)
project_root = os.path.dirname(ewma_dir)
sys.path.insert(0, project_root)

# Import EWMA test wrapper functions BEFORE importing anything that uses synth.miner.simulations
from test_wrapper import generate_simulations, get_asset_price, reset_forecast_engine

# Reset state directory to ensure clean start
# This prevents stale state from previous runs from affecting test results
state_dir = os.path.join(ewma_dir, "state")
if os.path.exists(state_dir):
    print(f"[run_crps_test] Clearing state directory: {state_dir}")
    shutil.rmtree(state_dir)
os.makedirs(state_dir, exist_ok=True)

# Reset the forecast engine to ensure fresh initialization
reset_forecast_engine()

# Patch synth.miner.simulations module BEFORE it gets imported by offline_crps_simple
import synth.miner.simulations
synth.miner.simulations.generate_simulations = generate_simulations
synth.miner.simulations.get_asset_price = get_asset_price

# Now import and run the offline CRPS test
# The test will use our patched functions
from offline_CRPS_test.offline_crps_simple import main

if __name__ == "__main__":
    sys.exit(main())
