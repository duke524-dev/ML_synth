#!/usr/bin/env python3
"""
Test runner for LGBM miner with offline CRPS test
Patches synth.miner.simulations to use LGBM miner instead of baseline
"""
import sys
import os

# Add LGBM-miner to path
lgbm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, lgbm_dir)

# Add project root to path (for offline_crps_simple.py)
project_root = os.path.dirname(lgbm_dir)
sys.path.insert(0, project_root)

# Import LGBM test wrapper functions BEFORE importing anything that uses synth.miner.simulations
from test_wrapper import generate_simulations, get_asset_price

# Patch synth.miner.simulations module BEFORE it gets imported by offline_crps_simple
import synth.miner.simulations
synth.miner.simulations.generate_simulations = generate_simulations
synth.miner.simulations.get_asset_price = get_asset_price

# Now import and run the offline CRPS test
# The test will use our patched functions
from offline_CRPS_test.offline_crps_simple import main

if __name__ == "__main__":
    sys.exit(main())
