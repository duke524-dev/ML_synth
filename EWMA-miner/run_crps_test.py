#!/usr/bin/env python3
"""
Test runner for EWMA miner with offline CRPS test
Patches synth.miner.simulations to use EWMA miner instead of baseline
"""
import sys
import os

# Add EWMA-miner to path
ewma_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ewma_dir)

# Add project root to path (for offline_crps_simple.py)
project_root = os.path.dirname(ewma_dir)
sys.path.insert(0, project_root)

# Import EWMA test wrapper functions BEFORE importing anything that uses synth.miner.simulations
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
