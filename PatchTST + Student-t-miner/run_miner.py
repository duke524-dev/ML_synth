#!/usr/bin/env python3
"""
Entry point for running the DukeMiner1 miner
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import miner
from miner import DukeMiner1
import time

if __name__ == "__main__":
    with DukeMiner1() as miner:
        while True:
            miner.print_info()
            time.sleep(20)
