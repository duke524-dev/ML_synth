"""
Test script for Fixed Hybrid miner
"""
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forecast_engine import ForecastEngine

def test_generate_paths():
    """Test path generation"""
    print("Testing Fixed Hybrid miner path generation...")
    
    # Initialize forecast engine
    engine = ForecastEngine(state_dir="state", model_dir="models")
    
    # Test HF prompt
    print("\n=== Testing HF prompt (BTC, 1-minute, 1 hour) ===")
    result = engine.generate_paths(
        asset="BTC",
        start_time="2026-01-23T12:00:00Z",
        time_increment=60,
        time_length=3600,  # 1 hour
        num_simulations=100,
    )
    
    if result:
        start_ts, time_increment, paths = result
        print(f"✓ Generated {len(paths)} paths")
        print(f"  Start timestamp: {start_ts}")
        print(f"  Time increment: {time_increment}s")
        print(f"  First path length: {len(paths[0])} steps")
        print(f"  First path sample: {paths[0][:3]}...")
    else:
        print("✗ Failed to generate paths")
    
    # Test LF prompt
    print("\n=== Testing LF prompt (BTC, 5-minute, 24 hours) ===")
    result = engine.generate_paths(
        asset="BTC",
        start_time="2026-01-23T12:00:00Z",
        time_increment=300,
        time_length=86400,  # 24 hours
        num_simulations=100,
    )
    
    if result:
        start_ts, time_increment, paths = result
        print(f"✓ Generated {len(paths)} paths")
        print(f"  Start timestamp: {start_ts}")
        print(f"  Time increment: {time_increment}s")
        print(f"  First path length: {len(paths[0])} steps")
        print(f"  First path sample: {paths[0][:3]}...")
    else:
        print("✗ Failed to generate paths")
    
    # Test with different assets
    print("\n=== Testing different assets ===")
    for asset in ["ETH", "SOL", "XAU"]:
        result = engine.generate_paths(
            asset=asset,
            start_time="2026-01-23T12:00:00Z",
            time_increment=60,
            time_length=3600,
            num_simulations=10,
        )
        if result:
            print(f"✓ {asset}: Generated {len(result[2])} paths")
        else:
            print(f"✗ {asset}: Failed")
    
    print("\n=== Test complete ===")

if __name__ == "__main__":
    test_generate_paths()
