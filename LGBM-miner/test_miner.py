"""
Test script for LGBM miner
"""
import sys
import os
import glob
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forecast_engine import ForecastEngine

def check_models_exist(model_dir: str, asset: str, is_hf: bool):
    """Check if models exist for an asset
    
    Returns:
        (exists: bool, found_models: list)
    """
    from config import HF_LEADS, LF_LEADS
    
    leads = HF_LEADS if is_hf else LF_LEADS
    freq = "HF" if is_hf else "LF"
    
    model_files = []
    for lead in leads:
        pattern = os.path.join(model_dir, f"{asset}_{freq}_{lead}.pkl")
        if glob.glob(pattern):
            model_files.append(lead)
    
    return len(model_files) > 0, model_files

def test_generate_paths():
    """Test path generation"""
    print("Testing LGBM miner path generation...")
    
    # Check if models exist
    model_dir = "models"
    asset = "BTC"
    is_hf = True
    
    models_exist, found_models = check_models_exist(model_dir, asset, is_hf)
    
    if not models_exist:
        print("\n⚠️  WARNING: No trained models found!")
        print(f"   Looking for models in: {model_dir}/")
        print(f"   Expected pattern: {asset}_{'HF' if is_hf else 'LF'}_*.pkl")
        print("\n   Please train models first:")
        print(f"   python train_models.py --asset {asset}")
        print("\n   Or train all assets:")
        print("   python train_models.py")
        print("\n   The test will continue but may use fallback paths...\n")
    else:
        print(f"\n✓ Found {len(found_models)} model(s) for {asset}")
        print(f"   Models: {found_models}\n")
    
    # Initialize forecast engine
    engine = ForecastEngine(state_dir="test_state", model_dir=model_dir)
    
    # Test parameters
    asset = "BTC"
    start_time = datetime.now(timezone.utc).isoformat()
    time_increment = 300  # 5 minutes
    time_length = 86400  # 24 hours
    num_simulations = 10  # Small number for testing
    
    print(f"\nGenerating paths for {asset}:")
    print(f"  start_time: {start_time}")
    print(f"  time_increment: {time_increment}s")
    print(f"  time_length: {time_length}s")
    print(f"  num_simulations: {num_simulations}")
    
    try:
        result = engine.generate_paths(
            asset=asset,
            start_time=start_time,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
        )
        
        if result is None:
            print("ERROR: No result returned")
            return False
        
        # Check result format
        if not isinstance(result, tuple):
            print(f"ERROR: Result is not a tuple: {type(result)}")
            return False
        
        if len(result) < 3:
            print(f"ERROR: Result tuple too short: {len(result)}")
            return False
        
        start_ts, time_inc, *paths = result
        
        print(f"\nResult:")
        print(f"  start_timestamp: {start_ts}")
        print(f"  time_increment: {time_inc}")
        print(f"  number of paths: {len(paths)}")
        
        if len(paths) != num_simulations:
            print(f"WARNING: Expected {num_simulations} paths, got {len(paths)}")
        
        # Check path lengths
        expected_steps = (time_length // time_increment) + 1
        for i, path in enumerate(paths[:3]):  # Check first 3 paths
            if len(path) != expected_steps:
                print(f"WARNING: Path {i} has length {len(path)}, expected {expected_steps}")
            else:
                print(f"  Path {i}: length={len(path)}, first={path[0]:.2f}, last={path[-1]:.2f}")
        
        print("\n✓ Test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_generate_paths()
    sys.exit(0 if success else 1)