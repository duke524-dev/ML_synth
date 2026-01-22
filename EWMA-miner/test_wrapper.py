"""
Test wrapper/adapter for offline CRPS tests
Provides generate_simulations interface compatible with offline_crps_simple.py
"""
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forecast_engine import ForecastEngine
from data_fetcher import HermesFetcher

# Patch HermesFetcher to skip HTTP calls in test mode
# This makes testing fully offline (except first-time Pyth data fetch which is cached)
_original_hermes_fetch = HermesFetcher.fetch_anchor_price

def _mock_hermes_fetch(self, asset: str):
    """Skip HTTP call, return None to trigger fallback to latest candle from state"""
    return None

# Apply patch immediately to disable all Hermes HTTP calls during testing
HermesFetcher.fetch_anchor_price = _mock_hermes_fetch

# Global forecast engine instance
_forecast_engine = None
_state_dir = os.path.join(os.path.dirname(__file__), "state")


def _get_forecast_engine():
    """Lazy initialization of forecast engine"""
    global _forecast_engine
    if _forecast_engine is None:
        _forecast_engine = ForecastEngine(state_dir=_state_dir)
    return _forecast_engine


def reset_forecast_engine():
    """Drop the cached ForecastEngine so it will be recreated on next use."""
    global _forecast_engine
    _forecast_engine = None


def set_state_dir(state_dir: str, reset: bool = True):
    """
    Point the ForecastEngine at a specific persistence directory.

    Useful for calibration runs to avoid reusing prior saved state.
    """
    global _state_dir
    _state_dir = state_dir
    if reset:
        reset_forecast_engine()


def update_states_from_price_data(asset: str, prices: list, base_start: datetime, target_time: datetime):
    """
    Update volatility states from price data up to target_time.
    Called by test before each generate_simulations call to simulate real-time state updates.
    
    Args:
        asset: Asset symbol
        prices: List of close prices (1-minute resolution)
        base_start: Start datetime of the price series
        target_time: Update states up to this time
    """
    engine = _get_forecast_engine()
    
    # Handle string datetime inputs
    if isinstance(base_start, str):
        base_start = datetime.fromisoformat(base_start.replace('Z', '+00:00'))
    if base_start.tzinfo is None:
        base_start = base_start.replace(tzinfo=timezone.utc)
    
    if isinstance(target_time, str):
        target_time = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    
    engine.update_states_from_prices(asset, prices, base_start, target_time, time_increment=60)


def generate_simulations(
    asset="BTC",
    start_time: str = "",
    time_increment=300,
    time_length=86400,
    num_simulations=1,
):
    """
    Generate simulated price paths - compatible interface with synth.miner.simulations
    
    This wrapper adapts the EWMA miner's ForecastEngine to the interface
    expected by offline_crps_simple.py
    
    Parameters:
        asset (str): The asset to simulate. Default is 'BTC'.
        start_time (str): The start time of the simulation. Defaults to current time.
        time_increment (int): Time increment in seconds.
        time_length (int): Total time length in seconds.
        num_simulations (int): Number of simulation runs.

    Returns:
        tuple: (start_timestamp, time_increment, [path1, path2, ...])
    """
    if start_time == "":
        raise ValueError("Start time must be provided.")
    
    engine = _get_forecast_engine()
    
    try:
        result = engine.generate_paths(
            asset=asset,
            start_time=start_time,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to generate simulations for asset {asset}: {e}")


# For compatibility with offline_crps_simple.py patching
# The test patches synth.miner.simulations.get_asset_price, but EWMA doesn't use it
# So we provide a dummy function
def get_asset_price(asset: str) -> float:
    """
    Get asset price using the same fallback chain as ForecastEngine
    """
    engine = _get_forecast_engine()
    return engine._get_anchor_price(asset)
