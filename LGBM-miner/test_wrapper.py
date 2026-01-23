"""
Test wrapper/adapter for offline CRPS tests
Provides generate_simulations interface compatible with offline_crps_simple.py
"""
import sys
import os
import numpy as np
from datetime import datetime, timezone, timedelta

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
_model_dir = os.path.join(os.path.dirname(__file__), "models")


def _get_forecast_engine():
    """Lazy initialization of forecast engine"""
    global _forecast_engine
    if _forecast_engine is None:
        _forecast_engine = ForecastEngine(state_dir=_state_dir, model_dir=_model_dir)
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
    
    # Get data manager and volatility states for this asset
    data_manager = engine.data_managers.get(asset)
    vol_state_1m = engine.vol_states_1m.get(asset)
    vol_state_5m = engine.vol_states_5m.get(asset)
    
    if data_manager is None or vol_state_1m is None:
        return
    
    # Import here to avoid circular imports
    from volatility_state import OHLCBar
    from persistence import StatePersistence
    
    # Calculate how many minutes to process
    minutes_to_process = int((target_time - base_start).total_seconds() / 60)
    minutes_to_process = min(minutes_to_process, len(prices))
    
    # Process prices up to target_time
    for i in range(minutes_to_process):
        current_time = base_start.replace(second=0, microsecond=0) + timedelta(minutes=i)
        if current_time > target_time:
            break
        
        price = prices[i]
        if price is None or (isinstance(price, float) and (np.isnan(price) or price <= 0)):
            continue
        
        # Create OHLC bar (using close price for all OHLC if only close is available)
        bar = OHLCBar(
            timestamp=int(current_time.timestamp()),
            open=price,
            high=price,
            low=price,
            close=price
        )
        
        # Add to data manager
        data_manager.add_1m_bar(bar)
        
        # Update volatility state (need previous bar for gap detection)
        prev_bar = data_manager.get_prev_bar(resolution=1) if i > 0 else None
        vol_state_1m.update_with_bar(bar, prev_bar)
        
        # Update 5-minute volatility state (every 5 minutes)
        if (i + 1) % 5 == 0:
            # Resample 5 1-minute bars into one 5-minute bar
            bars_1m = data_manager.get_1m_bars()
            if len(bars_1m) >= 5:
                recent_5 = bars_1m[-5:]
                bar_5m = OHLCBar(
                    timestamp=recent_5[0].timestamp,
                    open=recent_5[0].open,
                    high=max(b.high for b in recent_5),
                    low=min(b.low for b in recent_5),
                    close=recent_5[-1].close
                )
                prev_bar_5m = data_manager.get_prev_bar(resolution=5)
                vol_state_5m.update_with_bar(bar_5m, prev_bar_5m)
    
    # Save updated states
    persistence = StatePersistence(_state_dir)
    persistence.save_asset_state(
        asset=asset,
        vol_state_1m=vol_state_1m,
        vol_state_5m=vol_state_5m,
        data_manager=data_manager
    )


def generate_simulations(
    asset="BTC",
    start_time: str = "",
    time_increment=300,
    time_length=86400,
    num_simulations=1,
):
    """
    Generate simulated price paths - compatible interface with synth.miner.simulations
    
    This wrapper adapts the LGBM miner's ForecastEngine to the interface
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
# The test patches synth.miner.simulations.get_asset_price, but LGBM doesn't use it directly
# So we provide a function that uses the ForecastEngine's anchor price logic
def get_asset_price(asset: str) -> float:
    """
    Get asset price using the same fallback chain as ForecastEngine
    """
    engine = _get_forecast_engine()
    return engine._get_anchor_price(asset)
