"""
State persistence with atomic msgpack snapshots
"""
import logging
import os
import tempfile
import shutil
from typing import Dict, Optional
import msgpack

from volatility_state import VolatilityState
from data_manager import DataManager

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles atomic state persistence per asset"""
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
    
    def save_asset_state(
        self,
        asset: str,
        vol_state_1m: VolatilityState,
        vol_state_5m: VolatilityState,
        data_manager: DataManager
    ):
        """
        Save state for one asset atomically
        
        Uses atomic write: write to temp file, then rename
        """
        try:
            # Prepare state dict
            state = {
                'asset': asset,
                'vol_state_1m': vol_state_1m.to_dict(),
                'vol_state_5m': vol_state_5m.to_dict(),
                'data_1m': self._serialize_bars(data_manager.data_1m),
                'data_5m': self._serialize_bars(data_manager.data_5m),
                'agg_5m': data_manager._agg_5m,
            }
            
            # Write to temp file first
            state_file = os.path.join(self.state_dir, f"{asset}.msgpack")
            temp_file = state_file + ".tmp"
            
            with open(temp_file, 'wb') as f:
                msgpack.pack(state, f)
            
            # Atomic rename
            shutil.move(temp_file, state_file)
            
            logger.debug(f"Saved state for {asset}")
            
        except Exception as e:
            logger.error(f"Error saving state for {asset}: {e}", exc_info=True)
            # Clean up temp file on error
            temp_file = os.path.join(self.state_dir, f"{asset}.msgpack.tmp")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def load_asset_state(
        self,
        asset: str,
        vol_state_1m: VolatilityState,
        vol_state_5m: VolatilityState,
        data_manager: DataManager
    ) -> bool:
        """
        Load state for one asset
        
        Returns:
            True if loaded successfully, False otherwise
        """
        state_file = os.path.join(self.state_dir, f"{asset}.msgpack")
        
        if not os.path.exists(state_file):
            logger.debug(f"No saved state for {asset}")
            return False
        
        try:
            with open(state_file, 'rb') as f:
                state = msgpack.unpack(f, raw=False)
            
            # Restore volatility states
            if 'vol_state_1m' in state:
                vol_state_1m.from_dict(state['vol_state_1m'])
            if 'vol_state_5m' in state:
                vol_state_5m.from_dict(state['vol_state_5m'])
            
            # Restore data
            if 'data_1m' in state:
                data_manager.data_1m = self._deserialize_bars(state['data_1m'])
            if 'data_5m' in state:
                data_manager.data_5m = self._deserialize_bars(state['data_5m'])
            if 'agg_5m' in state:
                data_manager._agg_5m = state['agg_5m']
            
            logger.info(f"Loaded state for {asset}")
            return True
            
        except Exception as e:
            logger.warning(f"Error loading state for {asset}: {e}")
            return False
    
    def _serialize_bars(self, bars) -> list:
        """Serialize OHLC bars to list of tuples"""
        from volatility_state import OHLCBar
        return [
            (bar.timestamp, bar.open, bar.high, bar.low, bar.close)
            for bar in bars
        ]
    
    def _deserialize_bars(self, data: list):
        """Deserialize bars from list of tuples"""
        from collections import deque
        from volatility_state import OHLCBar
        
        bars = deque()
        for item in data:
            if len(item) == 5:
                bars.append(OHLCBar(
                    timestamp=item[0],
                    open=item[1],
                    high=item[2],
                    low=item[3],
                    close=item[4]
                ))
        return bars
    
    def save_all_states(
        self,
        assets: Dict[str, Dict]
    ):
        """Save states for all assets"""
        for asset, states in assets.items():
            self.save_asset_state(
                asset=asset,
                vol_state_1m=states['vol_state_1m'],
                vol_state_5m=states['vol_state_5m'],
                data_manager=states['data_manager']
            )