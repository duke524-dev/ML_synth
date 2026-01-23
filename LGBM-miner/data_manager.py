"""
Data manager for training data with recency weighting support
"""
import logging
import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from collections import deque
import threading

from volatility_state import OHLCBar

logger = logging.getLogger(__name__)


class DataManager:
    """Manages 1-minute and 5-minute OHLC data with resampling and recency weighting"""
    
    def __init__(self, asset: str):
        self.asset = asset
        self.lock = threading.RLock()
        
        # 1-minute data (rolling buffer, ~10,080 bars for 7 days)
        self.data_1m: deque = deque(maxlen=10080)
        
        # 5-minute data (derived, ~2,016 bars)
        self.data_5m: deque = deque(maxlen=2016)
        
        # Last 5-minute aggregation state
        self._agg_5m: Optional[dict] = None
    
    def add_1m_bar(self, bar: OHLCBar):
        """Add 1-minute bar and update 5-minute aggregation"""
        with self.lock:
            self.data_1m.append(bar)
            self._update_5m_aggregation(bar)
    
    def _update_5m_aggregation(self, bar_1m: OHLCBar):
        """Update 5-minute aggregation state"""
        # Round timestamp to 5-minute boundary
        bar_ts_5m = (bar_1m.timestamp // 300) * 300
        
        if self._agg_5m is None or self._agg_5m['timestamp'] != bar_ts_5m:
            # New 5-minute block: finalize previous if exists
            if self._agg_5m is not None:
                agg_bar = OHLCBar(
                    timestamp=self._agg_5m['timestamp'],
                    open=self._agg_5m['open'],
                    high=self._agg_5m['high'],
                    low=self._agg_5m['low'],
                    close=self._agg_5m['close']
                )
                self.data_5m.append(agg_bar)
            
            # Start new aggregation
            self._agg_5m = {
                'timestamp': bar_ts_5m,
                'open': bar_1m.open,
                'high': bar_1m.high,
                'low': bar_1m.low,
                'close': bar_1m.close,
            }
        else:
            # Update existing aggregation
            self._agg_5m['high'] = max(self._agg_5m['high'], bar_1m.high)
            self._agg_5m['low'] = min(self._agg_5m['low'], bar_1m.low)
            self._agg_5m['close'] = bar_1m.close
    
    def get_1m_bars(self, start_ts: Optional[int] = None, 
                    end_ts: Optional[int] = None) -> List[OHLCBar]:
        """Get 1-minute bars in range"""
        with self.lock:
            bars = list(self.data_1m)
            
            if start_ts is not None:
                bars = [b for b in bars if b.timestamp >= start_ts]
            if end_ts is not None:
                bars = [b for b in bars if b.timestamp <= end_ts]
            
            return sorted(bars, key=lambda x: x.timestamp)
    
    def get_5m_bars(self, start_ts: Optional[int] = None,
                    end_ts: Optional[int] = None) -> List[OHLCBar]:
        """Get 5-minute bars in range (including finalized aggregation)"""
        with self.lock:
            # Finalize current aggregation if exists
            if self._agg_5m is not None:
                agg_bar = OHLCBar(
                    timestamp=self._agg_5m['timestamp'],
                    open=self._agg_5m['open'],
                    high=self._agg_5m['high'],
                    low=self._agg_5m['low'],
                    close=self._agg_5m['close']
                )
                bars = list(self.data_5m) + [agg_bar]
            else:
                bars = list(self.data_5m)
            
            if start_ts is not None:
                bars = [b for b in bars if b.timestamp >= start_ts]
            if end_ts is not None:
                bars = [b for b in bars if b.timestamp <= end_ts]
            
            return sorted(bars, key=lambda x: x.timestamp)
    
    def get_latest_bar(self, resolution: int = 1) -> Optional[OHLCBar]:
        """Get latest bar"""
        with self.lock:
            if resolution == 1:
                if len(self.data_1m) > 0:
                    return self.data_1m[-1]
            else:
                bars = self.get_5m_bars()
                if len(bars) > 0:
                    return bars[-1]
            return None
    
    def get_latest_close(self, resolution: int = 1) -> Optional[float]:
        """Get latest close price"""
        bar = self.get_latest_bar(resolution)
        return bar.close if bar is not None else None
    
    def get_prev_bar(self, resolution: int = 1) -> Optional[OHLCBar]:
        """Get previous bar (for volatility updates)"""
        with self.lock:
            if resolution == 1:
                if len(self.data_1m) > 1:
                    return self.data_1m[-2]
            else:
                bars = self.get_5m_bars()
                if len(bars) > 1:
                    return bars[-2]
            return None
    
    def compute_recency_weights(
        self,
        timestamps: List[int],
        half_life_days: float,
        reference_time: Optional[datetime] = None
    ) -> List[float]:
        """
        Compute exponential recency weights for training data
        
        Args:
            timestamps: List of Unix timestamps
            half_life_days: Half-life in days for exponential decay
            reference_time: Reference time for computing weights (default: latest timestamp)
        
        Returns:
            List of weights (sum to 1.0)
        """
        if not timestamps:
            return []
        
        if reference_time is None:
            # Use latest timestamp as reference
            reference_ts = max(timestamps)
        else:
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
            reference_ts = int(reference_time.timestamp())
        
        # Compute decay rate: lambda = exp(-ln(2) / half_life_days)
        # For each day, weight decays by factor of 0.5
        half_life_seconds = half_life_days * 86400
        decay_rate = math.log(2) / half_life_seconds
        
        # Compute weights: w(t) = exp(-decay_rate * (reference_ts - t))
        weights = []
        for ts in timestamps:
            age_seconds = reference_ts - ts
            weight = math.exp(-decay_rate * age_seconds)
            weights.append(weight)
        
        # Normalize to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights