"""
EWMA volatility state management with gap detection
"""
import logging
import math
from typing import Optional
from dataclasses import dataclass

from config import (
    HALF_LIFE_1M, HALF_LIFE_5M, VARIANCE_MIXING_ALPHA
)

logger = logging.getLogger(__name__)


@dataclass
class OHLCBar:
    """OHLC bar data structure"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float


class VolatilityState:
    """Maintains EWMA volatility state for one asset at one resolution"""
    
    def __init__(self, asset: str, resolution: int):
        """
        Args:
            asset: Asset symbol
            resolution: 1 or 5 (minutes)
        """
        self.asset = asset
        self.resolution = resolution
        self.dt = resolution * 60  # seconds
        
        # Get half-life for this resolution
        if resolution == 1:
            half_life_seconds = HALF_LIFE_1M.get(asset, 120 * 60)
        else:
            half_life_seconds = HALF_LIFE_5M.get(asset, 12 * 3600)
        
        # Compute lambda: lambda = exp(-ln(2) * dt / H)
        self.lambda_val = math.exp(-math.log(2) * self.dt / half_life_seconds)
        
        # State: conditional variance h_t
        # Initial variance: typical 1-min log returns are ~0.0001 (0.01%), so r^2 ~ 0.00000001
        # For 5-min, typical r^2 ~ 0.00000025. Initial should be small but non-zero.
        self.h = 0.00001  # Initial variance (much smaller than 0.001)
        
        # Last update timestamp
        self.last_update_ts: Optional[int] = None
        
        logger.info(
            f"VolatilityState({asset}, {resolution}m): "
            f"half_life={half_life_seconds/3600:.1f}h, lambda={self.lambda_val:.6f}, "
            f"initial_h={self.h:.8f}"
        )
    
    def compute_variance_proxy(self, bar: OHLCBar, prev_close: float) -> float:
        """
        Compute variance proxy using mixing rule:
        v_t = alpha * r_t^2 + (1 - alpha) * v_range
        """
        # Log return
        if prev_close > 0:
            log_return = math.log(bar.close / prev_close)
            r_squared = log_return ** 2
        else:
            r_squared = 0.0
        
        # Range-based variance (Parkinson)
        if bar.high > 0 and bar.low > 0 and bar.high >= bar.low:
            log_range = math.log(bar.high / bar.low)
            v_range = (log_range ** 2) / (4 * math.log(2))
        else:
            v_range = r_squared  # Fallback
        
        # Mix
        v_t = VARIANCE_MIXING_ALPHA * r_squared + (1 - VARIANCE_MIXING_ALPHA) * v_range
        
        return max(v_t, 1e-10)  # Ensure positive
    
    def update_with_bar(self, bar: OHLCBar, prev_bar: Optional[OHLCBar] = None):
        """
        Update volatility state with OHLC bar, handling gaps
        
        Args:
            bar: Current OHLC bar
            prev_bar: Previous OHLC bar (None for first bar)
        """
        # Detect gaps
        if self.last_update_ts is not None:
            expected_next_ts = self.last_update_ts + self.dt
            if bar.timestamp > expected_next_ts:
                # Gap detected
                gap_seconds = bar.timestamp - expected_next_ts
                num_missing_bars = int(gap_seconds / self.dt)
                if num_missing_bars > 0:
                    logger.debug(
                        f"Gap detected for {self.asset} ({self.resolution}m): "
                        f"{num_missing_bars} missing bars"
                    )
                    self.decay(num_missing_bars)
        
        # Get previous close
        prev_close = prev_bar.close if prev_bar is not None else bar.close
        
        # Compute variance proxy
        v_t = self.compute_variance_proxy(bar, prev_close)
        
        # EWMA update: h_t = lambda * h_{t-1} + (1 - lambda) * v_t
        self.h = self.lambda_val * self.h + (1 - self.lambda_val) * v_t
        
        self.last_update_ts = bar.timestamp
    
    def decay(self, num_missing_bars: int):
        """
        Apply variance decay for missing bars:
        h <- lambda^k * h
        
        Args:
            num_missing_bars: Number of missing bars
        """
        if num_missing_bars > 0:
            decay_factor = self.lambda_val ** num_missing_bars
            self.h = decay_factor * self.h
    
    def get_volatility(self) -> float:
        """Get current volatility (sqrt of variance)"""
        return math.sqrt(max(self.h, 1e-10))
    
    def get_variance(self) -> float:
        """Get current variance"""
        variance = max(self.h, 1e-10)
        logger.debug(
            f"VolatilityState({self.asset}, {self.resolution}m).get_variance(): "
            f"h={self.h:.8f}, volatility={math.sqrt(variance):.6f}"
        )
        return variance
    
    def reset(self):
        """Reset state to initial value"""
        self.h = 0.00001  # Match initial value
        self.last_update_ts = None
    
    def to_dict(self) -> dict:
        """Serialize state to dict"""
        return {
            'h': float(self.h),
            'last_update_ts': self.last_update_ts,
            'lambda_val': float(self.lambda_val),
        }
    
    def from_dict(self, data: dict):
        """Load state from dict"""
        old_h = self.h
        self.h = float(data.get('h', 0.00001))  # Use smaller default
        self.last_update_ts = data.get('last_update_ts')
        self.lambda_val = float(data.get('lambda_val', self.lambda_val))
        
        if self.last_update_ts is not None:
            logger.info(
                f"VolatilityState({self.asset}, {self.resolution}m) loaded: "
                f"h={self.h:.8f} (was {old_h:.8f}), "
                f"volatility={math.sqrt(self.h):.6f}, "
                f"last_update={self.last_update_ts}"
            )
