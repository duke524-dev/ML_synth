"""
Volatility state management for Fixed Hybrid miner
EWMA-based with support for GARCH blending
"""
import logging
import math
from typing import Optional
from dataclasses import dataclass

from config import (
    EWMA_HALF_LIFE_HF, EWMA_HALF_LIFE_LF, VARIANCE_MIXING_ALPHA, SIGMA_MAP
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
    
    def __init__(self, asset: str, resolution: int, is_hf: bool = False):
        """
        Args:
            asset: Asset symbol
            resolution: 1 or 5 (minutes)
            is_hf: True for high-frequency (HF), False for low-frequency (LF)
        """
        self.asset = asset
        self.resolution = resolution
        self.is_hf = is_hf
        self.dt = resolution * 60  # seconds
        
        # Get half-life based on HF/LF
        if is_hf:
            half_life_seconds = EWMA_HALF_LIFE_HF
        else:
            half_life_seconds = EWMA_HALF_LIFE_LF
        
        # Compute lambda: lambda = exp(-ln(2) * dt / H)
        self.lambda_val = math.exp(-math.log(2) * self.dt / half_life_seconds)
        
        # State: conditional variance h_t
        # Use SIGMA_MAP as initial variance estimate if available
        # Convert sigma (per-minute volatility) to variance: h = sigma^2
        if asset in SIGMA_MAP:
            sigma = SIGMA_MAP[asset]
            # For 1-minute: sigma is already per-minute
            # For 5-minute: scale by sqrt(5) to get per-5min volatility
            if resolution == 5:
                sigma = sigma * math.sqrt(5)
            self.h = sigma ** 2
        else:
            self.h = 0.00001  # Default initial variance
        
        # Last update timestamp
        self.last_update_ts: Optional[int] = None
        
        logger.info(
            f"VolatilityState({asset}, {resolution}m, {'HF' if is_hf else 'LF'}): "
            f"half_life={half_life_seconds/60:.1f}m, lambda={self.lambda_val:.6f}, "
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
        
        # Ensure non-negative
        self.h = max(self.h, 1e-10)
        
        self.last_update_ts = bar.timestamp
        
        logger.debug(
            f"VolatilityState.update_with_bar({self.asset}): "
            f"v_t={v_t:.8f}, h={self.h:.8f}, volatility={math.sqrt(self.h):.6f}"
        )
    
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
    
    def update(self, bar: OHLCBar, prev_close: Optional[float] = None):
        """Legacy method - use update_with_bar instead"""
        prev_bar = None
        if prev_close is not None:
            # Create a dummy previous bar
            prev_bar = OHLCBar(
                timestamp=bar.timestamp - self.dt,
                open=prev_close,
                high=prev_close,
                low=prev_close,
                close=prev_close
            )
        self.update_with_bar(bar, prev_bar)
    
    def get_variance(self) -> float:
        """Get current variance"""
        variance = max(self.h, 1e-10)
        logger.debug(
            f"VolatilityState({self.asset}, {self.resolution}m).get_variance(): "
            f"h={self.h:.8f}, volatility={math.sqrt(variance):.6f}"
        )
        return variance
    
    def get_volatility(self) -> float:
        """Get current volatility (sqrt of variance)"""
        return math.sqrt(max(self.h, 1e-10))
    
    def reset(self):
        """Reset state to initial value"""
        if self.asset in SIGMA_MAP:
            sigma = SIGMA_MAP[self.asset]
            if self.resolution == 5:
                sigma = sigma * math.sqrt(5)
            self.h = sigma ** 2
        else:
            self.h = 0.00001
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
        self.h = float(data.get('h', 0.00001))
        self.last_update_ts = data.get('last_update_ts')
        self.lambda_val = float(data.get('lambda_val', self.lambda_val))
        
        if self.last_update_ts is not None:
            logger.info(
                f"VolatilityState({self.asset}, {self.resolution}m) loaded: "
                f"h={self.h:.8f} (was {old_h:.8f}), "
                f"volatility={math.sqrt(self.h):.6f}, "
                f"last_update={self.last_update_ts}"
            )
