"""
Vectorized simulation engine using EWMA volatility and Student-t innovations
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Tuple
from scipy.stats import t as student_t

from config import NU, EQUITY_OFF_HOURS_MULT, LF_EQUITY_ASSETS
from volatility_state import VolatilityState

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Vectorized simulation engine"""
    
    def __init__(self, vol_state_1m: VolatilityState, vol_state_5m: VolatilityState):
        self.vol_state_1m = vol_state_1m
        self.vol_state_5m = vol_state_5m
        self.asset = vol_state_1m.asset
        self.nu = NU.get(self.asset, 6.0)
        
        # Standardization factor for Student-t
        if self.nu > 2:
            self.std_factor = np.sqrt(self.nu / (self.nu - 2))
        else:
            self.std_factor = 1.0  # Fallback
    
    def generate_paths(
        self,
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
        anchor_price: float,
        include_start: bool = True,
    ) -> Tuple:
        """
        Generate simulated price paths
        
        Args:
            start_time: ISO format start time
            time_increment: dt in seconds (60 or 300)
            time_length: Total length in seconds
            num_simulations: Number of paths (typically 1000)
            anchor_price: Starting price
            include_start: Whether to include start point (steps+1 vs steps)
        
        Returns:
            Tuple: (start_timestamp, time_increment, [path1, path2, ...])
        """
        # Parse start time
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        
        start_ts = int(start_dt.timestamp())
        
        # Calculate steps
        H = time_length // time_increment
        if H * time_increment != time_length:
            raise ValueError(f"time_length must be divisible by time_increment")
        
        # Select volatility state
        if time_increment == 60:
            vol_state = self.vol_state_1m
        elif time_increment == 300:
            vol_state = self.vol_state_5m
        else:
            raise ValueError(f"Unsupported time_increment: {time_increment}")
        
        # Get initial variance
        h_0 = vol_state.get_variance()
        lambda_val = vol_state.lambda_val
        
        # Apply off-hours multiplier for equities (24-hour paths)
        # Multiplier applies to volatility, so square it for variance
        if self.asset in LF_EQUITY_ASSETS and time_length >= 86400:
            h_0 = h_0 * (EQUITY_OFF_HOURS_MULT ** 2)
            logger.debug(
                f"Applied off-hours multiplier for {self.asset}: "
                f"h_0={h_0:.8f}, volatility={np.sqrt(h_0):.6f}"
            )
        
        logger.info(
            f"SimulationEngine.generate_paths({self.asset}): "
            f"anchor_price={anchor_price:.2f}, "
            f"h_0={h_0:.8f}, volatility={np.sqrt(h_0):.6f}, "
            f"time_increment={time_increment}s, time_length={time_length}s, "
            f"num_simulations={num_simulations}"
        )
        
        # Pre-allocate arrays (float32 for memory efficiency)
        paths = np.zeros((num_simulations, H + 1), dtype=np.float32)
        paths[:, 0] = anchor_price
        
        # Initialize variance state per path
        h_paths = np.full(num_simulations, h_0, dtype=np.float32)
        
        # Vectorized simulation loop over time steps
        for t in range(H):
            # Get current volatility (sqrt of variance)
            sigma_t = np.sqrt(h_paths)
            
            # Sample Student-t shocks (vectorized over paths)
            eps_raw = student_t.rvs(df=self.nu, size=num_simulations)
            # Standardize to unit variance
            eps = eps_raw / self.std_factor
            
            # Generate log returns: r_t = sigma_t * eps (no drift for short horizons)
            log_returns = sigma_t * eps
            
            # Update prices: P_t = P_{t-1} * exp(r_t)
            paths[:, t + 1] = paths[:, t] * np.exp(log_returns)
            
            # Update variance with volatility clustering:
            # h <- lambda * h + (1 - lambda) * r^2
            r_squared = log_returns ** 2
            h_paths = lambda_val * h_paths + (1 - lambda_val) * r_squared
        
        # Ensure positive and finite
        paths = np.maximum(paths, 1e-8)
        paths = np.where(np.isfinite(paths), paths, anchor_price)
        
        # Format output
        if not include_start:
            paths = paths[:, 1:]  # Remove start point
        
        # Round to 8 significant digits
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from synth.utils.helpers import round_to_8_significant_digits
        
        formatted_paths = []
        for path in paths:
            formatted_path = [
                round_to_8_significant_digits(float(p)) for p in path
            ]
            formatted_paths.append(formatted_path)
        
        return (start_ts, time_increment, *formatted_paths)
