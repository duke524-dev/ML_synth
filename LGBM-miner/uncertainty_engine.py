"""
Uncertainty engine: EWMA + Student-t + coherence for generating paths
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple
from scipy.stats import t as student_t

from config import NU_1M, NU_5M, NU, EQUITY_OFF_HOURS_MULT, LF_EQUITY_ASSETS, SIGMA_MAP
from volatility_state import VolatilityState

logger = logging.getLogger(__name__)


class UncertaintyEngine:
    """Generates uncertainty around center path using EWMA + Student-t + coherence"""
    
    def __init__(
        self,
        vol_state_1m: VolatilityState,
        vol_state_5m: VolatilityState,
        asset: str
    ):
        self.vol_state_1m = vol_state_1m
        self.vol_state_5m = vol_state_5m
        self.asset = asset
    
    def generate_paths(
        self,
        center_path: List[float],  # Log prices
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
        anchor_price: float,
    ) -> Tuple:
        """
        Generate price paths around center path using EWMA + Student-t
        
        Args:
            center_path: List of predicted log prices (center path)
            start_time: ISO format start time
            time_increment: dt in seconds (60 or 300)
            time_length: Total length in seconds
            num_simulations: Number of paths (typically 1000)
            anchor_price: Starting price
        
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
        
        # Select volatility state and nu based on time_increment
        if time_increment == 60:
            vol_state = self.vol_state_1m
            nu = NU_1M.get(self.asset, NU.get(self.asset, 6.0))
        elif time_increment == 300:
            vol_state = self.vol_state_5m
            nu = NU_5M.get(self.asset, NU.get(self.asset, 6.0))
        else:
            raise ValueError(f"Unsupported time_increment: {time_increment}")
        
        # Compute standardization factor for this nu
        if nu > 2:
            std_factor = np.sqrt(nu / (nu - 2))
        else:
            std_factor = 1.0  # Fallback
        
        # Get initial variance
        h_0 = vol_state.get_variance()
        lambda_val = vol_state.lambda_val
        
        # Fallback: if variance is too small (uninitialized), use SIGMA_MAP
        if h_0 < 1e-8:
            logger.warning(
                f"Volatility state for {self.asset} appears uninitialized "
                f"(h_0={h_0:.8f}), using SIGMA_MAP fallback"
            )
            if self.asset in SIGMA_MAP:
                sigma = SIGMA_MAP[self.asset]
                # Scale sigma based on time_increment
                # For 1-minute: sigma is per-minute
                # For 5-minute: scale by sqrt(5)
                if time_increment == 300:
                    sigma = sigma * (300 / 60) ** 0.5  # sqrt(5) ≈ 2.236
                h_0 = sigma ** 2
                logger.info(
                    f"Using fallback variance from SIGMA_MAP: "
                    f"sigma={sigma:.6f}, h_0={h_0:.8f}"
                )
            else:
                logger.warning(
                    f"No SIGMA_MAP entry for {self.asset}, using default h_0=0.00001"
                )
                h_0 = 0.00001
        
        # Apply off-hours multiplier for equities (24-hour paths)
        if self.asset in LF_EQUITY_ASSETS and time_length >= 86400:
            h_0 = h_0 * (EQUITY_OFF_HOURS_MULT ** 2)
            logger.debug(
                f"Applied off-hours multiplier for {self.asset}: "
                f"h_0={h_0:.8f}, volatility={np.sqrt(h_0):.6f}"
            )
        
        logger.info(
            f"UncertaintyEngine.generate_paths({self.asset}): "
            f"anchor_price={anchor_price:.2f}, "
            f"h_0={h_0:.8f}, volatility={np.sqrt(h_0):.6f}, "
            f"time_increment={time_increment}s, time_length={time_length}s, "
            f"num_simulations={num_simulations}, nu={nu:.1f}"
        )
        
        # Convert center path from log prices to prices
        if len(center_path) != H:
            logger.warning(
                f"Center path length ({len(center_path)}) != H ({H}), "
                f"padding or truncating"
            )
            if len(center_path) < H:
                # Pad with last value
                center_path = center_path + [center_path[-1]] * (H - len(center_path))
            else:
                # Truncate
                center_path = center_path[:H]
        
        center_prices = np.exp(center_path)
        
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
            eps_raw = student_t.rvs(df=nu, size=num_simulations)
            # Standardize to unit variance
            eps = eps_raw / std_factor
            
            # Center path deviation
            if t < len(center_prices):
                center_price = center_prices[t]
            else:
                center_price = center_prices[-1]  # Use last center price
            
            # Generate log returns: r_t = (log(center_price / prev_price)) + sigma_t * eps
            # This adds uncertainty around the center path
            prev_price = paths[:, t]
            center_log_return = np.log(center_price / prev_price)
            
            # Add uncertainty: r_t = center_log_return + sigma_t * eps
            log_returns = center_log_return + sigma_t * eps
            
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