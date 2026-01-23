"""
Simulation engine for Fixed Hybrid miner
Combines: GARCH volatility curve + EWMA regime + Student-t innovations + AR(1) coherence
"""
import logging
import math
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, List
from scipy.stats import t as student_t

from config import (
    NU_HF, NU_LF_CRYPTO, NU_LF_EQUITY,
    VOL_BLEND_WEIGHT_HF, VOL_BLEND_WEIGHT_LF,
    COHERENCE_RHO_HF, COHERENCE_RHO_LF_CRYPTO, COHERENCE_RHO_LF_EQUITY,
    COHERENCE_S_VOL_HF, COHERENCE_S_VOL_LF_CRYPTO, COHERENCE_S_VOL_LF_EQUITY,
    SIGMA_SCALE_HF, SIGMA_SCALE_LF,
    EQUITY_OFF_HOURS_MULT, LF_EQUITY_ASSETS, HF_ASSETS, SIGMA_MAP
)
from volatility_state import VolatilityState
from garch_engine import GARCHEngine

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Vectorized simulation engine with GARCH + EWMA + Student-t + coherence"""
    
    def __init__(
        self,
        vol_state_1m: VolatilityState,
        vol_state_5m: VolatilityState,
        garch_engine: GARCHEngine,
        asset: str
    ):
        self.vol_state_1m = vol_state_1m
        self.vol_state_5m = vol_state_5m
        self.garch_engine = garch_engine
        self.asset = asset
    
    def generate_paths(
        self,
        center_path: List[float],  # Center path as log prices
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
        anchor_price: float,
        garch_vol_curve: Optional[np.ndarray] = None,
    ) -> Tuple:
        """
        Generate simulated price paths using hybrid approach
        
        Args:
            center_path: Center path as log prices (from LightGBM)
            start_time: ISO format start time
            time_increment: dt in seconds (60 or 300)
            time_length: Total length in seconds
            num_simulations: Number of paths (typically 1000)
            anchor_price: Starting price (oracle anchor)
            garch_vol_curve: Optional GARCH volatility curve (if available)
        
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
        
        # Determine if HF or LF
        is_hf = time_increment == 60
        
        # Select volatility state and parameters
        if is_hf:
            vol_state = self.vol_state_1m
            nu = NU_HF
            vol_blend_weight = VOL_BLEND_WEIGHT_HF
            coherence_rho = COHERENCE_RHO_HF
            coherence_s_vol = COHERENCE_S_VOL_HF
            sigma_scale = SIGMA_SCALE_HF.get(self.asset, 1.0)
        else:
            vol_state = self.vol_state_5m
            if self.asset in LF_EQUITY_ASSETS:
                nu = NU_LF_EQUITY
                coherence_rho = COHERENCE_RHO_LF_EQUITY
                coherence_s_vol = COHERENCE_S_VOL_LF_EQUITY
            else:
                nu = NU_LF_CRYPTO
                coherence_rho = COHERENCE_RHO_LF_CRYPTO
                coherence_s_vol = COHERENCE_S_VOL_LF_CRYPTO
            vol_blend_weight = VOL_BLEND_WEIGHT_LF
            sigma_scale = SIGMA_SCALE_LF.get(self.asset, 1.0)
        
        # Compute standardization factor for Student-t
        if nu > 2:
            std_factor = np.sqrt(nu / (nu - 2))
        else:
            std_factor = 1.0
        
        # Get EWMA volatility
        ewma_vol = vol_state.get_volatility()
        ewma_var = vol_state.get_variance()
        
        # Fallback: if variance is too small (uninitialized), use SIGMA_MAP
        if ewma_var < 1e-8:
            logger.warning(
                f"Volatility state for {self.asset} appears uninitialized "
                f"(h_0={ewma_var:.8f}), using SIGMA_MAP fallback"
            )
            if self.asset in SIGMA_MAP:
                sigma = SIGMA_MAP[self.asset]
                # Scale sigma based on time_increment
                # For 1-minute: sigma is per-minute
                # For 5-minute: scale by sqrt(5)
                if time_increment == 300:
                    sigma = sigma * (300 / 60) ** 0.5  # sqrt(5) ≈ 2.236
                ewma_var = sigma ** 2
                ewma_vol = sigma
                logger.info(
                    f"Using fallback variance from SIGMA_MAP: "
                    f"sigma={sigma:.6f}, h_0={ewma_var:.8f}"
                )
            else:
                logger.warning(
                    f"No SIGMA_MAP entry for {self.asset}, using default h_0=0.00001"
                )
                ewma_var = 0.00001
                ewma_vol = math.sqrt(ewma_var)
        
        # Get GARCH volatility curve (if available)
        if garch_vol_curve is None:
            # Fallback: use constant EWMA volatility
            garch_vol_curve = np.full(H, ewma_vol)
            logger.warning(f"No GARCH curve for {self.asset}, using EWMA")
        
        # Blend GARCH and EWMA volatilities
        # sigma_blend^2(t) = w * sigma_GARCH^2(t) + (1-w) * sigma_EWMA^2
        sigma_blend_sq = (
            vol_blend_weight * (garch_vol_curve ** 2) +
            (1 - vol_blend_weight) * ewma_var
        )
        sigma_blend = np.sqrt(sigma_blend_sq)
        
        # Apply calibration scalar
        sigma = sigma_blend * sigma_scale
        
        # Apply off-hours multiplier for equities
        if self.asset in LF_EQUITY_ASSETS and time_length >= 86400:
            # Reduce volatility during off-hours
            sigma = sigma * EQUITY_OFF_HOURS_MULT
        
        logger.info(
            f"SimulationEngine.generate_paths({self.asset}): "
            f"anchor_price={anchor_price:.2f}, "
            f"ewma_vol={ewma_vol:.6f}, "
            f"sigma_scale={sigma_scale:.3f}, "
            f"time_increment={time_increment}s, H={H}, "
            f"num_simulations={num_simulations}, nu={nu:.1f}"
        )
        
        # Convert center path from log prices to prices
        if len(center_path) != H + 1:
            logger.warning(
                f"Center path length mismatch: {len(center_path)} != {H + 1}. "
                f"Interpolating or truncating."
            )
            if len(center_path) < H + 1:
                # Extend with last value
                center_path = center_path + [center_path[-1]] * (H + 1 - len(center_path))
            else:
                # Truncate
                center_path = center_path[:H + 1]
        
        center_prices = np.exp(center_path)
        
        # Rescale center path to match anchor
        if center_prices[0] > 0:
            scale_factor = anchor_price / center_prices[0]
            center_prices = center_prices * scale_factor
        
        # Compute drift (log returns from center path)
        log_center = np.log(center_prices)
        mu = np.diff(log_center)  # Drift per step
        
        # Pre-allocate arrays
        paths = np.zeros((num_simulations, H + 1), dtype=np.float32)
        paths[:, 0] = anchor_price
        
        # Generate AR(1) coherence factors for each path
        # z_t = rho * z_{t-1} + epsilon_z, where epsilon_z ~ N(0, s_vol^2)
        z = np.zeros((num_simulations, H), dtype=np.float32)
        for t in range(H):
            if t == 0:
                z[:, t] = np.random.normal(0, coherence_s_vol, num_simulations)
            else:
                z[:, t] = coherence_rho * z[:, t-1] + np.random.normal(
                    0, coherence_s_vol * np.sqrt(1 - coherence_rho**2), num_simulations
                )
        
        # Generate Student-t innovations
        # Sample epsilon_t ~ StudentT(nu) with unit variance
        epsilon = student_t.rvs(nu, size=(num_simulations, H)) / std_factor
        
        # Apply coherence to volatility: log sigma_sim(t) = log sigma(t) + s_vol * z_t
        log_sigma = np.log(sigma + 1e-10)  # Add small epsilon to avoid log(0)
        log_sigma_sim = log_sigma[None, :] + coherence_s_vol * z
        sigma_sim = np.exp(log_sigma_sim)
        
        # Simulate returns: r_t = mu_t + sigma_sim(t) * epsilon_t
        returns = mu[None, :] + sigma_sim * epsilon
        
        # Integrate to prices: P(t) = P0 * exp(sum_{i=1..t} r_i)
        cumulative_returns = np.cumsum(returns, axis=1)
        paths[:, 1:] = anchor_price * np.exp(cumulative_returns)
        
        # Ensure positive and finite
        paths = np.maximum(paths, 1e-8)
        paths = np.where(np.isfinite(paths), paths, anchor_price)
        
        # Format output with proper rounding
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from synth.utils.helpers import round_to_8_significant_digits
        
        output_paths = []
        for i in range(num_simulations):
            path = []
            for t in range(H + 1):
                timestamp = start_ts + t * time_increment
                price = round_to_8_significant_digits(float(paths[i, t]))
                path.append({"time": timestamp, "price": price})
            output_paths.append(path)
        
        return (start_ts, time_increment, output_paths)
