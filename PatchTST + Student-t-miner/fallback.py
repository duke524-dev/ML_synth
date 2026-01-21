"""
Fallback generator for when model/data is unavailable
"""
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Tuple
from scipy.stats import t as student_t

from config import FALLBACK_PRICES
from path_sampling import sample_student_t

logger = logging.getLogger(__name__)


class FallbackGenerator:
    """Generates valid paths using simple baseline when model fails"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
    
    def _compute_realized_volatility(self, asset: str, resolution: int) -> float:
        """Compute realized volatility from recent cache data"""
        try:
            cache = self.cache_manager.get_cache(asset, resolution, resolution == 1)
            
            with cache.lock:
                if len(cache.data) < 100:
                    return 0.02  # Default
                
                # Get recent prices
                recent = cache.data[-100:]
                prices = [p for _, p in recent]
                
                if len(prices) < 2:
                    return 0.02
                
                # Compute log returns
                returns = np.diff(np.log(prices))
                
                # Annualized volatility
                periods_per_year = (365.25 * 24 * 60) / resolution
                vol = np.std(returns) * np.sqrt(periods_per_year)
                
                return max(0.01, min(0.5, vol))  # Clamp to reasonable range
                
        except Exception as e:
            logger.warning(f"Error computing realized vol: {e}")
            return 0.02  # Default
    
    def generate_paths(
        self,
        asset: str,
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
        anchor_price: float,
    ) -> Tuple:
        """
        Generate fallback paths using simple GBM with Student-t noise
        
        Returns:
            Tuple in format: (start_timestamp, time_increment, [path1, path2, ...])
        """
        try:
            # Parse start time
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            
            # Calculate horizon
            H = time_length // time_increment
            if H * time_increment != time_length:
                raise ValueError(f"time_length must be divisible by time_increment")
            
            # Get realized volatility
            resolution = 1 if time_increment == 60 else 5
            sigma = self._compute_realized_volatility(asset, resolution)
            
            # Convert to per-step volatility
            dt = time_increment / (365.25 * 24 * 3600)  # Years
            sigma_step = sigma * np.sqrt(dt)
            
            # Use Student-t with nu=5 (moderate fat tails)
            nu = 5.0
            
            # Generate paths: drift=0, volatility from realized vol
            mu = np.zeros(H)  # No drift
            log_sigma = np.full(H, np.log(sigma_step))
            
            # Sample paths
            paths = sample_student_t(nu, size=(num_simulations, H))
            
            # Convert to log returns
            log_returns = mu + np.exp(log_sigma) * paths  # [N, H]
            
            # Convert to prices
            price_paths = np.zeros((num_simulations, H + 1))
            price_paths[:, 0] = anchor_price
            
            for t in range(1, H + 1):
                price_paths[:, t] = price_paths[:, t - 1] * np.exp(log_returns[:, t - 1])
            
            # Ensure positive and finite
            price_paths = np.maximum(price_paths, 1e-8)
            price_paths = np.where(np.isfinite(price_paths), price_paths, anchor_price)
            
            # Format output
            start_timestamp = int(start_dt.timestamp())
            
            # Import here to avoid circular dependency
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from synth.utils.helpers import round_to_8_significant_digits
            
            formatted_paths = []
            for path in price_paths:
                formatted_path = [
                    round_to_8_significant_digits(float(p)) for p in path
                ]
                formatted_paths.append(formatted_path)
            
            logger.info(f"Generated {num_simulations} fallback paths for {asset}")
            return (start_timestamp, time_increment, *formatted_paths)
            
        except Exception as e:
            logger.error(f"Error in fallback generator: {e}", exc_info=True)
            # Last resort: constant price
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            
            H = time_length // time_increment
            start_timestamp = int(start_dt.timestamp())
            
            # Return constant price paths
            formatted_paths = []
            for _ in range(num_simulations):
                path = [anchor_price] * (H + 1)
                formatted_paths.append(path)
            
            return (start_timestamp, time_increment, *formatted_paths)
