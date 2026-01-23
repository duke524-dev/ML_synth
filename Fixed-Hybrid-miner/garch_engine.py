"""
GARCH volatility engine for Fixed Hybrid miner
Fits GARCH(1,1) on 5-minute returns and provides forward volatility curve
"""
import logging
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timezone

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch library not available. GARCH functionality will be limited.")

from config import GARCH_ORDER, GARCH_DIST, GARCH_DOWNSCALE_FACTOR

logger = logging.getLogger(__name__)


class GARCHEngine:
    """GARCH volatility engine using 5-minute returns"""
    
    def __init__(self, asset: str):
        self.asset = asset
        self.model = None
        self.last_fit_time: Optional[datetime] = None
        self.returns_history: List[float] = []
        self.min_samples = 100  # Minimum samples for GARCH fit
    
    def add_returns(self, returns: List[float]):
        """Add returns to history"""
        self.returns_history.extend(returns)
        # Keep only recent history (e.g., last 10,000 points)
        if len(self.returns_history) > 10000:
            self.returns_history = self.returns_history[-10000:]
    
    def fit(self, returns: Optional[List[float]] = None) -> bool:
        """
        Fit GARCH(1,1) model on returns
        
        Args:
            returns: List of log returns. If None, uses internal history.
        
        Returns:
            True if fit successful, False otherwise
        """
        if not ARCH_AVAILABLE:
            logger.warning("arch library not available. Cannot fit GARCH model.")
            return False
        
        if returns is None:
            returns = self.returns_history
        
        if len(returns) < self.min_samples:
            logger.warning(
                f"Insufficient samples for GARCH fit: {len(returns)} < {self.min_samples}"
            )
            return False
        
        try:
            # Convert to numpy array
            returns_array = np.array(returns, dtype=np.float64)
            
            # Fit GARCH(1,1) with Student-t distribution
            self.model = arch_model(
                returns_array * 100,  # Scale to percentage (arch expects this)
                vol='Garch',
                p=GARCH_ORDER[0],
                q=GARCH_ORDER[1],
                dist=GARCH_DIST,
            )
            
            # Fit model
            fit_result = self.model.fit(disp='off', show_warning=False)
            
            self.last_fit_time = datetime.now(timezone.utc)
            
            logger.info(
                f"GARCH fit successful for {self.asset}: "
                f"omega={fit_result.params['omega']:.6f}, "
                f"alpha={fit_result.params['alpha[1]']:.6f}, "
                f"beta={fit_result.params['beta[1]']:.6f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model for {self.asset}: {e}", exc_info=True)
            self.model = None
            return False
    
    def forecast_volatility_curve(
        self,
        horizon_steps: int,
        time_increment: int,
        last_return: Optional[float] = None,
        last_variance: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Forecast volatility curve over horizon
        
        Args:
            horizon_steps: Number of steps in horizon
            time_increment: Time increment in seconds (60 for HF, 300 for LF)
            last_return: Last observed return (for conditional variance)
            last_variance: Last observed variance (if known)
        
        Returns:
            Array of volatility forecasts (one per step) or None if unavailable
        """
        if self.model is None:
            return None
        
        try:
            # Get model parameters
            params = self.model.params
            
            # Extract GARCH parameters (scaled back from percentage)
            omega = params['omega'] / 10000.0  # Convert from percentage^2
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            
            # Initialize conditional variance
            if last_variance is not None:
                h_t = last_variance
            elif last_return is not None:
                # Use last return to estimate initial variance
                h_t = (last_return ** 2) * alpha + omega / (1 - beta)
            else:
                # Use unconditional variance
                h_t = omega / (1 - alpha - beta)
            
            # Forecast forward
            volatility_curve = np.zeros(horizon_steps)
            
            for t in range(horizon_steps):
                # GARCH(1,1) forecast: h_{t+1} = omega + alpha * epsilon_t^2 + beta * h_t
                # For multi-step ahead, use unconditional expectation
                # For step 1: h_1 = omega + alpha * epsilon_0^2 + beta * h_0
                # For step t>1: h_t = omega + (alpha + beta) * h_{t-1}
                if t == 0:
                    # First step: use last return if available
                    if last_return is not None:
                        h_t = omega + alpha * (last_return ** 2) + beta * h_t
                    else:
                        h_t = omega + (alpha + beta) * h_t
                else:
                    # Subsequent steps: use unconditional expectation
                    h_t = omega + (alpha + beta) * h_t
                
                # Convert to volatility (sqrt of variance)
                vol = np.sqrt(h_t)
                
                # For HF (1-minute), downscale from 5-minute
                if time_increment == 60:
                    vol = vol / np.sqrt(GARCH_DOWNSCALE_FACTOR)
                
                volatility_curve[t] = vol
            
            return volatility_curve
            
        except Exception as e:
            logger.error(
                f"Error forecasting GARCH volatility for {self.asset}: {e}",
                exc_info=True
            )
            return None
    
    def get_unconditional_variance(self) -> Optional[float]:
        """Get unconditional variance from fitted model"""
        if self.model is None:
            return None
        
        try:
            params = self.model.params
            omega = params['omega'] / 10000.0
            alpha = params['alpha[1]']
            beta = params['beta[1]']
            
            # Unconditional variance: omega / (1 - alpha - beta)
            if alpha + beta < 1.0:
                return omega / (1 - alpha - beta)
            else:
                logger.warning("GARCH model not stationary (alpha + beta >= 1)")
                return None
                
        except Exception as e:
            logger.error(f"Error computing unconditional variance: {e}")
            return None
