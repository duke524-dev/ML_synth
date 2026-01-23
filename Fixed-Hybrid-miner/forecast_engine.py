"""
Forecast engine: Main orchestrator for Fixed Hybrid miner
Combines LightGBM (center path) + GARCH + EWMA + Student-t + coherence
"""
import logging
import os
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, List

from data_fetcher import HermesFetcher
from data_manager import DataManager
from volatility_state import VolatilityState, OHLCBar
from garch_engine import GARCHEngine
from simulation_engine import SimulationEngine
from lgbm_predictor import LGBMPredictor
from persistence import StatePersistence
from config import (
    FALLBACK_PRICES, TOKEN_MAP, HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS,
    HF_HORIZON_STEPS, LF_HORIZON_STEPS
)

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Main forecast engine orchestrating all components"""
    
    def __init__(self, state_dir: str = "state", model_dir: str = "models"):
        self.state_dir = state_dir
        self.model_dir = model_dir
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        self.hermes_fetcher = HermesFetcher()
        self.persistence = StatePersistence(state_dir)
        
        # Per-asset components
        self.data_managers: dict = {}
        self.vol_states_1m: dict = {}
        self.vol_states_5m: dict = {}
        self.garch_engines: dict = {}
        self.lgbm_predictors_hf: dict = {}
        self.lgbm_predictors_lf: dict = {}
        
        # Initialize for all assets
        for asset in TOKEN_MAP.keys():
            self.data_managers[asset] = DataManager(asset)
            self.vol_states_1m[asset] = VolatilityState(asset, resolution=1, is_hf=True)
            self.vol_states_5m[asset] = VolatilityState(asset, resolution=5, is_hf=False)
            self.garch_engines[asset] = GARCHEngine(asset)
            
            # Initialize LGBM predictors
            if asset in HF_ASSETS:
                self.lgbm_predictors_hf[asset] = LGBMPredictor(asset, is_hf=True, model_dir=model_dir)
                self.lgbm_predictors_hf[asset].load_models()
            
            if asset in LF_CRYPTO_ASSETS or asset in LF_EQUITY_ASSETS:
                self.lgbm_predictors_lf[asset] = LGBMPredictor(asset, is_hf=False, model_dir=model_dir)
                self.lgbm_predictors_lf[asset].load_models()
        
        # Load persisted states
        self._load_states()
    
    def _load_states(self):
        """Load persisted states for all assets"""
        for asset in TOKEN_MAP.keys():
            self.persistence.load_asset_state(
                asset=asset,
                vol_state_1m=self.vol_states_1m[asset],
                vol_state_5m=self.vol_states_5m[asset],
                data_manager=self.data_managers[asset]
            )
    
    def generate_paths(
        self,
        asset: str,
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
    ) -> Optional[Tuple]:
        """
        Generate simulation paths using hybrid approach
        
        Args:
            asset: Asset symbol
            start_time: ISO format start time
            time_increment: Time increment in seconds (60 or 300)
            time_length: Total time length in seconds
            num_simulations: Number of paths (typically 1000)
        
        Returns:
            Tuple: (start_timestamp, time_increment, [path1, path2, ...])
            or None if error
        """
        try:
            # Determine if HF or LF
            is_hf = time_increment == 60
            
            # Fetch anchor price
            anchor_price = self.hermes_fetcher.fetch_anchor_price(asset)
            if anchor_price is None:
                anchor_price = FALLBACK_PRICES.get(asset, 100.0)
                logger.warning(f"Using fallback price for {asset}: {anchor_price}")
            
            # Get components
            vol_state_1m = self.vol_states_1m[asset]
            vol_state_5m = self.vol_states_5m[asset]
            garch_engine = self.garch_engines[asset]
            
            # Get LGBM predictor
            if is_hf:
                lgbm_predictor = self.lgbm_predictors_hf.get(asset)
            else:
                lgbm_predictor = self.lgbm_predictors_lf.get(asset)
            
            # Prepare features for LGBM (simplified - would need actual feature engineering)
            # For now, use dummy features if predictor not available
            if lgbm_predictor and lgbm_predictor.models:
                # Get historical data for features
                data_manager = self.data_managers[asset]
                bars = data_manager.get_5m_bars() if not is_hf else data_manager.get_1m_bars()
                
                if bars:
                    # Create simple features (would need proper feature engineering)
                    closes = [b.close for b in bars[-20:]]
                    features = np.array(closes[-10:] + [0] * (10 - len(closes[-10:]))).astype(np.float32)
                    
                    # Predict center path
                    center_path = lgbm_predictor.predict_center_path(
                        features, time_increment, time_length
                    )
                else:
                    center_path = None
            else:
                center_path = None
            
            # Fallback: use constant drift if no center path
            if center_path is None:
                H = time_length // time_increment
                # Use anchor price as center (zero drift)
                log_anchor = np.log(anchor_price)
                center_path = [log_anchor] * (H + 1)
                logger.warning(f"No LGBM center path for {asset}, using constant")
            
            # Get GARCH volatility curve
            garch_vol_curve = None
            if garch_engine.model is not None:
                # Get last return for conditional variance
                data_manager = self.data_managers[asset]
                bars_5m = data_manager.get_5m_bars()
                if len(bars_5m) >= 2:
                    last_return = np.log(bars_5m[-1].close / bars_5m[-2].close)
                    H = time_length // time_increment
                    garch_vol_curve = garch_engine.forecast_volatility_curve(
                        horizon_steps=H,
                        time_increment=time_increment,
                        last_return=last_return,
                    )
            
            # Create simulation engine
            sim_engine = SimulationEngine(
                vol_state_1m=vol_state_1m,
                vol_state_5m=vol_state_5m,
                garch_engine=garch_engine,
                asset=asset
            )
            
            # Generate paths
            result = sim_engine.generate_paths(
                center_path=center_path,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                anchor_price=anchor_price,
                garch_vol_curve=garch_vol_curve,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating paths for {asset}: {e}", exc_info=True)
            return None
