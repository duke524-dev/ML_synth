"""
Forecast engine: Orchestrates LGBM prediction + uncertainty engine
"""
import logging
import os
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional, List
from collections import deque

from data_fetcher import HermesFetcher, BenchmarksFetcher, parse_benchmarks_ohlc
from data_manager import DataManager
from volatility_state import VolatilityState, OHLCBar
from uncertainty_engine import UncertaintyEngine
from lgbm_trainer import LGBMTrainer
from lgbm_predictor import LGBMPredictor
from persistence import StatePersistence
from config import (
    FALLBACK_PRICES, TOKEN_MAP, HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS,
    HF_RETRAIN_INTERVAL_DAYS, LF_RETRAIN_INTERVAL_DAYS, TRAINING_YEARS
)

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Main forecast engine orchestrating LGBM, volatility, and simulation"""
    
    def __init__(self, state_dir: str = "state", model_dir: str = "models"):
        self.state_dir = state_dir
        self.model_dir = model_dir
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        self.hermes_fetcher = HermesFetcher()
        self.benchmarks_fetcher = BenchmarksFetcher()
        self.persistence = StatePersistence(state_dir)
        
        # Per-asset data managers and volatility states
        self.data_managers: dict = {}
        self.vol_states_1m: dict = {}
        self.vol_states_5m: dict = {}
        
        # Per-asset LGBM trainers and predictors
        self.trainers: dict = {}  # (asset, is_hf) -> LGBMTrainer
        self.predictors: dict = {}  # (asset, is_hf) -> LGBMPredictor
        
        # Track last retrain time
        self.last_retrain: dict = {}  # (asset, is_hf) -> datetime
        
        # Initialize for all assets
        for asset in TOKEN_MAP.keys():
            self.data_managers[asset] = DataManager(asset)
            self.vol_states_1m[asset] = VolatilityState(asset, resolution=1)
            self.vol_states_5m[asset] = VolatilityState(asset, resolution=5)
            
            # Initialize trainers and predictors
            is_hf = asset in HF_ASSETS
            self.trainers[(asset, is_hf)] = LGBMTrainer(asset, is_hf, model_dir)
            self.predictors[(asset, is_hf)] = LGBMPredictor(
                asset, is_hf, self.trainers[(asset, is_hf)]
            )
        
        # Load persisted states
        self._load_states()
        
        # Load models
        self._load_models()
    
    def _load_states(self):
        """Load persisted states for all assets"""
        for asset in TOKEN_MAP.keys():
            self.persistence.load_asset_state(
                asset=asset,
                vol_state_1m=self.vol_states_1m[asset],
                vol_state_5m=self.vol_states_5m[asset],
                data_manager=self.data_managers[asset]
            )
    
    def _load_models(self):
        """Load trained models"""
        for (asset, is_hf), trainer in self.trainers.items():
            trainer.load_models()
    
    def _get_anchor_price(self, asset: str) -> float:
        """
        Get anchor price with fallback chain:
        1. Try Hermes API (if feed ID exists)
        2. Try latest 1-minute candle close from data manager
        3. Use fallback price from config
        """
        # Try Hermes first
        hermes_price = self.hermes_fetcher.fetch_anchor_price(asset)
        if hermes_price is not None:
            return hermes_price
        
        # Fallback to latest 1-minute candle
        data_manager = self.data_managers.get(asset)
        if data_manager is not None:
            latest_close = data_manager.get_latest_close(resolution=1)
            if latest_close is not None:
                logger.info(f"Using latest 1m candle close for {asset}: {latest_close}")
                return latest_close
        
        # Last resort: use fallback price from config
        fallback = FALLBACK_PRICES.get(asset, 100.0)
        logger.warning(f"Using fallback price for {asset}: {fallback}")
        return fallback
    
    def _should_retrain(self, asset: str, is_hf: bool) -> bool:
        """Check if model should be retrained"""
        key = (asset, is_hf)
        trainer = self.trainers.get(key)
        
        # If no models loaded, don't retrain (models should be pre-trained)
        if trainer is None or not trainer.models:
            return False
        
        # If never retrained, check if models exist on disk
        if key not in self.last_retrain:
            # Models are loaded, so they exist - don't retrain immediately
            # Set last_retrain to now to avoid immediate retraining
            self.last_retrain[key] = datetime.now(timezone.utc)
            return False
        
        last_time = self.last_retrain[key]
        now = datetime.now(timezone.utc)
        days_since = (now - last_time).total_seconds() / 86400
        
        if is_hf:
            return days_since >= HF_RETRAIN_INTERVAL_DAYS
        else:
            return days_since >= LF_RETRAIN_INTERVAL_DAYS
    
    def _retrain_model(self, asset: str, is_hf: bool):
        """Retrain LGBM model for an asset"""
        logger.info(f"Retraining {asset} ({'HF' if is_hf else 'LF'})")
        
        try:
            # Fetch 1 year of training data
            end_time = datetime.now(timezone.utc)
            data = self.benchmarks_fetcher.fetch_training_data(
                asset, resolution=1, end_time=end_time
            )
            
            if data is None:
                logger.warning(f"No training data for {asset}")
                return
            
            timestamps, opens, highs, lows, closes = parse_benchmarks_ohlc(data)
            
            if not timestamps or len(timestamps) < 100:
                logger.warning(f"Insufficient training data for {asset}: {len(timestamps)} points")
                return
            
            # Train model
            trainer = self.trainers[(asset, is_hf)]
            trainer.train(timestamps, closes, opens, highs, lows, end_time)
            trainer.save_models()
            
            # Update retrain time
            self.last_retrain[(asset, is_hf)] = datetime.now(timezone.utc)
            
            logger.info(f"Retrained {asset} ({'HF' if is_hf else 'LF'})")
            
        except Exception as e:
            logger.error(f"Error retraining {asset}: {e}", exc_info=True)
    
    def generate_paths(
        self,
        asset: str,
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
    ) -> Tuple:
        """
        Generate price paths for a simulation request
        
        Returns:
            Tuple: (start_timestamp, time_increment, [path1, path2, ...])
        """
        try:
            # Get anchor price
            anchor_price = self._get_anchor_price(asset)
            
            # Determine if HF or LF
            is_hf = asset in HF_ASSETS
            
            # Check if retraining is needed
            if self._should_retrain(asset, is_hf):
                self._retrain_model(asset, is_hf)
            
            # Get historical data for prediction
            data_manager = self.data_managers.get(asset)
            if data_manager is None:
                logger.error(f"No data manager for {asset}")
                return self._fallback_paths(asset, start_time, time_increment,
                                           time_length, num_simulations, anchor_price)
            
            # Get recent bars for feature creation
            bars_1m = data_manager.get_1m_bars()
            
            # Minimum data for feature creation (N_LAGS + rolling windows + buffer)
            MIN_BARS_FOR_FEATURES = 50
            
            if len(bars_1m) < MIN_BARS_FOR_FEATURES:
                logger.warning(
                    f"Insufficient historical data for {asset}: {len(bars_1m)} bars "
                    f"(need at least {MIN_BARS_FOR_FEATURES}). Using fallback paths."
                )
                # Use fallback paths instead of fetching (which may fail in test mode)
                return self._fallback_paths(asset, start_time, time_increment,
                                           time_length, num_simulations, anchor_price)
            
            if len(bars_1m) == 0:
                logger.error(f"No historical data for {asset}")
                return self._fallback_paths(asset, start_time, time_increment,
                                           time_length, num_simulations, anchor_price)
            
            # Extract data for prediction
            timestamps = [b.timestamp for b in bars_1m]
            closes = [b.close for b in bars_1m]
            opens = [b.open for b in bars_1m]
            highs = [b.high for b in bars_1m]
            lows = [b.low for b in bars_1m]
            
            # Predict center path using LGBM
            predictor = self.predictors[(asset, is_hf)]
            current_time = datetime.now(timezone.utc)
            center_path = predictor.predict_center_path(
                timestamps, closes, opens, highs, lows,
                current_time, time_increment, time_length
            )
            
            if not center_path:
                logger.warning(f"No center path predicted for {asset}, using constant")
                center_path = [np.log(anchor_price)] * (time_length // time_increment)
            else:
                # Validate center path - replace invalid values with log(anchor_price)
                log_anchor = np.log(anchor_price)
                validated_path = []
                for log_price in center_path:
                    if np.isfinite(log_price) and -50 < log_price < 50:
                        validated_path.append(log_price)
                    else:
                        logger.warning(f"Invalid log_price {log_price} in center path, using anchor")
                        validated_path.append(log_anchor)
                center_path = validated_path
            
            # Generate paths using uncertainty engine
            vol_state_1m = self.vol_states_1m.get(asset)
            vol_state_5m = self.vol_states_5m.get(asset)
            
            if vol_state_1m is None or vol_state_5m is None:
                logger.error(f"Volatility states not initialized for {asset}")
                return self._fallback_paths(asset, start_time, time_increment,
                                           time_length, num_simulations, anchor_price)
            
            uncertainty_engine = UncertaintyEngine(vol_state_1m, vol_state_5m, asset)
            
            result = uncertainty_engine.generate_paths(
                center_path=center_path,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                anchor_price=anchor_price,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_paths: {e}", exc_info=True)
            anchor_price = self._get_anchor_price(asset)
            return self._fallback_paths(asset, start_time, time_increment,
                                       time_length, num_simulations, anchor_price)
    
    def _fallback_paths(self, asset: str, start_time: str, time_increment: int,
                       time_length: int, num_simulations: int, 
                       anchor_price: float) -> Tuple:
        """Generate simple fallback paths"""
        import numpy as np
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        
        H = time_length // time_increment
        start_ts = int(start_dt.timestamp())
        
        # Return constant price paths
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from synth.utils.helpers import round_to_8_significant_digits
        
        formatted_paths = []
        for _ in range(num_simulations):
            path = [round_to_8_significant_digits(anchor_price)] * (H + 1)
            formatted_paths.append(path)
        
        return (start_ts, time_increment, *formatted_paths)