"""
ForecastEngine: Main engine for generating price path predictions
"""
import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
import numpy as np
import torch

from config import (
    HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS,
    HF_LOOKBACK_HOURS, LF_LOOKBACK_DAYS,
    FALLBACK_PRICES, SIGMA_SCALE_MIN, SIGMA_SCALE_MAX,
)
from data_fetcher import HermesFetcher
from cache_manager import CacheManager
from features import extract_hf_features, extract_lf_features, normalize_features
from model import PatchTSTModel, load_model_checkpoint
from path_sampling import sample_paths, apply_sigma_calibration
from fallback import FallbackGenerator

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Main forecast engine with routing and model management"""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.cache_manager = CacheManager()
        self.hermes_fetcher = HermesFetcher()
        self.fallback_gen = FallbackGenerator(self.cache_manager)
        
        # Model storage
        self.models: Dict[str, PatchTSTModel] = {}
        self.model_configs: Dict[str, dict] = {}
        self.normalization_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.calibration_scales: Dict[str, Dict[str, float]] = {}
        
        # Asset ID mapping
        self.asset_to_id: Dict[str, int] = {}
        self._build_asset_mapping()
        
        # Load models and artifacts
        self._load_models()
        self._load_normalization_stats()
        self._load_calibration()
    
    def _build_asset_mapping(self):
        """Build asset to ID mapping for each model type"""
        # HF model: BTC, ETH, SOL, XAU
        hf_assets = sorted(list(HF_ASSETS))
        self.asset_to_id["HF"] = {asset: i for i, asset in enumerate(hf_assets)}
        
        # LF-Crypto model: BTC, ETH, SOL, XAU
        lf_crypto_assets = sorted(list(LF_CRYPTO_ASSETS))
        self.asset_to_id["LF-crypto"] = {asset: i for i, asset in enumerate(lf_crypto_assets)}
        
        # LF-Equity model: SPYX, NVDAX, TSLAX, AAPLX, GOOGLX
        lf_equity_assets = sorted(list(LF_EQUITY_ASSETS))
        self.asset_to_id["LF-equity"] = {asset: i for i, asset in enumerate(lf_equity_assets)}
    
    def _get_model_key(self, asset: str, time_increment: int, time_length: int) -> str:
        """Determine which model to use"""
        if time_increment == 60 and time_length == 3600:
            return "HF"
        elif asset in LF_CRYPTO_ASSETS:
            return "LF-crypto"
        elif asset in LF_EQUITY_ASSETS:
            return "LF-equity"
        else:
            # Default to LF-crypto
            return "LF-crypto"
    
    def _load_models(self):
        """Load all three model checkpoints"""
        model_configs = {
            "HF": {
                "checkpoint": os.path.join(self.artifacts_dir, "current", "hf_model.pt"),
                "num_features": 15,  # HF feature count
                "num_assets": 4,
                "horizon": 60,  # 60 minutes
                "d_model": 192,
                "num_layers": 6,
            },
            "LF-crypto": {
                "checkpoint": os.path.join(self.artifacts_dir, "current", "lf_crypto_model.pt"),
                "num_features": 14,  # LF base features
                "num_assets": 4,
                "horizon": 288,  # 24h @ 5m
                "d_model": 256,
                "num_layers": 8,
            },
            "LF-equity": {
                "checkpoint": os.path.join(self.artifacts_dir, "current", "lf_equity_model.pt"),
                "num_features": 17,  # LF base + 3 session features
                "num_assets": 5,
                "horizon": 288,
                "d_model": 256,
                "num_layers": 8,
            },
        }
        
        for model_key, config in model_configs.items():
            try:
                model = load_model_checkpoint(
                    checkpoint_path=config["checkpoint"],
                    num_features=config["num_features"],
                    num_assets=config["num_assets"],
                    horizon=config["horizon"],
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    device="cpu"
                )
                self.models[model_key] = model
                self.model_configs[model_key] = config
                logger.info(f"Loaded {model_key} model")
            except Exception as e:
                logger.warning(f"Could not load {model_key} model: {e}")
                # Model will be None, will use fallback
    
    def _load_normalization_stats(self):
        """Load feature normalization statistics"""
        stats_file = os.path.join(self.artifacts_dir, "current", "normalization_stats.json")
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
                # Convert back to numpy arrays
                for model_key, stats in data.items():
                    self.normalization_stats[model_key] = {
                        "mean": np.array(stats["mean"]),
                        "std": np.array(stats["std"]),
                    }
            logger.info("Loaded normalization statistics")
        except Exception as e:
            logger.warning(f"Could not load normalization stats: {e}")
            # Will use unnormalized features
    
    def _load_calibration(self):
        """Load calibration scales"""
        calib_file = os.path.join(self.artifacts_dir, "current", "calibration.json")
        try:
            with open(calib_file, 'r') as f:
                self.calibration_scales = json.load(f)
            logger.info("Loaded calibration scales")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
            # Default to 1.0 for all
            self.calibration_scales = {}
    
    def _get_calibration_scale(self, model_key: str, asset: str) -> float:
        """Get calibration scale for model/asset"""
        scale = self.calibration_scales.get(model_key, {}).get(asset, 1.0)
        # Clamp to valid range
        return max(SIGMA_SCALE_MIN, min(SIGMA_SCALE_MAX, scale))
    
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
            Tuple in format: (start_timestamp, time_increment, [path1, path2, ...])
            where each path is a list of prices
        """
        try:
            # Parse start time
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            
            # Determine model
            model_key = self._get_model_key(asset, time_increment, time_length)
            is_hf = (model_key == "HF")
            resolution = 1 if is_hf else 5
            
            # Calculate horizon
            H = time_length // time_increment
            if H * time_increment != time_length:
                raise ValueError(f"time_length must be divisible by time_increment")
            
            # Get anchor price
            anchor_price = self.hermes_fetcher.get_anchor_price_with_fallback(asset)
            
            # Get cache
            cache = self.cache_manager.get_cache(asset, resolution, is_hf)
            cache.refresh()  # Try to refresh if needed
            
            # Get lookback window
            if is_hf:
                lookback_points = HF_LOOKBACK_HOURS * 60  # 1440 points
            else:
                lookback_points = LF_LOOKBACK_DAYS * 288  # 6048 points (21 days * 288 per day)
            
            # Get features window
            window_data = cache.get_features_window(start_dt, lookback_points)
            
            if window_data is None:
                logger.warning(f"Insufficient data for {asset}, using fallback")
                return self.fallback_gen.generate_paths(
                    asset, start_time, time_increment, time_length,
                    num_simulations, anchor_price
                )
            
            timestamps, prices = window_data
            
            # Extract features
            if is_hf:
                features = extract_hf_features(timestamps, prices)
            else:
                features = extract_lf_features(timestamps, prices, asset)
            
            # Normalize features
            if model_key in self.normalization_stats:
                stats = self.normalization_stats[model_key]
                features = normalize_features(
                    features, stats["mean"], stats["std"]
                )
            
            # Check model availability
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not loaded, using fallback")
                return self.fallback_gen.generate_paths(
                    asset, start_time, time_increment, time_length,
                    num_simulations, anchor_price
                )
            
            model = self.models[model_key]
            
            # Get asset ID
            asset_id_map = self.asset_to_id.get(model_key, {})
            if asset not in asset_id_map:
                logger.warning(f"Asset {asset} not in {model_key} mapping, using fallback")
                return self.fallback_gen.generate_paths(
                    asset, start_time, time_increment, time_length,
                    num_simulations, anchor_price
                )
            
            asset_id = asset_id_map[asset]
            
            # Prepare input
            # Use last features for prediction (most recent context)
            features_tensor = torch.from_numpy(features[-1:]).unsqueeze(0)  # [1, L, C]
            # Pad/truncate to model's expected length if needed
            # For now, assume model can handle variable length or we pad
            
            asset_ids_tensor = torch.tensor([asset_id], dtype=torch.long)
            
            # Forward pass
            with torch.no_grad():
                mu, log_sigma, nu = model(features_tensor, asset_ids_tensor)
            
            # Convert to numpy
            mu = mu.squeeze(0).cpu().numpy()  # [H]
            log_sigma = log_sigma.squeeze(0).cpu().numpy()  # [H]
            nu = nu.item()  # scalar
            
            # Truncate to actual horizon if model horizon is larger
            if len(mu) > H:
                mu = mu[:H]
                log_sigma = log_sigma[:H]
            
            # Apply calibration
            sigma_scale = self._get_calibration_scale(model_key, asset)
            log_sigma = apply_sigma_calibration(log_sigma, sigma_scale)
            
            # Check for NaN/inf
            if not (np.isfinite(mu).all() and np.isfinite(log_sigma).all() and np.isfinite(nu)):
                logger.warning("Model output contains NaN/inf, using fallback")
                return self.fallback_gen.generate_paths(
                    asset, start_time, time_increment, time_length,
                    num_simulations, anchor_price
                )
            
            # Sample paths
            paths = sample_paths(
                mu=mu,
                log_sigma=log_sigma,
                nu=nu,
                num_simulations=num_simulations,
                anchor_price=anchor_price,
                mode=model_key,
                use_latent_vol=True,
            )  # [N, H+1]
            
            # Format output to match validator expectations
            start_timestamp = int(start_dt.timestamp())
            
            # Round prices to 8 significant digits
            # Import here to avoid circular dependency
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
            
            return (start_timestamp, time_increment, *formatted_paths)
            
        except Exception as e:
            logger.error(f"Error in generate_paths: {e}", exc_info=True)
            # Fallback on any error
            anchor_price = self.hermes_fetcher.get_anchor_price_with_fallback(asset)
            return self.fallback_gen.generate_paths(
                asset, start_time, time_increment, time_length,
                num_simulations, anchor_price
            )
