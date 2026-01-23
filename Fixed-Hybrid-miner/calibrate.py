"""
Calibration system for Fixed Hybrid miner
Updates sigma_scale per asset per prompt type based on realized vs predicted variance
"""
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import json
import os

from config import (
    SIGMA_SCALE_HF, SIGMA_SCALE_LF,
    SIGMA_SCALE_MIN, SIGMA_SCALE_MAX,
    SIGMA_SCALE_UPDATE_CLIP, SIGMA_SCALE_ABS_MIN, SIGMA_SCALE_ABS_MAX,
    HF_LEADS, LF_LEADS, HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS
)

logger = logging.getLogger(__name__)


class CalibrationSystem:
    """Calibrates sigma_scale based on realized vs predicted variance"""
    
    def __init__(self, calibration_file: str = "calibration_state.json"):
        self.calibration_file = calibration_file
        self.sigma_scales_hf = SIGMA_SCALE_HF.copy()
        self.sigma_scales_lf = SIGMA_SCALE_LF.copy()
        self.load_calibration()
    
    def load_calibration(self):
        """Load calibration state from file"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.sigma_scales_hf.update(data.get('sigma_scales_hf', {}))
                    self.sigma_scales_lf.update(data.get('sigma_scales_lf', {}))
                logger.info(f"Loaded calibration from {self.calibration_file}")
            except Exception as e:
                logger.warning(f"Error loading calibration: {e}")
    
    def save_calibration(self):
        """Save calibration state to file"""
        try:
            data = {
                'sigma_scales_hf': self.sigma_scales_hf,
                'sigma_scales_lf': self.sigma_scales_lf,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved calibration to {self.calibration_file}")
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
    
    def update_sigma_scale(
        self,
        asset: str,
        is_hf: bool,
        realized_variance: float,
        predicted_variance: float,
    ) -> float:
        """
        Update sigma_scale based on realized vs predicted variance
        
        Args:
            asset: Asset symbol
            is_hf: True for high-frequency, False for low-frequency
            realized_variance: Realized variance over scored horizons
            predicted_variance: Predicted variance over scored horizons
        
        Returns:
            Updated sigma_scale value
        """
        if predicted_variance <= 0:
            logger.warning(f"Invalid predicted_variance for {asset}: {predicted_variance}")
            return self.get_sigma_scale(asset, is_hf)
        
        # Compute update factor
        ratio = realized_variance / predicted_variance
        update_factor = np.sqrt(ratio)
        
        # Clip update factor
        update_factor = np.clip(
            update_factor,
            SIGMA_SCALE_UPDATE_CLIP[0],
            SIGMA_SCALE_UPDATE_CLIP[1]
        )
        
        # Get current sigma_scale
        current_scale = self.get_sigma_scale(asset, is_hf)
        
        # Update: new_scale = current_scale * update_factor
        new_scale = current_scale * update_factor
        
        # Clip to absolute bounds
        new_scale = np.clip(new_scale, SIGMA_SCALE_ABS_MIN, SIGMA_SCALE_ABS_MAX)
        
        # Store
        if is_hf:
            self.sigma_scales_hf[asset] = new_scale
        else:
            self.sigma_scales_lf[asset] = new_scale
        
        logger.info(
            f"Updated sigma_scale for {asset} ({'HF' if is_hf else 'LF'}): "
            f"{current_scale:.3f} -> {new_scale:.3f} "
            f"(realized_var={realized_variance:.6f}, predicted_var={predicted_variance:.6f}, "
            f"ratio={ratio:.3f}, update_factor={update_factor:.3f})"
        )
        
        return new_scale
    
    def get_sigma_scale(self, asset: str, is_hf: bool) -> float:
        """Get current sigma_scale for asset"""
        if is_hf:
            return self.sigma_scales_hf.get(asset, 1.0)
        else:
            return self.sigma_scales_lf.get(asset, 1.0)
    
    def compute_realized_variance(
        self,
        actual_prices: np.ndarray,
        anchor_price: float,
        leads: list,
        time_increment: int,
    ) -> float:
        """
        Compute realized variance over scored horizons
        
        Args:
            actual_prices: Array of actual prices at each step
            anchor_price: Starting price
            leads: List of lead times in seconds
            time_increment: Time increment in seconds
        
        Returns:
            Average realized variance over horizons
        """
        if len(actual_prices) == 0:
            return 0.0
        
        variances = []
        log_anchor = np.log(anchor_price)
        
        for lead in leads:
            step = lead // time_increment
            if step < len(actual_prices):
                log_price = np.log(actual_prices[step])
                log_return = log_price - log_anchor
                variance = log_return ** 2
                variances.append(variance)
        
        if len(variances) == 0:
            return 0.0
        
        return np.mean(variances)
    
    def compute_predicted_variance(
        self,
        predicted_volatility_curve: np.ndarray,
        leads: list,
        time_increment: int,
    ) -> float:
        """
        Compute predicted variance over scored horizons
        
        Args:
            predicted_volatility_curve: Array of predicted volatilities
            leads: List of lead times in seconds
            time_increment: Time increment in seconds
        
        Returns:
            Average predicted variance over horizons
        """
        if len(predicted_volatility_curve) == 0:
            return 0.0
        
        variances = []
        
        for lead in leads:
            step = lead // time_increment
            if step < len(predicted_volatility_curve):
                vol = predicted_volatility_curve[step]
                variance = vol ** 2
                variances.append(variance)
        
        if len(variances) == 0:
            return 0.0
        
        return np.mean(variances)


def calibrate_from_crps_results(
    crps_results_file: str,
    calibration_system: CalibrationSystem,
):
    """
    Calibrate sigma_scales from offline CRPS test results
    
    This is a placeholder - actual implementation would parse CRPS results
    and compute realized vs predicted variance
    """
    logger.info(f"Calibrating from CRPS results: {crps_results_file}")
    # TODO: Implement actual calibration logic
    calibration_system.save_calibration()
