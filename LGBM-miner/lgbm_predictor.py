"""
LightGBM predictor for point price predictions with interpolation
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List
import lightgbm as lgb

from config import (
    HF_LEADS, LF_LEADS, HF_BUCKET_INTERVAL, LF_BUCKET_STEP
)
from lgbm_trainer import LGBMTrainer

logger = logging.getLogger(__name__)


class LGBMPredictor:
    """Predicts point prices using trained LightGBM models with interpolation"""
    
    def __init__(self, asset: str, is_hf: bool, trainer: LGBMTrainer):
        self.asset = asset
        self.is_hf = is_hf
        self.trainer = trainer
        
        # Select leads based on frequency
        if is_hf:
            self.leads = HF_LEADS
            self.bucket_interval = HF_BUCKET_INTERVAL
        else:
            self.leads = LF_LEADS
            self.bucket_step = LF_BUCKET_STEP
    
    def predict_anchor_points(
        self,
        timestamps: List[int],
        closes: List[float],
        opens: List[float],
        highs: List[float],
        lows: List[float],
        current_time: datetime,
        time_increment: int,
        time_length: int
    ) -> Dict[int, float]:
        """
        Predict point prices at anchor leads
        
        Args:
            timestamps: Historical timestamps
            closes: Historical close prices
            opens: Historical open prices
            highs: Historical high prices
            lows: Historical low prices
            current_time: Current time for prediction
            time_increment: Time increment in seconds
            time_length: Total time length in seconds
        
        Returns:
            Dict mapping lead_seconds -> predicted_log_price
        """
        # Check minimum data requirement (N_LAGS + rolling window + buffer)
        MIN_DATA_POINTS = 50  # Need at least 50 data points for features
        if len(timestamps) < MIN_DATA_POINTS:
            logger.warning(
                f"Insufficient data for {self.asset}: {len(timestamps)} points "
                f"(need at least {MIN_DATA_POINTS})"
            )
            return {}
        
        # Create features from historical data
        try:
            df = self.trainer.create_features(timestamps, closes, opens, highs, lows)
        except Exception as e:
            logger.error(f"Error creating features for {self.asset}: {e}")
            return {}
        
        if len(df) == 0:
            logger.warning(f"No features available for {self.asset}")
            return {}
        
        # Get the latest row for prediction
        latest_row = df.iloc[-1:].copy()
        
        # Predict for each anchor lead
        predictions = {}
        for lead_seconds in self.leads:
            if lead_seconds not in self.trainer.models:
                logger.warning(f"No model for {self.asset} lead={lead_seconds}s")
                continue
            
            model = self.trainer.models[lead_seconds]
            
            # Get expected feature names for this model
            expected_features = self.trainer.feature_names.get(lead_seconds, [])
            
            if expected_features:
                # Use stored feature names to ensure consistency
                # Create a dataframe with all expected features, filling missing ones with NaN
                X_pred_dict = {}
                for feat in expected_features:
                    if feat in latest_row.columns:
                        X_pred_dict[feat] = latest_row[feat].values[0]
                    else:
                        # Feature missing - fill with NaN (will be handled by model or cause error)
                        logger.warning(f"Feature {feat} missing for {self.asset} lead={lead_seconds}s")
                        X_pred_dict[feat] = np.nan
                
                # Create array in the exact order expected by model
                X_pred = np.array([[X_pred_dict[feat] for feat in expected_features]])
            else:
                # Fallback: use dynamic feature selection (may cause mismatch)
                feature_cols = [col for col in df.columns 
                               if not col.startswith('target_') 
                               and col not in ['timestamp', 'datetime', 'close', 'open', 'high', 'low']]
                X_pred = latest_row[feature_cols].values
                logger.warning(f"No stored feature names for {self.asset} lead={lead_seconds}s, using dynamic selection")
            
            # Disable shape check to allow prediction even if features don't match exactly
            # (This is a workaround - ideally features should match)
            try:
                log_price_pred = model.predict(X_pred, predict_disable_shape_check=True)[0]
            except Exception as e:
                logger.error(f"Prediction failed for {self.asset} lead={lead_seconds}s: {e}")
                continue
            
            predictions[lead_seconds] = log_price_pred
        
        return predictions
    
    def interpolate_path(
        self,
        anchor_predictions: Dict[int, float],
        time_increment: int,
        time_length: int
    ) -> List[float]:
        """
        Interpolate/extrapolate center path between anchor predictions
        
        Args:
            anchor_predictions: Dict mapping lead_seconds -> predicted_log_price
            time_increment: Time increment in seconds
            time_length: Total time length in seconds
        
        Returns:
            List of predicted log prices for each time step
        """
        if not anchor_predictions:
            logger.warning(f"No anchor predictions for {self.asset}")
            return []
        
        # Calculate number of steps
        num_steps = time_length // time_increment
        
        # Create time points for interpolation
        anchor_times = sorted(anchor_predictions.keys())
        anchor_log_prices = [anchor_predictions[t] for t in anchor_times]
        
        # Validate anchor log prices - clip to reasonable range
        # Log prices should be roughly in range (0, 20) for assets priced $1 to $485M
        anchor_log_prices = np.clip(anchor_log_prices, 0, 20)
        
        # Replace any NaN/inf values with mean of valid values
        anchor_log_prices = np.array(anchor_log_prices)
        valid_mask = np.isfinite(anchor_log_prices)
        if not valid_mask.all():
            if valid_mask.any():
                mean_val = anchor_log_prices[valid_mask].mean()
            else:
                mean_val = 10.0  # Default for ~$22k (reasonable for BTC)
            anchor_log_prices = np.where(valid_mask, anchor_log_prices, mean_val)
            logger.warning(f"Replaced invalid anchor log prices for {self.asset}")
        
        # Create target time points
        target_times = [time_increment * (i + 1) for i in range(num_steps)]
        
        # Interpolate using linear interpolation
        # Convert to numpy for easier interpolation
        anchor_times_np = np.array(anchor_times)
        anchor_log_prices_np = np.array(anchor_log_prices)
        target_times_np = np.array(target_times)
        
        # Linear interpolation
        interpolated_log_prices = np.interp(target_times_np, anchor_times_np, anchor_log_prices_np)
        
        # Final validation
        interpolated_log_prices = np.clip(interpolated_log_prices, 0, 20)
        
        return interpolated_log_prices.tolist()
    
    def predict_center_path(
        self,
        timestamps: List[int],
        closes: List[float],
        opens: List[float],
        highs: List[float],
        lows: List[float],
        current_time: datetime,
        time_increment: int,
        time_length: int
    ) -> List[float]:
        """
        Predict center path (point predictions) for the entire horizon
        
        Args:
            timestamps: Historical timestamps
            closes: Historical close prices
            opens: Historical open prices
            highs: Historical high prices
            lows: Historical low prices
            current_time: Current time for prediction
            time_increment: Time increment in seconds
            time_length: Total time length in seconds
        
        Returns:
            List of predicted log prices for each time step
        """
        # Predict anchor points
        anchor_predictions = self.predict_anchor_points(
            timestamps, closes, opens, highs, lows,
            current_time, time_increment, time_length
        )
        
        if not anchor_predictions:
            logger.warning(f"No anchor predictions for {self.asset}")
            return []
        
        # Interpolate to get full path
        center_path = self.interpolate_path(
            anchor_predictions, time_increment, time_length
        )
        
        return center_path