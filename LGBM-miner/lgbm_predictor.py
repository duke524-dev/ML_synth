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
        # Create features from historical data
        df = self.trainer.create_features(timestamps, closes, opens, highs, lows)
        
        if len(df) == 0:
            logger.warning(f"No features available for {self.asset}")
            return {}
        
        # Get the latest row for prediction
        latest_row = df.iloc[-1:].copy()
        
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if not col.startswith('target_') 
                       and col not in ['timestamp', 'datetime', 'close', 'open', 'high', 'low']]
        
        X_pred = latest_row[feature_cols].values
        
        # Predict for each anchor lead
        predictions = {}
        for lead_seconds in self.leads:
            if lead_seconds not in self.trainer.models:
                logger.warning(f"No model for {self.asset} lead={lead_seconds}s")
                continue
            
            model = self.trainer.models[lead_seconds]
            log_price_pred = model.predict(X_pred)[0]
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
        
        # Create target time points
        target_times = [time_increment * (i + 1) for i in range(num_steps)]
        
        # Interpolate using linear interpolation
        # Convert to numpy for easier interpolation
        anchor_times_np = np.array(anchor_times)
        anchor_log_prices_np = np.array(anchor_log_prices)
        target_times_np = np.array(target_times)
        
        # Linear interpolation
        interpolated_log_prices = np.interp(target_times_np, anchor_times_np, anchor_log_prices_np)
        
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