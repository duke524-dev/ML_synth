"""
LightGBM predictor for center path prediction
Simplified version for Fixed Hybrid miner
"""
import logging
import numpy as np
from typing import Dict, List, Optional
import lightgbm as lgb

logger = logging.getLogger(__name__)


class LGBMPredictor:
    """Predicts center path using trained LightGBM models"""
    
    def __init__(self, asset: str, is_hf: bool, model_dir: str = "models"):
        self.asset = asset
        self.is_hf = is_hf
        self.model_dir = model_dir
        self.models: Dict[int, lgb.Booster] = {}
        self.freq_label = "HF" if is_hf else "LF"
    
    def load_models(self) -> bool:
        """Load trained models for all leads"""
        from config import HF_LEADS, LF_LEADS
        
        leads = HF_LEADS if self.is_hf else LF_LEADS
        loaded = False
        
        for lead in leads:
            model_path = f"{self.model_dir}/{self.asset}_{self.freq_label}_{lead}.pkl"
            try:
                model = lgb.Booster(model_file=model_path)
                self.models[lead] = model
                loaded = True
                logger.info(f"Loaded model for {self.asset} {self.freq_label} lead={lead}s")
            except Exception as e:
                logger.warning(f"Could not load model {model_path}: {e}")
        
        return loaded
    
    def predict_center_path(
        self,
        features: np.ndarray,
        time_increment: int,
        time_length: int,
    ) -> Optional[List[float]]:
        """
        Predict center path (log prices) for full horizon
        
        Args:
            features: Feature vector for prediction
            time_increment: Time increment in seconds
            time_length: Total time length in seconds
        
        Returns:
            List of predicted log prices (one per step) or None
        """
        from config import HF_LEADS, LF_LEADS
        
        if not self.models:
            logger.warning(f"No models loaded for {self.asset}")
            return None
        
        leads = HF_LEADS if self.is_hf else LF_LEADS
        H = time_length // time_increment
        
        # Predict at anchor points
        anchor_predictions = {}
        for lead in leads:
            if lead in self.models:
                pred = self.models[lead].predict(features.reshape(1, -1))[0]
                anchor_predictions[lead] = pred
        
        if not anchor_predictions:
            return None
        
        # Interpolate/extrapolate to full horizon
        center_path = self._interpolate_path(anchor_predictions, time_increment, H)
        
        return center_path
    
    def _interpolate_path(
        self,
        anchor_predictions: Dict[int, float],
        time_increment: int,
        H: int
    ) -> List[float]:
        """Interpolate between anchor predictions"""
        # Convert leads to step indices
        anchor_steps = {lead // time_increment: log_price 
                       for lead, log_price in anchor_predictions.items()}
        
        # Sort by step
        sorted_steps = sorted(anchor_steps.keys())
        
        if not sorted_steps:
            return [0.0] * (H + 1)
        
        # Interpolate
        center_path = []
        for t in range(H + 1):
            if t in anchor_steps:
                center_path.append(anchor_steps[t])
            elif t < sorted_steps[0]:
                # Extrapolate backward
                center_path.append(anchor_steps[sorted_steps[0]])
            elif t > sorted_steps[-1]:
                # Extrapolate forward (linear)
                if len(sorted_steps) >= 2:
                    # Use last two points for linear extrapolation
                    s1, s2 = sorted_steps[-2], sorted_steps[-1]
                    p1, p2 = anchor_steps[s1], anchor_steps[s2]
                    slope = (p2 - p1) / (s2 - s1) if s2 > s1 else 0
                    center_path.append(p2 + slope * (t - s2))
                else:
                    center_path.append(anchor_steps[sorted_steps[-1]])
            else:
                # Interpolate between anchors
                # Find surrounding anchors
                for i in range(len(sorted_steps) - 1):
                    if sorted_steps[i] <= t < sorted_steps[i + 1]:
                        s1, s2 = sorted_steps[i], sorted_steps[i + 1]
                        p1, p2 = anchor_steps[s1], anchor_steps[s2]
                        # Linear interpolation
                        alpha = (t - s1) / (s2 - s1) if s2 > s1 else 0
                        center_path.append(p1 + alpha * (p2 - p1))
                        break
        
        return center_path
