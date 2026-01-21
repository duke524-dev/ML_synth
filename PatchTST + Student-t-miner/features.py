"""
Feature engineering for HF and LF models
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import pytz

from config import HF_ASSETS, LF_EQUITY_ASSETS

logger = logging.getLogger(__name__)


def compute_log_returns(prices: List[float]) -> np.ndarray:
    """Compute log returns from prices"""
    prices_arr = np.array(prices, dtype=np.float64)
    returns = np.diff(np.log(prices_arr))
    return returns


def compute_rolling_std(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation"""
    if len(returns) < window:
        # Pad with first value
        padded = np.pad(returns, (window - 1, 0), mode='edge')
    else:
        padded = returns
    
    result = np.zeros_like(returns)
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        window_returns = padded[start:i+1]
        if len(window_returns) > 0:
            result[i] = np.std(window_returns)
        else:
            result[i] = 0.0
    
    return result


def compute_rolling_mean(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean"""
    if len(returns) < window:
        padded = np.pad(returns, (window - 1, 0), mode='edge')
    else:
        padded = returns
    
    result = np.zeros_like(returns)
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        window_returns = padded[start:i+1]
        if len(window_returns) > 0:
            result[i] = np.mean(window_returns)
        else:
            result[i] = 0.0
    
    return result


def compute_rolling_absmean(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean of absolute values"""
    abs_returns = np.abs(returns)
    return compute_rolling_mean(abs_returns, window)


def extract_time_features(timestamps: List[int], resolution_minutes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract time-based features
    
    Returns:
        (sin_hour, cos_hour, sin_dow, cos_dow)
    """
    dt_objs = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]
    
    hours = np.array([dt.hour for dt in dt_objs], dtype=np.float32)
    day_of_week = np.array([dt.weekday() for dt in dt_objs], dtype=np.float32)
    
    # Normalize to [0, 2*pi]
    sin_hour = np.sin(2 * np.pi * hours / 24.0)
    cos_hour = np.cos(2 * np.pi * hours / 24.0)
    sin_dow = np.sin(2 * np.pi * day_of_week / 7.0)
    cos_dow = np.cos(2 * np.pi * day_of_week / 7.0)
    
    return sin_hour, cos_hour, sin_dow, cos_dow


def extract_session_features(
    timestamps: List[int], resolution_minutes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract equity session features (America/New_York timezone)
    
    Returns:
        (is_session_open, minutes_to_open, minutes_to_close)
    """
    ny_tz = pytz.timezone('America/New_York')
    
    is_open = []
    mins_to_open = []
    mins_to_close = []
    
    for ts in timestamps:
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        dt_ny = dt_utc.astimezone(ny_tz)
        
        hour = dt_ny.hour
        minute = dt_ny.minute
        weekday = dt_ny.weekday()  # 0=Monday, 6=Sunday
        
        # Regular session: 9:30 AM - 4:00 PM ET, Monday-Friday
        if weekday < 5:  # Monday-Friday
            session_open_minutes = 9 * 60 + 30  # 9:30 AM
            session_close_minutes = 16 * 60  # 4:00 PM
            current_minutes = hour * 60 + minute
            
            if session_open_minutes <= current_minutes < session_close_minutes:
                # Session is open
                is_open.append(1.0)
                mins_to_open.append(0.0)
                mins_to_close.append(float(session_close_minutes - current_minutes))
            else:
                # Session is closed
                is_open.append(0.0)
                
                if current_minutes < session_open_minutes:
                    # Before open today
                    mins_to_open.append(float(session_open_minutes - current_minutes))
                else:
                    # After close today, next open is tomorrow
                    mins_to_open.append(float(24 * 60 - current_minutes + session_open_minutes))
                
                mins_to_close.append(0.0)
        else:
            # Weekend
            is_open.append(0.0)
            # Next Monday 9:30 AM
            days_until_monday = (7 - weekday) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            mins_to_open.append(float(days_until_monday * 24 * 60 - hour * 60 - minute + 9 * 60 + 30))
            mins_to_close.append(0.0)
        
        # Clip values
        mins_to_open[-1] = max(0.0, min(780.0, mins_to_open[-1]))
        mins_to_close[-1] = max(0.0, min(390.0, mins_to_close[-1]))
    
    return np.array(is_open), np.array(mins_to_open), np.array(mins_to_close)


def extract_hf_features(
    timestamps: List[int], prices: List[float]
) -> np.ndarray:
    """
    Extract HF features (1-minute resolution)
    
    Returns:
        Array of shape [L, C] where C ~ 15 channels
    """
    returns = compute_log_returns(prices)
    
    # Rolling std windows (in minutes): 2, 5, 15, 30, 60, 240
    std_2 = compute_rolling_std(returns, 2)
    std_5 = compute_rolling_std(returns, 5)
    std_15 = compute_rolling_std(returns, 15)
    std_30 = compute_rolling_std(returns, 30)
    std_60 = compute_rolling_std(returns, 60)
    std_240 = compute_rolling_std(returns, 240)
    
    # Rolling means
    mean_15 = compute_rolling_mean(returns, 15)
    mean_60 = compute_rolling_mean(returns, 60)
    
    # Rolling abs means
    absmean_15 = compute_rolling_absmean(returns, 15)
    absmean_60 = compute_rolling_absmean(returns, 60)
    
    # Time features
    sin_hour, cos_hour, sin_dow, cos_dow = extract_time_features(
        timestamps[1:], resolution_minutes=1
    )
    
    # Stack features: [r_t, std_2, std_5, std_15, std_30, std_60, std_240,
    #                  mean_15, mean_60, absmean_15, absmean_60,
    #                  sin_hour, cos_hour, sin_dow, cos_dow]
    features = np.stack([
        returns,  # r_t
        std_2, std_5, std_15, std_30, std_60, std_240,
        mean_15, mean_60,
        absmean_15, absmean_60,
        sin_hour, cos_hour, sin_dow, cos_dow
    ], axis=1)
    
    return features.astype(np.float32)


def extract_lf_features(
    timestamps: List[int], prices: List[float], asset: str
) -> np.ndarray:
    """
    Extract LF features (5-minute resolution)
    
    Returns:
        Array of shape [L, C] where C depends on asset type
    """
    returns = compute_log_returns(prices)
    
    # Rolling std windows (in 5m steps): 1, 6, 36, 288, 864
    # (5m, 30m, 3h, 24h, 3d)
    std_1 = compute_rolling_std(returns, 1)
    std_6 = compute_rolling_std(returns, 6)
    std_36 = compute_rolling_std(returns, 36)
    std_288 = compute_rolling_std(returns, 288)
    std_864 = compute_rolling_std(returns, 864)
    
    # Rolling means
    mean_36 = compute_rolling_mean(returns, 36)
    mean_288 = compute_rolling_mean(returns, 288)
    
    # Rolling abs means
    absmean_36 = compute_rolling_absmean(returns, 36)
    absmean_288 = compute_rolling_absmean(returns, 288)
    
    # Time features
    sin_hour, cos_hour, sin_dow, cos_dow = extract_time_features(
        timestamps[1:], resolution_minutes=5
    )
    
    # Base features
    features_list = [
        returns,  # r_t
        std_1, std_6, std_36, std_288, std_864,
        mean_36, mean_288,
        absmean_36, absmean_288,
        sin_hour, cos_hour, sin_dow, cos_dow
    ]
    
    # Add session features for equities
    if asset in LF_EQUITY_ASSETS:
        is_open, mins_to_open, mins_to_close = extract_session_features(
            timestamps[1:], resolution_minutes=5
        )
        features_list.extend([is_open, mins_to_open, mins_to_close])
    
    features = np.stack(features_list, axis=1)
    return features.astype(np.float32)


def normalize_features(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Normalize features using stored mean/std"""
    # Avoid division by zero
    std_safe = np.where(std > 1e-8, std, 1.0)
    normalized = (features - mean) / std_safe
    
    # Clamp extreme values (optional)
    normalized = np.clip(normalized, -10.0, 10.0)
    
    return normalized
