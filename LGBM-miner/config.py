"""
Configuration constants for LGBM miner
"""
from typing import Dict, List

# Pyth Benchmarks TradingView API
BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
HERMES_BASE_URL = "https://hermes.pyth.network/v2/updates/price/latest"

# Token mapping for Benchmarks API
TOKEN_MAP = {
    "BTC": "Crypto.BTC/USD",
    "ETH": "Crypto.ETH/USD",
    "XAU": "Crypto.XAUT/USD",
    "SOL": "Crypto.SOL/USD",
    "SPYX": "Crypto.SPYX/USD",
    "NVDAX": "Crypto.NVDAX/USD",
    "TSLAX": "Crypto.TSLAX/USD",
    "AAPLX": "Crypto.AAPLX/USD",
    "GOOGLX": "Crypto.GOOGLX/USD",
}

# Hermes feed IDs (for anchor price)
HERMES_FEED_IDS = {
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "XAU": "765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}

# Asset groups
HF_ASSETS = {"BTC", "ETH", "SOL", "XAU"}
LF_CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XAU"}
LF_EQUITY_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}

# Data retention
# Crypto: 1 year, Equities: 6 months (Pyth doesn't support 1 year for equities)
TRAINING_YEARS = 1
TRAINING_DAYS_CRYPTO = 365  # 1 year for crypto
TRAINING_DAYS_EQUITY = 180  # 6 months (half year) for equities
MINUTES_PER_DAY = 1440
DAYS_PER_YEAR = 365
BARS_1M_YEAR = DAYS_PER_YEAR * MINUTES_PER_DAY  # ~525,600 bars

def get_training_days(asset: str) -> int:
    """
    Get training days for an asset based on asset type
    
    Args:
        asset: Asset symbol
        
    Returns:
        Number of days to use for training (365 for crypto, 180 for equities)
    """
    if asset in LF_EQUITY_ASSETS:
        return TRAINING_DAYS_EQUITY
    else:
        return TRAINING_DAYS_CRYPTO

# Recency weighting half-lives (in days)
HF_RECENCY_HALF_LIFE_DAYS = 14  # High frequency: 14 days
LF_RECENCY_HALF_LIFE_DAYS = 60  # Low frequency: 60 days

# Training leads (in seconds)
# HF: 1m, 2m, 5m, 15m, 30m, 60m (+ gap prefixes)
HF_LEADS = [60, 120, 300, 900, 1800, 3600]  # 1m, 2m, 5m, 15m, 30m, 60m
HF_GAP_PREFIXES = [60, 300]  # Additional gap prefixes for HF

# LF: 5m, 30m, 3h, 24h
LF_LEADS = [300, 1800, 10800, 86400]  # 5m, 30m, 3h, 24h

# Lead bucketing for interpolation
# HF: predict every 5th minute
HF_BUCKET_INTERVAL = 300  # 5 minutes
# LF: predict every 3rd step (depends on time_increment)
LF_BUCKET_STEP = 3

# Stochastic lead sampling during training
# Subsample leads per timestamp to avoid multiplying rows by H
STOCHASTIC_LEAD_SAMPLING = True
LEADS_PER_TIMESTAMP = 3  # Sample 3 leads per timestamp during training

# Retrain cadence (in days)
HF_RETRAIN_INTERVAL_DAYS = 1  # Daily or every few days
LF_RETRAIN_INTERVAL_DAYS = 3  # Every few days to weekly

# EWMA half-lives (in seconds, converted to lambda per dt)
# Using similar values to EWMA-miner but can be recalibrated
HALF_LIFE_1M = {
    "BTC": 900,  # 15 minutes
    "ETH": 900,
    "SOL": 1800,  # 30 minutes
    "XAU": 900,
}

HALF_LIFE_5M = {
    "AAPLX": 3600,  # 1 hour
    "BTC": 3600,
    "ETH": 3600,
    "GOOGLX": 14400,  # 4 hours
    "NVDAX": 7200,  # 2 hours
    "SOL": 7200,
    "SPYX": 3600,
    "TSLAX": 3600,
    "XAU": 3600,
}

# Student-t degrees of freedom (nu)
# Can be recalibrated separately
NU_1M = {
    "BTC": 10.0,
    "ETH": 9.0,
    "SOL": 7.0,
    "XAU": 6.0,
    "SPYX": 10.0,
    "NVDAX": 10.0,
    "TSLAX": 10.0,
    "AAPLX": 10.0,
    "GOOGLX": 10.0,
}

NU_5M = {
    "BTC": 9.0,
    "ETH": 10.0,
    "SOL": 5.0,
    "XAU": 5.5,
    "SPYX": 10.0,
    "NVDAX": 10.0,
    "TSLAX": 10.0,
    "AAPLX": 10.0,
    "GOOGLX": 10.0,
}

# Legacy fallback
NU = {
    "BTC": 8.0,
    "ETH": 7.8,
    "SOL": 6.0,
    "XAU": 6.2,
    "SPYX": 10.0,
    "NVDAX": 10.0,
    "TSLAX": 10.0,
    "AAPLX": 10.0,
    "GOOGLX": 10.0,
}

# Equities off-hours volatility multiplier
EQUITY_OFF_HOURS_MULT = 0.6

# Variance mixing parameter (alpha for mixing log return and range)
VARIANCE_MIXING_ALPHA = 0.5

# Calibration parameters
KAPPA_MIN = 0.5
KAPPA_MAX = 2.0

# Startup warmup periods (for state initialization)
WARMUP_1M_HOURS = 72  # 72 hours for crypto (3 days)
WARMUP_5M_DAYS = 5    # 5 days for equity warmup

# Fallback prices
FALLBACK_PRICES: Dict[str, float] = {
    "BTC": 50000.0,
    "ETH": 3000.0,
    "SOL": 100.0,
    "XAU": 2000.0,
    "SPYX": 500.0,
    "NVDAX": 150.0,
    "TSLAX": 250.0,
    "AAPLX": 180.0,
    "GOOGLX": 150.0,
}

# Fallback volatility (sigma) for initial variance estimate
# Used when EWMA state is not yet initialized or as fallback
# Values are per-minute volatility (for 1-minute resolution)
# Convert to variance: h = sigma^2
SIGMA_MAP: Dict[str, float] = {
    "BTC": 0.00472,   # ~0.47% per minute
    "ETH": 0.00695,   # ~0.70% per minute
    "XAU": 0.00208,   # ~0.21% per minute
    "SOL": 0.00782,   # ~0.78% per minute
    "SPYX": 0.00156,  # ~0.16% per minute
    "NVDAX": 0.00342, # ~0.34% per minute
    "TSLAX": 0.00332, # ~0.33% per minute
    "AAPLX": 0.00250, # ~0.25% per minute
    "GOOGLX": 0.00332, # ~0.33% per minute
}

# LightGBM training parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4,
}

# Rolling validation split
# Use last N% of data for validation (rolling time split)
VALIDATION_SPLIT = 0.2  # 20% for validation

# Feature engineering
# Number of lag features to include
N_LAGS = 20  # Include last 20 timesteps as features
# Technical indicators
USE_TECHNICAL_INDICATORS = True