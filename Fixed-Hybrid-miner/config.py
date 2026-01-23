"""
Configuration constants for Fixed Hybrid miner
LightGBM + GARCH + EWMA + Student-t
"""
from typing import Dict

# Pyth Benchmarks TradingView API
BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
HERMES_BASE_URL = "https://hermes.pyth.network/v2/updates/price/latest"

# Token mapping for Benchmarks API
TOKEN_MAP = {
    "BTC": "Crypto.BTC/USD",
    "ETH": "Crypto.ETH/USD",
    "XAU": "Crypto.XAUT/USD",  # Note: XAUT for gold
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

# Data retention (1 year for training)
TRAINING_YEARS = 1
MINUTES_PER_DAY = 1440
DAYS_PER_YEAR = 365
BARS_1M_YEAR = DAYS_PER_YEAR * MINUTES_PER_DAY  # ~525,600 bars

# Recency weighting half-lives (in days) - from plan
HF_RECENCY_HALF_LIFE_DAYS = 21  # High frequency: 21 days
LF_RECENCY_HALF_LIFE_DAYS = 90  # Low frequency: 90 days

# Training leads (in seconds) - from plan
# HF: 1m, 2m, 5m, 15m, 30m, 60m
HF_LEADS = [60, 120, 300, 900, 1800, 3600]  # 1m, 2m, 5m, 15m, 30m, 60m

# LF: 5m, 30m, 3h, 24h
LF_LEADS = [300, 1800, 10800, 86400]  # 5m, 30m, 3h, 24h

# Prompt specifications - from plan
HF_HORIZON_STEPS = 60  # 60 x 1-minute steps = 1 hour
LF_HORIZON_STEPS = 288  # 288 x 5-minute steps = 24 hours

# EWMA half-lives (in seconds) - from plan
EWMA_HALF_LIFE_HF = 20 * 60  # 20 minutes
EWMA_HALF_LIFE_LF = 6 * 3600  # 6 hours

# GARCH configuration - from plan
# Use 5-minute returns for GARCH (downscale to 1m for HF)
GARCH_INPUT_RESOLUTION = 5  # Always use 5-minute returns
GARCH_DOWNSCALE_FACTOR = 5  # For HF: sigma_1m = sigma_5m / sqrt(5)

# Volatility blend weight w - from plan
VOL_BLEND_WEIGHT_HF = 0.65  # 65% GARCH, 35% EWMA
VOL_BLEND_WEIGHT_LF = 0.80  # 80% GARCH, 20% EWMA

# Student-t degrees of freedom (nu) - from plan
NU_HF = 6
NU_LF_CRYPTO = 8
NU_LF_EQUITY = 10

# Coherence parameters (AR(1) latent factor) - from plan
COHERENCE_RHO_HF = 0.98
COHERENCE_RHO_LF_CRYPTO = 0.99
COHERENCE_RHO_LF_EQUITY = 0.995

COHERENCE_S_VOL_HF = 0.15
COHERENCE_S_VOL_LF_CRYPTO = 0.10
COHERENCE_S_VOL_LF_EQUITY = 0.08

# Calibration parameters
# Single-knob sigma_scale per asset per prompt type
SIGMA_SCALE_MIN = 0.7
SIGMA_SCALE_MAX = 1.4
SIGMA_SCALE_UPDATE_CLIP = (0.7, 1.4)  # Clip per update
SIGMA_SCALE_ABS_MIN = 0.3  # Absolute minimum
SIGMA_SCALE_ABS_MAX = 3.0  # Absolute maximum

# Initial sigma_scale values (will be calibrated)
SIGMA_SCALE_HF: Dict[str, float] = {
    "BTC": 1.0,
    "ETH": 1.0,
    "SOL": 1.0,
    "XAU": 1.0,
}

SIGMA_SCALE_LF: Dict[str, float] = {
    "BTC": 1.0,
    "ETH": 1.0,
    "SOL": 1.0,
    "XAU": 1.0,
    "SPYX": 1.0,
    "NVDAX": 1.0,
    "TSLAX": 1.0,
    "AAPLX": 1.0,
    "GOOGLX": 1.0,
}

# Calibration update frequency
CALIBRATION_UPDATE_INTERVAL_HOURS = 24  # Daily updates

# Equities off-hours volatility multiplier
EQUITY_OFF_HOURS_MULT = 0.6

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
VALIDATION_SPLIT = 0.2  # 20% for validation

# Feature engineering
N_LAGS = 20  # Include last 20 timesteps as features
USE_TECHNICAL_INDICATORS = True

# GARCH model parameters
GARCH_ORDER = (1, 1)  # GARCH(1,1)
GARCH_DIST = 't'  # Student-t distribution

# Retrain cadence (in days)
HF_RETRAIN_INTERVAL_DAYS = 1  # Daily or every few days
LF_RETRAIN_INTERVAL_DAYS = 3  # Every few days to weekly

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
