"""
Configuration constants for PatchTST + Student-t miner
"""
from typing import Dict

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

# Model routing
HF_ASSETS = {"BTC", "ETH", "SOL", "XAU"}
LF_CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XAU"}
LF_EQUITY_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}

# Cache requirements
HF_CACHE_MIN_HOURS = 30
LF_CACHE_MIN_DAYS = 23

# Cache refresh intervals (seconds)
HF_REFRESH_INTERVAL_MIN = 120  # 2 minutes
HF_REFRESH_INTERVAL_MAX = 300  # 5 minutes
LF_REFRESH_INTERVAL_MIN = 1800  # 30 minutes
LF_REFRESH_INTERVAL_MAX = 3600  # 60 minutes

# Lookback windows
HF_LOOKBACK_HOURS = 24  # 24 hours @ 1m = 1440 points
LF_LOOKBACK_DAYS = 21  # 21 days @ 5m = 6048 points

# Missing data thresholds
HF_MISSING_THRESHOLD = 0.05  # 5%
LF_MISSING_THRESHOLD = 0.02  # 2%

# Latent volatility clustering parameters
LATENT_VOL_PARAMS = {
    "HF": {"rho": 0.98, "s_vol": 0.15},
    "LF-crypto": {"rho": 0.99, "s_vol": 0.10},
    "LF-equity": {"rho": 0.995, "s_vol": 0.08},
}

# Calibration horizons
HF_CALIBRATION_HORIZONS = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
LF_CALIBRATION_HORIZONS = [1, 6, 36, 288]

# Calibration weights
HF_CALIBRATION_WEIGHTS = {
    1: 0.18, 2: 0.12, 5: 0.10, 10: 0.08, 15: 0.08, 20: 0.06, 25: 0.05,
    30: 0.08, 35: 0.05, 40: 0.05, 45: 0.05, 50: 0.04, 55: 0.03, 60: 0.03
}
LF_CALIBRATION_WEIGHTS = {
    1: 0.10, 6: 0.25, 36: 0.35, 288: 0.30
}

# Calibration EMA alphas
HF_CALIBRATION_ALPHA = 0.4
LF_CALIBRATION_ALPHA = 0.2

# Calibration clamps
SIGMA_SCALE_MIN = 0.5
SIGMA_SCALE_MAX = 2.0
HF_MAX_MOVE = 0.10  # ±10%
LF_MAX_MOVE = 0.07  # ±7%

# Calibration minimum counts
HF_MIN_COUNT = 150
LF_MIN_COUNT = 120

# Fallback prices (if Hermes fails)
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
