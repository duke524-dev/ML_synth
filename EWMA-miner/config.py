"""
Configuration constants for EWMA miner
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

# Asset groups
HF_ASSETS = {"BTC", "ETH", "SOL", "XAU"}
LF_CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XAU"}
LF_EQUITY_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}

# Data retention (7 days)
RETENTION_DAYS = 7
MINUTES_PER_DAY = 1440
BARS_1M = RETENTION_DAYS * MINUTES_PER_DAY  # ~10,080 bars
BARS_5M = RETENTION_DAYS * (MINUTES_PER_DAY // 5)  # ~2,016 bars

# EWMA half-lives (in seconds, converted to lambda per dt)
# HALF_LIFE_1M = {
#     "BTC": 120 * 60,  # 120 minutes
#     "ETH": 120 * 60,
#     "SOL": 120 * 60,
#     "XAU": 240 * 60,  # 240 minutes
#     "SPYX": 240 * 60,
#     "NVDAX": 240 * 60,
#     "TSLAX": 240 * 60,
#     "AAPLX": 240 * 60,
#     "GOOGLX": 240 * 60,
# }

HALF_LIFE_1M = {
    "BTC": 900,  # 15 minutes, lambda=0.954842, avg_crps=718.699953
    "ETH": 900,  # 15 minutes, lambda=0.954842, avg_crps=933.388881
    "SOL": 1800,  # 30 minutes, lambda=0.977160, avg_crps=1227.728625
    "XAU": 900,  # 15 minutes, lambda=0.954842, avg_crps=339.870880
}

# HALF_LIFE_5M = {
#     "BTC": 12 * 3600,  # 12 hours
#     "ETH": 12 * 3600,
#     "SOL": 12 * 3600,
#     "XAU": 18 * 3600,  # 18 hours
#     "SPYX": 24 * 3600,  # 24 hours
#     "NVDAX": 24 * 3600,
#     "TSLAX": 24 * 3600,
#     "AAPLX": 24 * 3600,
#     "GOOGLX": 24 * 3600,
# }

HALF_LIFE_5M = {
    "AAPLX": 3600,  # 1 hours, lambda=0.943874, avg_crps=1419.814763
    "BTC": 3600,  # 1 hours, lambda=0.943874, avg_crps=2676.698297
    "ETH": 3600,  # 1 hours, lambda=0.943874, avg_crps=3412.704547
    "GOOGLX": 14400,  # 4 hours, lambda=0.985663, avg_crps=2067.576918
    "NVDAX": 7200,  # 2 hours, lambda=0.971532, avg_crps=1794.930340
    "SOL": 7200,  # 2 hours, lambda=0.971532, avg_crps=4431.903435
    "SPYX": 3600,  # 1 hours, lambda=0.943874, avg_crps=1316.974182
    "TSLAX": 3600,  # 1 hours, lambda=0.943874, avg_crps=1800.990672
    "XAU": 3600,  # 1 hours, lambda=0.943874, avg_crps=1402.122978
}

# Student-t degrees of freedom (nu)
# Separate values for 1-minute (high frequency) and 5-minute (low frequency) prompts
# Calibrated from calibration_results_nu_20260108_both_20260123_041210.txt
NU_1M = {
    "BTC": 9.5,  # avg_crps=717.552117 (HIGH_FREQUENCY optimized)
    "ETH": 7.0,  # avg_crps=932.977186 (HIGH_FREQUENCY optimized)
    "SOL": 8.0,  # avg_crps=1226.943205 (HIGH_FREQUENCY optimized)
    "XAU": 9.5,  # avg_crps=339.815651 (HIGH_FREQUENCY optimized)
    "SPYX": 10.0,  # Not calibrated, using default
    "NVDAX": 10.0,  # Not calibrated, using default
    "TSLAX": 10.0,  # Not calibrated, using default
    "AAPLX": 10.0,  # Not calibrated, using default
    "GOOGLX": 10.0,  # Not calibrated, using default
}

NU_5M = {
    "BTC": 6.5,  # avg_crps=2677.208480 (LOW_FREQUENCY optimized)
    "ETH": 8.5,  # avg_crps=3410.764007 (LOW_FREQUENCY optimized)
    "SOL": 4.0,  # avg_crps=4423.264927 (LOW_FREQUENCY optimized)
    "XAU": 3.0,  # avg_crps=1348.498700 (LOW_FREQUENCY optimized)
    "SPYX": 10.0,  # Not calibrated, using default
    "NVDAX": 10.0,  # Not calibrated, using default
    "TSLAX": 10.0,  # Not calibrated, using default
    "AAPLX": 10.0,  # Not calibrated, using default
    "GOOGLX": 10.0,  # Not calibrated, using default
}

# Legacy: kept for backward compatibility (fallback if not found in NU_1M/NU_5M)
# Using averages of NU_1M and NU_5M for calibrated assets
NU = {
    "BTC": 8.0,  # avg of high=9.5 and low=6.5
    "ETH": 7.8,  # avg of high=7.0 and low=8.5
    "SOL": 6.0,  # avg of high=8.0 and low=4.0
    "XAU": 6.2,  # avg of high=9.5 and low=3.0
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

# Calibration parameters (optional)
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
