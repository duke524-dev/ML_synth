# Distribution Constants Guide

## Overview

The distribution is created using **EWMA + Student-t** in `uncertainty_engine.py`. The constants that control it are in `config.py`.

## Key Constants in `config.py`

### 1. Student-t Degrees of Freedom (nu)

**Location**: `NU_1M` and `NU_5M` dictionaries

**What it does**: Controls tail thickness of the distribution
- **Lower nu** (3-5): Fatter tails, more extreme price moves
- **Higher nu** (8-10): Thinner tails, closer to normal distribution

**How to set**:
```python
# In config.py
NU_1M = {
    "BTC": 9.5,   # High frequency (1-minute prompts)
    "ETH": 7.0,
    "SOL": 8.0,
    "XAU": 9.5,
}

NU_5M = {
    "BTC": 6.5,   # Low frequency (5-minute prompts)
    "ETH": 8.5,
    "SOL": 4.0,
    "XAU": 3.0,
}
```

**Important**: `nu` must be > 2.0 (variance requirement)

### 2. EWMA Half-Lives

**Location**: `HALF_LIFE_1M` and `HALF_LIFE_5M` dictionaries

**What it does**: Controls how fast volatility adapts to new information
- **Shorter** (15 min): More reactive, adapts quickly
- **Longer** (4 hours): More stable, slower adaptation

**How to set**:
```python
# In config.py (values in seconds)
HALF_LIFE_1M = {
    "BTC": 900,   # 15 minutes = 900 seconds
    "ETH": 900,
    "SOL": 1800,  # 30 minutes
    "XAU": 900,
}

HALF_LIFE_5M = {
    "BTC": 3600,   # 1 hour = 3600 seconds
    "ETH": 3600,
    "SOL": 7200,   # 2 hours
    "XAU": 3600,
}
```

### 3. Variance Mixing Parameter

**Location**: `VARIANCE_MIXING_ALPHA`

**What it does**: Mixes log return variance with range-based variance (Parkinson estimator)

**How to set**:
```python
VARIANCE_MIXING_ALPHA = 0.5  # 50% log returns, 50% range
```

- `0.0` = Only range-based variance
- `1.0` = Only log return variance  
- `0.5` = Equal mix (recommended)

### 4. Equity Off-Hours Multiplier

**Location**: `EQUITY_OFF_HOURS_MULT`

**What it does**: Reduces volatility for equities during off-market hours

**How to set**:
```python
EQUITY_OFF_HOURS_MULT = 0.6  # 60% of normal volatility
```

## How to Calibrate

### Quick Method: Use EWMA-miner Values

The default values in `config.py` are already calibrated from EWMA-miner. They should work well.

### Manual Calibration

1. **Modify constants in `config.py`**:
   ```python
   # Example: Adjust BTC HF nu
   NU_1M["BTC"] = 8.0  # Try different values
   ```

2. **Test with offline CRPS**:
   ```bash
   cd offline_CRPS_test
   python offline_crps_simple.py
   ```

3. **Compare CRPS scores** - Lower is better

4. **Iterate** until you find optimal values

### Advanced: Grid Search Calibration

You can create a calibration script similar to `EWMA-miner/calibrate_nu.py`:

```python
# Example calibration loop
for nu in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    NU_1M["BTC"] = nu
    crps_score = run_offline_crps_test()
    print(f"nu={nu}, CRPS={crps_score}")
```

## Current Default Values

The defaults are from EWMA-miner calibration:

**HF (1-minute prompts)**:
- BTC: nu=9.5, half-life=15min
- ETH: nu=7.0, half-life=15min
- SOL: nu=8.0, half-life=30min
- XAU: nu=9.5, half-life=15min

**LF (5-minute prompts)**:
- BTC: nu=6.5, half-life=1h
- ETH: nu=8.5, half-life=1h
- SOL: nu=4.0, half-life=2h
- XAU: nu=3.0, half-life=1h

## Where Constants Are Used

1. **`uncertainty_engine.py`**: Uses `NU_1M`, `NU_5M` to sample Student-t
2. **`volatility_state.py`**: Uses `HALF_LIFE_1M`, `HALF_LIFE_5M` to compute lambda
3. **`volatility_state.py`**: Uses `VARIANCE_MIXING_ALPHA` for variance proxy
4. **`uncertainty_engine.py`**: Uses `EQUITY_OFF_HOURS_MULT` for 24h equity paths

## Example: Changing Distribution Width

To make distributions wider (more uncertainty):

```python
# In config.py
NU_1M["BTC"] = 5.0  # Lower nu = fatter tails
HALF_LIFE_1M["BTC"] = 600  # Shorter half-life = more reactive
```

To make distributions narrower (less uncertainty):

```python
# In config.py
NU_1M["BTC"] = 12.0  # Higher nu = thinner tails
HALF_LIFE_1M["BTC"] = 1800  # Longer half-life = more stable
```

## Tips

1. **Start with defaults**: They're already calibrated
2. **Calibrate per asset**: Each asset may need different values
3. **Separate HF/LF**: High and low frequency prompts need different nu
4. **Test with CRPS**: Use offline CRPS to validate changes
5. **nu > 2.0**: Always ensure nu is greater than 2.0

## Quick Reference

| Constant | Location | Typical Range | Controls |
|----------|----------|---------------|----------|
| `NU_1M[asset]` | config.py | 3.0 - 10.0 | Tail thickness (HF) |
| `NU_5M[asset]` | config.py | 3.0 - 10.0 | Tail thickness (LF) |
| `HALF_LIFE_1M[asset]` | config.py | 600 - 3600s | Volatility decay (HF) |
| `HALF_LIFE_5M[asset]` | config.py | 3600 - 14400s | Volatility decay (LF) |
| `VARIANCE_MIXING_ALPHA` | config.py | 0.0 - 1.0 | Variance estimator mix |
| `EQUITY_OFF_HOURS_MULT` | config.py | 0.4 - 0.8 | Equity off-hours vol |