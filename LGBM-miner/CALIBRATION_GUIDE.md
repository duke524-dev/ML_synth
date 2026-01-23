# Distribution Constants Calibration Guide

## Overview

The LGBM-miner uses **EWMA + Student-t distribution** to generate uncertainty around the center path. The key constants that control the distribution are:

1. **Student-t degrees of freedom (nu)** - Controls tail thickness
2. **EWMA half-lives** - Controls volatility decay rate
3. **Variance mixing parameter (alpha)** - Mixes log returns and range-based variance
4. **Equity off-hours multiplier** - Adjusts volatility for equities during off-hours

## Constants Location

All constants are defined in `config.py`:

### 1. Student-t Degrees of Freedom (nu)

Controls the tail thickness of the distribution:
- **Lower nu** (e.g., 3-5): Fatter tails, more extreme moves
- **Higher nu** (e.g., 8-10): Thinner tails, closer to normal distribution

```python
# High frequency (1-minute prompts)
NU_1M = {
    "BTC": 9.5,
    "ETH": 7.0,
    "SOL": 8.0,
    "XAU": 9.5,
    ...
}

# Low frequency (5-minute prompts)
NU_5M = {
    "BTC": 6.5,
    "ETH": 8.5,
    "SOL": 4.0,
    "XAU": 3.0,
    ...
}
```

**Note**: `nu` must be > 2.0 for variance to exist.

### 2. EWMA Half-Lives

Controls how fast volatility adapts to new information:
- **Shorter half-life** (e.g., 15 min): Faster adaptation, more reactive
- **Longer half-life** (e.g., 4 hours): Slower adaptation, more stable

```python
# 1-minute resolution half-lives (in seconds)
HALF_LIFE_1M = {
    "BTC": 900,   # 15 minutes
    "ETH": 900,
    "SOL": 1800,  # 30 minutes
    "XAU": 900,
}

# 5-minute resolution half-lives (in seconds)
HALF_LIFE_5M = {
    "BTC": 3600,   # 1 hour
    "ETH": 3600,
    "SOL": 7200,   # 2 hours
    "XAU": 3600,
    ...
}
```

### 3. Variance Mixing Parameter

Mixes log return variance with range-based variance (Parkinson estimator):

```python
VARIANCE_MIXING_ALPHA = 0.5  # 50% log returns, 50% range
```

- **alpha = 0.0**: Use only range-based variance
- **alpha = 1.0**: Use only log return variance
- **alpha = 0.5**: Equal mix (default)

### 4. Equity Off-Hours Multiplier

Reduces volatility for equities during off-market hours:

```python
EQUITY_OFF_HOURS_MULT = 0.6  # 60% of normal volatility
```

## How to Calibrate

### Option 1: Use Calibration Script (Recommended)

Run the calibration script to find optimal `nu` values:

```bash
cd LGBM-miner
python calibrate_nu.py --asset BTC --prompt-type HIGH_FREQUENCY
python calibrate_nu.py --asset BTC --prompt-type LOW_FREQUENCY
```

This will:
1. Test different `nu` values
2. Calculate CRPS scores for each
3. Find the `nu` that minimizes CRPS
4. Update `config.py` with optimal values

### Option 2: Manual Calibration

1. **Start with EWMA-miner values**: The default values in `config.py` are from EWMA-miner calibration
2. **Test different values**: Modify `NU_1M` or `NU_5M` in `config.py`
3. **Run offline CRPS test**: Use `offline_CRPS_test/offline_crps_simple.py` to evaluate
4. **Iterate**: Adjust values based on CRPS scores

### Option 3: Grid Search

Use the calibration script with a custom grid:

```bash
python calibrate_nu.py --asset BTC --nu-grid "3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
```

## Calibration Workflow

1. **Train models first**: Ensure you have trained LGBM models
   ```bash
   python train_models.py --asset BTC
   ```

2. **Run calibration**: Calibrate nu for each asset/prompt type
   ```bash
   python calibrate_nu.py --asset BTC --prompt-type HIGH_FREQUENCY
   python calibrate_nu.py --asset BTC --prompt-type LOW_FREQUENCY
   ```

3. **Update config**: The script will suggest optimal values, update `config.py` manually

4. **Verify**: Run offline CRPS test to verify improvements

## Understanding the Constants

### Student-t nu (Degrees of Freedom)

- **Lower nu** = More extreme tails = Higher uncertainty
- **Higher nu** = Closer to normal = Lower uncertainty
- Typical range: 3.0 to 10.0
- Must be > 2.0

### EWMA Half-Life

- **Shorter** = More reactive to recent volatility
- **Longer** = More stable, less reactive
- Typical range: 15 minutes to 4 hours

### Variance Mixing Alpha

- Controls balance between return-based and range-based variance
- Range-based (Parkinson) uses high-low information
- Default 0.5 works well for most assets

## Example: Calibrating BTC

```bash
# 1. Train models
python train_models.py --asset BTC

# 2. Calibrate HF nu
python calibrate_nu.py --asset BTC --prompt-type HIGH_FREQUENCY

# 3. Calibrate LF nu  
python calibrate_nu.py --asset BTC --prompt-type LOW_FREQUENCY

# 4. Update config.py with suggested values

# 5. Test
python test_miner.py
```

## Current Default Values

The current defaults in `config.py` are from EWMA-miner calibration:

- **HF nu**: BTC=9.5, ETH=7.0, SOL=8.0, XAU=9.5
- **LF nu**: BTC=6.5, ETH=8.5, SOL=4.0, XAU=3.0
- **Half-lives**: Similar to EWMA-miner (15min-4h range)

These should work well, but you can recalibrate for your specific use case.