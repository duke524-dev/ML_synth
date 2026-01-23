# Fixed Hybrid Miner

**LightGBM + GARCH + EWMA + Student-t** hybrid approach for Synth subnet mining.

## Overview

This miner implements a hybrid forecasting approach combining:

1. **LightGBM**: Predicts the full-horizon center path (point forecasts)
2. **GARCH**: Provides forward conditional volatility curve (term structure)
3. **EWMA**: Fast regime volatility estimate and scaling factor
4. **Student-t**: Heavy-tailed innovations for realistic market behavior
5. **AR(1) Coherence Factor**: Time coherence / volatility clustering within horizon

## Architecture

### Components

- **Forecast Engine** (`forecast_engine.py`): Main orchestrator
- **LightGBM Predictor** (`lgbm_predictor.py`): Center path prediction
- **GARCH Engine** (`garch_engine.py`): Volatility curve forecasting
- **Simulation Engine** (`simulation_engine.py`): Path generation with all components
- **Volatility State** (`volatility_state.py`): EWMA volatility tracking
- **Data Components**: Data fetching, management, and persistence

### Prompt Types

- **High-Frequency (HF)**: 1-minute steps, 1-hour horizon (60 steps)
  - Assets: BTC, ETH, SOL, XAU
- **Low-Frequency (LF)**: 5-minute steps, 24-hour horizon (288 steps)
  - Crypto/Gold: BTC, ETH, SOL, XAU
  - Equities: SPYX, NVDAX, TSLAX, AAPLX, GOOGLX

## Configuration

Key constants in `config.py`:

- **EWMA half-lives**: 20 minutes (HF), 6 hours (LF)
- **Volatility blend**: 65% GARCH + 35% EWMA (HF), 80% GARCH + 20% EWMA (LF)
- **Student-t degrees of freedom**: 6 (HF), 8 (LF Crypto), 10 (LF Equity)
- **Coherence parameters**: AR(1) rho and s_vol per prompt type
- **Calibration**: Single-knob `sigma_scale` per asset per prompt type

## Usage

### Running the Miner

```bash
cd Fixed-Hybrid-miner
python miner.py
```

### Training LightGBM Models

First, train models for center path prediction:

```bash
# Train HF models
python train_models.py --asset BTC --hf

# Train LF models
python train_models.py --asset BTC --lf
```

### Calibration

Calibrate `sigma_scale` parameters:

```bash
python calibrate.py --asset BTC --hf
python calibrate.py --asset BTC --lf
```

### Testing

Test path generation:

```bash
python test_miner.py
```

## Dependencies

- `bittensor`: Bittensor framework
- `lightgbm`: LightGBM for center path prediction
- `arch`: GARCH model fitting (optional, falls back to EWMA if unavailable)
- `numpy`, `scipy`: Numerical computations
- `requests`: API calls to Pyth

## Implementation Status

✅ Core structure created
✅ Configuration system
✅ Data fetching and management
✅ Volatility state tracking
✅ GARCH engine (with fallback)
✅ Simulation engine with all components
✅ Forecast engine orchestrator
✅ Miner main entry point
✅ Calibration system
✅ Test system

⚠️ **Note**: LightGBM model training and feature engineering need to be implemented separately. The current implementation includes a simplified predictor that can load pre-trained models.

## Plan Issues Identified

1. **GARCH Implementation**: Requires `arch` library; falls back to EWMA if unavailable
2. **Calibration Initialization**: Initial `sigma_scale` values set to 1.0 (need calibration)
3. **AR(1) Coherence**: Implemented but may need tuning
4. **LightGBM Training**: Feature engineering and training pipeline need to be implemented
5. **Fallback Behavior**: Falls back to EWMA + Student-t if components fail

## Next Steps

1. Implement LightGBM training pipeline with proper feature engineering
2. Test GARCH fitting on historical data
3. Calibrate `sigma_scale` parameters using offline CRPS tests
4. Validate path generation against validator expectations
5. Optimize for CPU inference performance

## References

- Plan document: `synth_miner_fixed_hybrid_plan.docx`
- Similar miners: `EWMA-miner/`, `LGBM-miner/`
