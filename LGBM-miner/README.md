# LGBM Miner

LightGBM-based miner that combines:
- **LightGBM** for point price predictions
- **EWMA** (Exponentially Weighted Moving Average) for volatility estimation
- **Student-t distribution** for generating realistic price paths

## Architecture

### Components

1. **LGBM Trainer** (`lgbm_trainer.py`)
   - Trains LightGBM models with exponential recency weighting
   - HF: 14-day half-life
   - LF: 60-day half-life
   - Uses rolling time split for validation (never random)
   - Stochastic lead sampling to avoid multiplying rows by H

2. **LGBM Predictor** (`lgbm_predictor.py`)
   - Predicts point prices at anchor leads
   - HF: 1m, 2m, 5m, 15m, 30m, 60m
   - LF: 5m, 30m, 3h, 24h
   - Interpolates/extrapolates center path between anchors

3. **Uncertainty Engine** (`uncertainty_engine.py`)
   - EWMA + calibration scalars (regime controller)
   - Student-t + coherence (distribution generator)
   - Generates 1000 paths around center path

4. **Forecast Engine** (`forecast_engine.py`)
   - Orchestrates LGBM prediction + uncertainty engine
   - Handles retraining cadence:
     - HF: daily or every few days
     - LF: every few days to weekly

## Training

- Uses 1 year of historical data where available
- Applies exponential recency weighting during training
- Rolling time split for validation (no random split)
- Trains only the leads that matter for scoring
- Buckets leads (e.g., predict every 5th minute for HF), then interpolates

## Dependencies

Additional dependencies beyond base requirements:
- `lightgbm` - Gradient boosting framework
- `scipy` - For Student-t distribution
- `msgpack` - For state persistence
- `pandas` - For data manipulation (should already be in requirements.txt)

Install with:
```bash
pip install lightgbm scipy msgpack
```

Or add to requirements.txt:
```
lightgbm>=4.0.0
scipy>=1.10.0
msgpack>=1.0.0
```

## Usage

### 1. Train Models First

Before running the miner, you need to train the LGBM models:

```bash
cd LGBM-miner

# Train all assets (HF and LF)
python train_models.py

# Train only high-frequency assets
python train_models.py --hf-only

# Train only low-frequency assets
python train_models.py --lf-only

# Train a specific asset
python train_models.py --asset BTC

# Use custom model directory
python train_models.py --model-dir my_models
```

Training will:
- Fetch 1 year of historical data for each asset
- Apply exponential recency weighting (HF: 14 days, LF: 60 days)
- Train models for all anchor leads
- Save models to the `models/` directory

**Note:** Training can take a while, especially for the first time as it fetches 1 year of data for each asset.

### 2. Test the Miner

After training, test path generation:

```bash
python test_miner.py
```

### 3. Run the Miner

```bash
python miner.py
```

## Configuration

Key configuration parameters in `config.py`:

- `HF_RECENCY_HALF_LIFE_DAYS = 14` - High frequency recency weighting
- `LF_RECENCY_HALF_LIFE_DAYS = 60` - Low frequency recency weighting
- `HF_LEADS = [60, 120, 300, 900, 1800, 3600]` - HF anchor leads (seconds)
- `LF_LEADS = [300, 1800, 10800, 86400]` - LF anchor leads (seconds)
- `HF_RETRAIN_INTERVAL_DAYS = 1` - HF retrain cadence
- `LF_RETRAIN_INTERVAL_DAYS = 3` - LF retrain cadence

## State Management

States are persisted in the `state/` directory:
- Volatility states (EWMA)
- Historical data (1m and 5m bars)

Models are saved in the `models/` directory:
- One model per asset/frequency/lead combination