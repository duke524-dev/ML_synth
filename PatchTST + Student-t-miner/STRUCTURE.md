# Project Structure

## Files Created

### Core Modules
- `__init__.py` - Package initialization
- `config.py` - Configuration constants (API endpoints, token maps, parameters)
- `data_fetcher.py` - Benchmarks and Hermes API clients
- `cache_manager.py` - Rolling cache for price data
- `features.py` - Feature engineering (HF/LF features, session features)
- `model.py` - PatchTST model architecture
- `path_sampling.py` - Student-t path sampling with latent volatility
- `forecast_engine.py` - Main forecast engine with routing
- `fallback.py` - Fallback generator for error cases
- `miner.py` - DukeMiner1 miner class integration

### Offline Scripts
- `train.py` - Model training script
- `calibrate.py` - Calibration script for sigma_scale
- `cron_jobs.py` - Cron job automation handler

### Testing & Documentation
- `test_acceptance.py` - Acceptance tests
- `README.md` - Main documentation
- `STRUCTURE.md` - This file

### Utilities
- `run_miner.py` - Entry point for running miner
- `setup_artifacts.sh` - Setup script for artifacts directory

## Directory Structure

```
PatchTST + Student-t-miner/
├── __init__.py
├── config.py
├── data_fetcher.py
├── cache_manager.py
├── features.py
├── model.py
├── path_sampling.py
├── forecast_engine.py
├── fallback.py
├── miner.py
├── train.py
├── calibrate.py
├── cron_jobs.py
├── test_acceptance.py
├── run_miner.py
├── setup_artifacts.sh
├── README.md
└── STRUCTURE.md

artifacts/
├── current/
│   ├── hf_model.pt
│   ├── lf_crypto_model.pt
│   ├── lf_equity_model.pt
│   ├── normalization_stats.json
│   └── calibration.json
└── staging/
    └── (temporary files during updates)
```

## Key Components

### 1. ForecastEngine
Main orchestrator that:
- Routes requests to appropriate model (HF/LF-crypto/LF-equity)
- Manages cache refresh
- Fetches anchor prices
- Generates features
- Runs model inference
- Samples paths
- Formats output

### 2. Model Architecture
- PatchTST: Transformer with patch embeddings
- Asset embeddings for multi-asset modeling
- Student-t outputs (mu, log_sigma, nu)
- Three separate models for different regimes

### 3. Data Pipeline
- Benchmarks API: Historical price data
- Hermes API: Current anchor prices
- Rolling cache: Efficient data management
- Feature extraction: Comprehensive feature engineering

### 4. Calibration System
- Offline calibration using historical data
- Computes sigma_scale per model/asset
- EMA updates with move limits
- Atomic publish to avoid corruption

### 5. Fallback System
- Automatic fallback when model/data unavailable
- Uses realized volatility from cache
- Ensures valid output always returned

## Integration Points

### With Base Project
- Inherits from `BaseMinerNeuron`
- Uses `Simulation` protocol
- Compatible with existing validator infrastructure
- No modifications to base project files

### Running the Miner
1. Place model checkpoints in `artifacts/current/`
2. Run `python run_miner.py` or use PM2 config
3. Miner automatically handles routing and caching

### Updating from Upstream
- All custom code is in `PatchTST + Student-t-miner/` folder
- No conflicts with upstream updates
- Can safely pull and merge upstream changes

## Next Steps

1. **Train initial models**: Run `train.py` for each model type
2. **Initial calibration**: Run `calibrate.py` to establish baseline scales
3. **Set up cron jobs**: Configure automated calibration and training
4. **Run acceptance tests**: Verify everything works correctly
5. **Deploy miner**: Start miner and monitor performance
