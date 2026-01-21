# PatchTST + Student-t Miner for Synth Subnet

A sophisticated Bittensor Synth subnet miner implementation using PatchTST architecture with Student-t distribution for probabilistic price forecasting.

## Overview

This miner implements three separate models:
- **HF-Shared**: High-frequency model (1-minute resolution) for BTC/ETH/SOL/XAU
- **LF-CryptoGold**: Low-frequency model (5-minute resolution) for BTC/ETH/SOL/XAU
- **LF-EquitiesTokens**: Low-frequency model (5-minute resolution) for SPYX/NVDAX/TSLAX/AAPLX/GOOGLX

The miner automatically routes requests to the appropriate model based on prompt parameters and asset type.

## Features

- **PatchTST Architecture**: Transformer-based time series forecasting with patch embeddings
- **Student-t Distribution**: Captures fat-tailed price movements and uncertainty
- **Latent Volatility Clustering**: AR(1) process for realistic volatility dynamics
- **Efficient Caching**: Rolling cache system for Benchmarks API data
- **Feature Engineering**: Comprehensive feature extraction (rolling stats, time features, session features)
- **Calibration System**: Offline calibration scripts for sigma_scale adjustment
- **Fallback Generator**: Robust fallback when model/data unavailable
- **CPU-Optimized**: Runs efficiently on CPU using PyTorch

## Installation

1. Install dependencies:
```bash
pip install torch numpy scipy requests tenacity pytz
```

2. Create artifacts directory structure:
```bash
mkdir -p artifacts/current artifacts/staging
```

3. Place model checkpoints in `artifacts/current/`:
- `hf_model.pt`
- `lf_crypto_model.pt`
- `lf_equity_model.pt`

4. Place normalization stats in `artifacts/current/normalization_stats.json`

5. Place calibration scales in `artifacts/current/calibration.json`

## Usage

### Running the Miner

The miner integrates with the base Bittensor miner infrastructure. To use it:

1. Update your miner config to use the custom miner class:

```python
# In your miner entrypoint
from "PatchTST + Student-t-miner.miner" import DukeMiner1

miner = DukeMiner1(config=config)
```

2. Or create a new PM2 config file:

```js
// duke_miner.config.js
module.exports = {
  apps: [
    {
      name: "duke-miner",
      interpreter: "python3",
      script: "./PatchTST + Student-t-miner/miner.py",
      args: "--netuid 50 --logging.info --wallet.name miner --wallet.hotkey default --axon.port 8092",
      env: {
        PYTHONPATH: ".",
      },
    },
  ],
};
```

3. Start with PM2:
```bash
pm2 start duke_miner.config.js
```

### Training Models

Train models offline using historical data:

```bash
# Train HF model
python -m "PatchTST + Student-t-miner.train" --model HF --epochs 50

# Train LF-crypto model
python -m "PatchTST + Student-t-miner.train" --model LF-crypto --epochs 50

# Train LF-equity model
python -m "PatchTST + Student-t-miner.train" --model LF-equity --epochs 50
```

### Calibration

Calibrate sigma scales to match realized volatility:

```bash
# Calibrate HF model (every 3 hours)
python -m "PatchTST + Student-t-miner.calibrate" --model HF

# Calibrate LF models (daily)
python -m "PatchTST + Student-t-miner.calibrate" --model LF-crypto
python -m "PatchTST + Student-t-miner.calibrate" --model LF-equity
```

### Running Tests

Run acceptance tests:

```bash
python -m "PatchTST + Student-t-miner.test_acceptance"
```

## Cron Automation

Set up cron jobs for automated calibration and training:

```bash
# HF calibration every 3 hours
0 */3 * * * cd /path/to/project && python -m "PatchTST + Student-t-miner.cron_jobs" --job calibrate-hf --restart-miner

# LF calibration daily at 2 AM
0 2 * * * cd /path/to/project && python -m "PatchTST + Student-t-miner.cron_jobs" --job calibrate-lf --restart-miner

# Training weekly on Sunday at 3 AM
0 3 * * 0 cd /path/to/project && python -m "PatchTST + Student-t-miner.cron_jobs" --job train
```

Or use the convenience script:

```bash
python -m "PatchTST + Student-t-miner.cron_jobs" --job calibrate-hf --artifacts-dir artifacts --restart-miner
```

## Architecture

### Model Routing

- `time_increment == 60 and time_length == 3600` → HF-Shared
- Asset in {BTC, ETH, SOL, XAU} → LF-CryptoGold
- Asset in {SPYX, NVDAX, TSLAX, AAPLX, GOOGLX} → LF-EquitiesTokens

### Data Flow

1. **Request received** → Parse `SimulationInput`
2. **Route to model** → Determine HF/LF-crypto/LF-equity
3. **Fetch anchor price** → Hermes API (with fallback)
4. **Get features** → From cache (refresh if needed)
5. **Model forward pass** → Generate mu, log_sigma, nu
6. **Apply calibration** → Scale sigma
7. **Sample paths** → Student-t with latent volatility
8. **Format output** → Tuple format for validator

### Caching Strategy

- **HF cache**: 1-minute bars, 30+ hours coverage, refresh every 2-5 minutes
- **LF cache**: 5-minute bars, 23+ days coverage, refresh every 30-60 minutes
- Automatic backfill when insufficient data

## Output Format

The miner returns predictions in the format expected by validators:

```python
(
    start_timestamp,      # int: Unix timestamp
    time_increment,       # int: Seconds
    [path1_prices],       # list: Prices for path 1
    [path2_prices],       # list: Prices for path 2
    ...                   # num_simulations paths
)
```

Each path contains `(time_length // time_increment) + 1` prices, including the anchor price at index 0.

## Configuration

Key configuration constants are in `config.py`:

- API endpoints (Benchmarks, Hermes)
- Token mappings
- Cache refresh intervals
- Calibration parameters
- Model hyperparameters

## Troubleshooting

### Model not loading
- Check that model checkpoints exist in `artifacts/current/`
- Verify model architecture matches checkpoint
- Check logs for specific error messages

### Insufficient data
- Miner will use fallback generator automatically
- Check cache refresh is working
- Verify Benchmarks API is accessible

### Calibration not updating
- Check minimum sample counts are met
- Verify calibration window has sufficient data
- Check logs for specific errors

## License

MIT License - see LICENSE file

## Contributing

This is a custom miner implementation. When pulling updates from the upstream synth-subnet repository, your custom miner files will remain untouched.
