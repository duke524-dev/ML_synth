# LGBM Miner Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install lightgbm scipy msgpack
```

## Step 2: Train Models

**Important:** You must train models before testing or running the miner!

```bash
cd LGBM-miner

# Train all assets (this will take a while - fetches 1 year of data for each)
python train_models.py

# Or train just one asset for testing
python train_models.py --asset BTC
```

Training progress will be shown in the console. Models are saved to the `models/` directory.

## Step 3: Test the Miner

```bash
python test_miner.py
```

This will:
- Check if models exist
- Generate test paths for BTC
- Verify the output format

## Step 4: Run the Miner

```bash
python miner.py
```

## Troubleshooting

### "No trained models found"
- Make sure you've run `train_models.py` first
- Check that models exist in the `models/` directory
- Verify the model files match the pattern: `{ASSET}_{HF|LF}_{LEAD}.pkl`

### Training fails with "No training data"
- Check your internet connection (needs to fetch from Pyth Benchmarks API)
- Verify the asset symbol is correct
- Try training a different asset (some may have limited historical data)

### Out of memory during training
- Train one asset at a time: `python train_models.py --asset BTC`
- Reduce the amount of training data (modify `TRAINING_YEARS` in config.py)

## Expected Training Time

- **Per asset:** 5-15 minutes (depends on data availability and system)
- **All assets:** 1-2 hours (9 assets total)

Training time includes:
- Fetching 1 year of historical data (~525,600 data points per asset)
- Feature engineering
- Training models for multiple leads (HF: 6 leads, LF: 4 leads)