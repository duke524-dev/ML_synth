# EWMA Miner for Synth Subnet

CPU-first, simulation-based miner using EWMA volatility models with Student-t innovations.

## Architecture

The miner is split into two components:

1. **State Updater** (separate process): Continuously fetches 1-minute OHLC data, updates volatility states, and resamples to 5-minute bars.
2. **Miner** (main process): Serves validator requests using the latest volatility states.

## Setup

### Dependencies

Install required packages:
```bash
pip install bittensor numpy scipy msgpack requests tenacity
```

### Directory Structure

```
EWMA-miner/
├── __init__.py
├── config.py              # Configuration constants
├── data_fetcher.py        # Pyth Benchmarks & Hermes API clients
├── volatility_state.py    # EWMA volatility state management
├── data_manager.py        # OHLC storage and 5-minute resampling
├── persistence.py         # Atomic state persistence (msgpack)
├── simulation_engine.py   # Vectorized path simulation
├── state_updater.py       # Background data updater process
├── forecast_engine.py     # Main orchestrator
├── miner.py               # Bittensor miner class
├── test_wrapper.py        # Adapter for offline CRPS tests
└── README.md
```

## Usage

### 1. Start State Updater (Background Process)

The state updater must be running before starting the miner:

```bash
cd EWMA-miner
python state_updater.py --state-dir state --update-interval 60
```

Options:
- `--state-dir`: Directory for state persistence (default: "state")
- `--update-interval`: Update interval in seconds (default: 60)
- `--log-level`: Logging level (default: "INFO")

The updater will:
- Load existing states if available
- Warm up volatility states with historical data (72h for crypto, 5 days for equity)
- Continuously fetch and update 1-minute OHLC data
- Update volatility states with gap detection
- Persist states atomically

### 2. Start Miner

In a separate terminal:

```bash
cd EWMA-miner
python miner.py
```

The miner will:
- Load persisted states from the state directory
- Serve validator requests using the latest volatility states
- Generate 1000 simulated paths per request

### 3. Testing with Offline CRPS Test

To test with `offline_crps_simple.py`, you need to patch the import:

```python
# In offline_crps_simple.py or a test script:
import sys
sys.path.insert(0, 'EWMA-miner')

# Patch the import
import test_wrapper
import synth.miner.simulations as sim_module
sim_module.generate_simulations = test_wrapper.generate_simulations
sim_module.get_asset_price = test_wrapper.get_asset_price

# Now run offline_crps_simple.py normally
```

Or create a wrapper script:

```python
# test_ewma_miner.py
import sys
import os

# Add EWMA-miner to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'EWMA-miner'))

# Patch synth.miner.simulations
from test_wrapper import generate_simulations, get_asset_price
import synth.miner.simulations as sim_module
sim_module.generate_simulations = generate_simulations
sim_module.get_asset_price = get_asset_price

# Import and run the test
from offline_CRPS_test.offline_crps_simple import main
main()
```

### Using the Test Runner Script (Recommended)

The easiest way to test is using the provided test runner:

```bash
cd EWMA-miner

# Step 1: Warm up states for your test date
python state_updater.py --warmup-date 2026-01-15 --one-shot --state-dir state

# Step 2: Run the CRPS test
python run_crps_test.py --start-day 2026-01-15 --num-days 1
```

The test runner automatically patches `synth.miner.simulations` to use the EWMA miner.

## Configuration

Key parameters in `config.py`:

- **Half-lives**: `HALF_LIFE_1M` and `HALF_LIFE_5M` - EWMA decay rates
- **Student-t nu**: `NU` - Degrees of freedom per asset
- **Warmup periods**: `WARMUP_1M_HOURS` (72h) and `WARMUP_5M_DAYS` (5 days)
- **Equity off-hours multiplier**: `EQUITY_OFF_HOURS_MULT` (0.6)

## State Persistence

States are saved atomically per asset in `state/{asset}.msgpack`:
- Volatility states (1m and 5m)
- Recent OHLC data (7 days rolling window)
- 5-minute aggregation state

## Performance

- **Vectorized simulation**: No Python loops over 1000 paths
- **CPU-optimized**: Uses float32, pre-allocated arrays
- **Thread-safe**: Data managers use locks
- **Atomic persistence**: Temp file + rename for safety

## Design Notes

- **Gap detection**: Automatically detects missing bars and applies variance decay
- **5-minute resampling**: Aggregates 1-minute bars using OHLC rules
- **Student-t standardization**: Shocks are normalized to unit variance
- **Volatility clustering**: Simulated variance updates during path generation
