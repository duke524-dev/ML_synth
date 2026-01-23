"""
Training script for LGBM models
Trains models for all assets with 1 year of historical data
"""
import sys
import os
import logging
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_fetcher import BenchmarksFetcher, parse_benchmarks_ohlc
from lgbm_trainer import LGBMTrainer
from config import TOKEN_MAP, HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS, TRAINING_YEARS
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_asset_with_data(
    asset: str, 
    is_hf: bool, 
    model_dir: str,
    timestamps: List[int],
    closes: List[float],
    opens: List[float],
    highs: List[float],
    lows: List[float],
    end_time: datetime
) -> bool:
    """
    Train LGBM model for a single asset using provided data
    
    Args:
        asset: Asset symbol
        is_hf: True for high frequency, False for low frequency
        model_dir: Directory to save models
        timestamps: List of Unix timestamps
        closes: List of close prices
        opens: List of open prices
        highs: List of high prices
        lows: List of low prices
        end_time: Reference time for recency weighting
    
    Returns:
        True if training successful, False otherwise
    """
    try:
        # Initialize trainer
        trainer = LGBMTrainer(asset, is_hf, model_dir)
        
        # Train models
        trainer.train(timestamps, closes, opens, highs, lows, end_time)
        
        # Save models
        trainer.save_models()
        
        logger.info(f"✓ Successfully trained {asset} ({'HF' if is_hf else 'LF'})")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error training {asset} ({'HF' if is_hf else 'LF'}): {e}", exc_info=True)
        return False


def train_asset(asset: str, is_hf: bool, model_dir: str = "models", 
                test_mode: bool = False, chunk_days: int = 30, use_cache: bool = True):
    """
    Train LGBM model for a single asset
    
    Args:
        asset: Asset symbol
        is_hf: True for high frequency, False for low frequency
        model_dir: Directory to save models
        test_mode: If True, use 30 days instead of 1 year
        chunk_days: Number of days per API request chunk
    """
    logger.info(f"Training {asset} ({'HF' if is_hf else 'LF'})...")
    
    try:
        # Initialize trainer
        trainer = LGBMTrainer(asset, is_hf, model_dir)
        
        # Fetch training data
        fetcher = BenchmarksFetcher()
        end_time = datetime.now(timezone.utc)
        
        if test_mode:
            logger.info(f"TEST MODE: Fetching 30 days of training data for {asset}...")
            data = fetcher.fetch_training_data(
                asset, resolution=1, end_time=end_time, 
                chunk_days=chunk_days, training_days=30, use_cache=use_cache
            )
        else:
            logger.info(f"Fetching {TRAINING_YEARS} year(s) of training data for {asset}...")
            data = fetcher.fetch_training_data(
                asset, resolution=1, end_time=end_time, 
                chunk_days=chunk_days, use_cache=use_cache
            )
        
        if data is None:
            logger.warning(f"No training data available for {asset}")
            return False
        
        timestamps, opens, highs, lows, closes = parse_benchmarks_ohlc(data)
        
        if not timestamps:
            logger.warning(f"No training data available for {asset}")
            return False
        
        if len(timestamps) < 100:
            logger.warning(
                f"Insufficient training data for {asset}: "
                f"{len(timestamps)} points (need at least 100). "
                f"Training may not be optimal."
            )
            # Still allow training with less data, but warn
        
        logger.info(
            f"Fetched {len(timestamps)} data points for {asset} "
            f"({timestamps[0]} to {timestamps[-1]})"
        )
        
        # Train models
        trainer.train(timestamps, closes, opens, highs, lows, end_time)
        
        # Save models
        trainer.save_models()
        
        logger.info(f"✓ Successfully trained {asset} ({'HF' if is_hf else 'LF'})")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error training {asset}: {e}", exc_info=True)
        return False


def main():
    """Train models for all assets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LGBM models")
    parser.add_argument(
        "--asset",
        type=str,
        nargs="+",  # Accept one or more assets
        default=None,
        help="Train specific asset(s) only (e.g., --asset BTC ETH or --asset SOL) (default: all assets)"
    )
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="Train only high-frequency assets"
    )
    parser.add_argument(
        "--lf-only",
        action="store_true",
        help="Train only low-frequency assets"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: use 30 days of data instead of 1 year (faster for testing)"
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=3,
        help="Number of days per API request chunk (default: 3, similar to EWMA warmup window)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache and fetch all data (ignore existing cached ranges)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Determine which assets to train
    if args.asset:
        # args.asset is now a list (can be one or more assets)
        assets_to_train = args.asset
        # Validate that all requested assets exist
        invalid_assets = [a for a in assets_to_train if a not in TOKEN_MAP]
        if invalid_assets:
            logger.error(f"Invalid asset(s): {', '.join(invalid_assets)}")
            logger.error(f"Valid assets: {', '.join(sorted(TOKEN_MAP.keys()))}")
            return False
    else:
        assets_to_train = list(TOKEN_MAP.keys())
    
    # Filter by frequency if requested
    if args.hf_only:
        assets_to_train = [a for a in assets_to_train if a in HF_ASSETS]
    elif args.lf_only:
        assets_to_train = [a for a in assets_to_train 
                          if a in LF_CRYPTO_ASSETS or a in LF_EQUITY_ASSETS]
    
    logger.info(f"Training models for {len(assets_to_train)} asset(s)")
    
    # Train each asset
    results = {}
    for asset in assets_to_train:
        use_cache = not args.no_cache
        
        # Determine which models to train based on flags
        if args.hf_only:
            needs_hf = asset in HF_ASSETS
            needs_lf = False
        elif args.lf_only:
            needs_hf = False
            needs_lf = asset in LF_CRYPTO_ASSETS or asset in LF_EQUITY_ASSETS
        else:
            # Train both if asset is in both groups
            needs_hf = asset in HF_ASSETS
            needs_lf = asset in LF_CRYPTO_ASSETS or asset in LF_EQUITY_ASSETS
        
        # Fetch data once (shared for both HF and LF if needed)
        if needs_hf or needs_lf:
            # Fetch data once - same 1-minute data is used for both
            fetcher = BenchmarksFetcher()
            end_time = datetime.now(timezone.utc)
            
            if args.test_mode:
                logger.info(f"TEST MODE: Fetching 30 days of training data for {asset}...")
                data = fetcher.fetch_training_data(
                    asset, resolution=1, end_time=end_time, 
                    chunk_days=args.chunk_days, training_days=30, use_cache=use_cache
                )
            else:
                logger.info(f"Fetching {TRAINING_YEARS} year(s) of training data for {asset}...")
                data = fetcher.fetch_training_data(
                    asset, resolution=1, end_time=end_time, 
                    chunk_days=args.chunk_days, use_cache=use_cache
                )
            
            if data is None:
                logger.warning(f"No training data available for {asset}")
                results[asset] = False
                continue
            
            timestamps, opens, highs, lows, closes = parse_benchmarks_ohlc(data)
            
            if not timestamps or len(timestamps) < 100:
                logger.warning(
                    f"Insufficient training data for {asset}: "
                    f"{len(timestamps)} points (need at least 100)"
                )
                results[asset] = False
                continue
            
            logger.info(
                f"Fetched {len(timestamps)} data points for {asset} "
                f"({timestamps[0]} to {timestamps[-1]})"
            )
            
            # Train HF model if needed
            hf_success = True
            if needs_hf:
                logger.info(f"Training {asset} HF model...")
                hf_success = train_asset_with_data(
                    asset, is_hf=True, model_dir=args.model_dir,
                    timestamps=timestamps, closes=closes, opens=opens,
                    highs=highs, lows=lows, end_time=end_time
                )
            
            # Train LF model if needed
            lf_success = True
            if needs_lf:
                logger.info(f"Training {asset} LF model...")
                lf_success = train_asset_with_data(
                    asset, is_hf=False, model_dir=args.model_dir,
                    timestamps=timestamps, closes=closes, opens=opens,
                    highs=highs, lows=lows, end_time=end_time
                )
            
            # Asset is successful if all needed models trained successfully
            results[asset] = hf_success and lf_success
        else:
            logger.warning(f"Asset {asset} not in HF or LF groups, skipping")
            results[asset] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    
    successful = [asset for asset, success in results.items() if success]
    failed = [asset for asset, success in results.items() if not success]
    
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        logger.info(f"  {', '.join(successful)}")
    
    if failed:
        logger.warning(f"Failed: {len(failed)}/{len(results)}")
        logger.warning(f"  {', '.join(failed)}")
    
    logger.info("="*60)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)