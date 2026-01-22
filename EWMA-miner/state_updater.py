"""
Background state updater process - runs separately to update data and volatility states
"""
import logging
import time
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict
import argparse

from data_fetcher import BenchmarksFetcher, parse_benchmarks_ohlc
from data_manager import DataManager
from volatility_state import VolatilityState, OHLCBar
from persistence import StatePersistence
from config import (
    TOKEN_MAP, RETENTION_DAYS, WARMUP_1M_HOURS, WARMUP_5M_DAYS,
    HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS
)

logger = logging.getLogger(__name__)


class StateUpdater:
    """Background process that continuously updates data and volatility states"""
    
    def __init__(self, state_dir: str = "state", update_interval: int = 60):
        self.state_dir = state_dir
        self.update_interval = update_interval  # seconds
        
        self.fetcher = BenchmarksFetcher()
        self.persistence = StatePersistence(state_dir)
        
        # Per-asset components
        self.data_managers: Dict[str, DataManager] = {}
        self.vol_states_1m: Dict[str, VolatilityState] = {}
        self.vol_states_5m: Dict[str, VolatilityState] = {}
        
        self.running = False
        
        # Initialize all assets
        for asset in TOKEN_MAP.keys():
            self.data_managers[asset] = DataManager(asset)
            self.vol_states_1m[asset] = VolatilityState(asset, resolution=1)
            self.vol_states_5m[asset] = VolatilityState(asset, resolution=5)
    
    def load_states(self):
        """Load persisted states for all assets"""
        for asset in TOKEN_MAP.keys():
            self.persistence.load_asset_state(
                asset=asset,
                vol_state_1m=self.vol_states_1m[asset],
                vol_state_5m=self.vol_states_5m[asset],
                data_manager=self.data_managers[asset]
            )
    
    def save_states(self):
        """Save states for all assets"""
        assets_state = {}
        for asset in TOKEN_MAP.keys():
            assets_state[asset] = {
                'vol_state_1m': self.vol_states_1m[asset],
                'vol_state_5m': self.vol_states_5m[asset],
                'data_manager': self.data_managers[asset],
            }
        self.persistence.save_all_states(assets_state)
    
    def warmup_states(self, target_date: datetime = None):
        """Warm up volatility states with historical data
        
        Args:
            target_date: Target date for warmup (default: current time).
                        States will be warmed up as of this date.
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc)
        
        logger.info(f"Warming up volatility states as of {target_date}...")
        
        for asset in TOKEN_MAP.keys():
            try:
                # Determine warmup period
                if asset in HF_ASSETS or asset in LF_CRYPTO_ASSETS:
                    warmup_hours = WARMUP_1M_HOURS
                else:
                    warmup_hours = WARMUP_5M_DAYS * 24
                
                # Fetch historical 1-minute data
                from_ts = int((target_date - timedelta(hours=warmup_hours)).timestamp())
                to_ts = int(target_date.timestamp())
                
                logger.info(f"Fetching warmup data for {asset} ({warmup_hours}h)")
                data = self.fetcher.fetch_history(asset, resolution=1, from_time=from_ts, to_time=to_ts)
                
                if data is None:
                    logger.warning(f"No warmup data for {asset}")
                    continue
                
                timestamps, opens, highs, lows, closes = parse_benchmarks_ohlc(data)
                
                if not timestamps:
                    logger.warning(f"Empty warmup data for {asset}")
                    continue
                
                # Process 1-minute bars
                prev_bar_1m = None
                for i in range(len(timestamps)):
                    bar_1m = OHLCBar(
                        timestamp=timestamps[i],
                        open=opens[i],
                        high=highs[i],
                        low=lows[i],
                        close=closes[i]
                    )
                    
                    self.data_managers[asset].add_1m_bar(bar_1m)
                    self.vol_states_1m[asset].update_with_bar(bar_1m, prev_bar_1m)
                    prev_bar_1m = bar_1m
                
                # Process 5-minute bars for 5-minute state
                bars_5m = self.data_managers[asset].get_5m_bars()
                prev_bar_5m = None
                for bar_5m in bars_5m:
                    self.vol_states_5m[asset].update_with_bar(bar_5m, prev_bar_5m)
                    prev_bar_5m = bar_5m
                
                logger.info(f"Warmed up {asset}: {len(timestamps)} 1m bars, {len(bars_5m)} 5m bars")
                
            except Exception as e:
                logger.error(f"Error warming up {asset}: {e}", exc_info=True)
    
    def update_asset(self, asset: str):
        """Update data and volatility for one asset"""
        try:
            # Fetch recent 1-minute data (last 2 hours)
            now = datetime.now(timezone.utc)
            from_ts = int((now - timedelta(hours=2)).timestamp())
            to_ts = int(now.timestamp())
            
            data = self.fetcher.fetch_history(asset, resolution=1, from_time=from_ts, to_time=to_ts)
            
            if data is None:
                return
            
            timestamps, opens, highs, lows, closes = parse_benchmarks_ohlc(data)
            
            if not timestamps:
                return
            
            # Get latest existing bar to detect gaps
            latest_bar = self.data_managers[asset].get_latest_bar(resolution=1)
            
            # Process new bars
            prev_bar_1m = latest_bar
            for i in range(len(timestamps)):
                bar_1m = OHLCBar(
                    timestamp=timestamps[i],
                    open=opens[i],
                    high=highs[i],
                    low=lows[i],
                    close=closes[i]
                )
                
                # Skip if already have this bar, but update prev_bar for gap detection
                if latest_bar is not None and bar_1m.timestamp <= latest_bar.timestamp:
                    prev_bar_1m = bar_1m  # Update even when skipping for correct gap detection
                    continue
                
                self.data_managers[asset].add_1m_bar(bar_1m)
                self.vol_states_1m[asset].update_with_bar(bar_1m, prev_bar_1m)
                prev_bar_1m = bar_1m
            
            # Update 5-minute state with new 5-minute bars only
            latest_5m = self.vol_states_5m[asset].last_update_ts
            if latest_5m is not None:
                # Only get bars after the last update
                bars_5m = self.data_managers[asset].get_5m_bars(start_ts=latest_5m + 1)
            else:
                # First time: get all bars
                bars_5m = self.data_managers[asset].get_5m_bars()
            
            if bars_5m:
                prev_bar_5m = self.data_managers[asset].get_prev_bar(resolution=5)
                for bar_5m in bars_5m:
                    self.vol_states_5m[asset].update_with_bar(bar_5m, prev_bar_5m)
                    prev_bar_5m = bar_5m
            
        except Exception as e:
            logger.error(f"Error updating {asset}: {e}", exc_info=True)
    
    def update_all(self):
        """Update all assets"""
        for asset in TOKEN_MAP.keys():
            self.update_asset(asset)
    
    def run(self, target_date: datetime = None, one_shot: bool = False):
        """Main update loop
        
        Args:
            target_date: Target date for warmup (default: None, uses current time)
            one_shot: If True, warm up and exit without continuous updates
        """
        logger.info("Starting state updater...")
        
        # Load existing states
        self.load_states()
        
        # Warm up if states are empty or target_date is specified
        needs_warmup = all(len(dm.data_1m) == 0 for dm in self.data_managers.values())
        if needs_warmup or target_date is not None:
            self.warmup_states(target_date=target_date)
            self.save_states()
        
        # Exit if one-shot mode
        if one_shot:
            logger.info("One-shot mode: states warmed up and saved. Exiting.")
            return
        
        self.running = True
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal, saving states...")
            self.save_states()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info(f"State updater running (update interval: {self.update_interval}s)")
        
        while self.running:
            try:
                self.update_all()
                self.save_states()
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}", exc_info=True)
                time.sleep(self.update_interval)
        
        # Final save
        self.save_states()
        logger.info("State updater stopped")


def main():
    """Main entry point for state updater process"""
    parser = argparse.ArgumentParser(description="EWMA Miner State Updater")
    parser.add_argument("--state-dir", type=str, default="state", help="State directory")
    parser.add_argument("--update-interval", type=int, default=60, help="Update interval in seconds")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--warmup-date", type=str, default=None, 
                        help="Target date for warmup (YYYY-MM-DD). If specified, states will be warmed up as of this date.")
    parser.add_argument("--one-shot", action="store_true", 
                        help="Warm up states and exit (no continuous updates)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse warmup date if provided
    target_date = None
    if args.warmup_date:
        try:
            # Parse YYYY-MM-DD format
            target_date = datetime.strptime(args.warmup_date, "%Y-%m-%d")
            # Set to end of day UTC for that date
            target_date = target_date.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            logger.info(f"Target warmup date: {target_date}")
        except ValueError:
            logger.error(f"Invalid date format: {args.warmup_date}. Expected YYYY-MM-DD")
            sys.exit(1)
    
    updater = StateUpdater(state_dir=args.state_dir, update_interval=args.update_interval)
    updater.run(target_date=target_date, one_shot=args.one_shot)


if __name__ == "__main__":
    main()
