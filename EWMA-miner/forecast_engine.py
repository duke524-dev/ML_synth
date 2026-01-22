"""
Forecast engine: Main entry point for generating predictions
"""
import logging
import os
from datetime import datetime, timezone
from typing import Tuple, Optional

from data_fetcher import HermesFetcher
from data_manager import DataManager
from volatility_state import VolatilityState, OHLCBar
from simulation_engine import SimulationEngine
from persistence import StatePersistence
from config import FALLBACK_PRICES, TOKEN_MAP

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Main forecast engine orchestrating data, volatility, and simulation"""
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        
        self.hermes_fetcher = HermesFetcher()
        self.persistence = StatePersistence(state_dir)
        
        # Per-asset data managers and volatility states
        self.data_managers: dict = {}
        self.vol_states_1m: dict = {}
        self.vol_states_5m: dict = {}
        
        # Initialize for all assets
        for asset in TOKEN_MAP.keys():
            self.data_managers[asset] = DataManager(asset)
            self.vol_states_1m[asset] = VolatilityState(asset, resolution=1)
            self.vol_states_5m[asset] = VolatilityState(asset, resolution=5)
        
        # Load persisted states
        self._load_states()
    
    def _load_states(self):
        """Load persisted states for all assets"""
        for asset in TOKEN_MAP.keys():
            self.persistence.load_asset_state(
                asset=asset,
                vol_state_1m=self.vol_states_1m[asset],
                vol_state_5m=self.vol_states_5m[asset],
                data_manager=self.data_managers[asset]
            )

    def _get_anchor_price(self, asset: str) -> float:
        """
        Get anchor price with fallback chain:
        1. Try Hermes API (if feed ID exists)
        2. Try latest 1-minute candle close from data manager
        3. Use fallback price from config
        """
        # Try Hermes first
        hermes_price = self.hermes_fetcher.fetch_anchor_price(asset)
        if hermes_price is not None:
            return hermes_price
        
        # Fallback to latest 1-minute candle
        data_manager = self.data_managers.get(asset)
        if data_manager is not None:
            latest_close = data_manager.get_latest_close(resolution=1)
            if latest_close is not None:
                logger.info(f"Using latest 1m candle close for {asset}: {latest_close}")
                return latest_close
        
        # Last resort: use fallback price from config
        fallback = FALLBACK_PRICES.get(asset, 100.0)
        logger.warning(f"Using fallback price for {asset}: {fallback}")
        return fallback

    
    def generate_paths(
        self,
        asset: str,
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
    ) -> Tuple:
        """
        Generate price paths for a simulation request
        
        Returns:
            Tuple: (start_timestamp, time_increment, [path1, path2, ...])
        """
        try:
            # Get anchor price
            anchor_price = self._get_anchor_price(asset)
            logger.info(
                f"ForecastEngine.generate_paths({asset}): "
                f"anchor_price={anchor_price:.2f}, "
                f"start_time={start_time}, "
                f"time_increment={time_increment}s, time_length={time_length}s"
            )
            
            # Get volatility states
            vol_state_1m = self.vol_states_1m.get(asset)
            vol_state_5m = self.vol_states_5m.get(asset)
            
            # Log volatility state info
            if vol_state_1m:
                logger.info(
                    f"Volatility state 1m for {asset}: "
                    f"h={vol_state_1m.h:.8f}, "
                    f"volatility={vol_state_1m.get_volatility():.6f}, "
                    f"last_update_ts={vol_state_1m.last_update_ts}"
                )
            if vol_state_5m:
                logger.info(
                    f"Volatility state 5m for {asset}: "
                    f"h={vol_state_5m.h:.8f}, "
                    f"volatility={vol_state_5m.get_volatility():.6f}, "
                    f"last_update_ts={vol_state_5m.last_update_ts}"
                )
            
            if vol_state_1m is None or vol_state_5m is None:
                logger.error(f"Volatility states not initialized for {asset}")
                return self._fallback_paths(asset, start_time, time_increment, 
                                           time_length, num_simulations, anchor_price)
            
            # Create simulation engine
            sim_engine = SimulationEngine(vol_state_1m, vol_state_5m)
            
            # Generate paths
            result = sim_engine.generate_paths(
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                anchor_price=anchor_price,
                include_start=True,  # Configurable
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_paths: {e}", exc_info=True)
            anchor_price = self._get_anchor_price(asset)
            return self._fallback_paths(asset, start_time, time_increment,
                                       time_length, num_simulations, anchor_price)
    
    def _fallback_paths(self, asset: str, start_time: str, time_increment: int,
                       time_length: int, num_simulations: int, 
                       anchor_price: float) -> Tuple:
        """Generate simple fallback paths"""
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        
        H = time_length // time_increment
        start_ts = int(start_dt.timestamp())
        
        # Return constant price paths
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from synth.utils.helpers import round_to_8_significant_digits
        
        formatted_paths = []
        for _ in range(num_simulations):
            path = [round_to_8_significant_digits(anchor_price)] * (H + 1)
            formatted_paths.append(path)
        
        return (start_ts, time_increment, *formatted_paths)
    
    def update_states_from_prices(
        self,
        asset: str,
        prices: list,
        base_start: datetime,
        target_time: datetime,
        time_increment: int = 60
    ):
        """
        Update volatility states from price data up to target_time.
        Used during testing to simulate state updates.
        
        Args:
            asset: Asset symbol
            prices: List of close prices (1-minute resolution)
            base_start: Start datetime of the price series
            target_time: Update states up to this time
            time_increment: Time increment in seconds (default 60 for 1-minute)
        """
        if asset not in self.data_managers:
            logger.warning(f"Unknown asset: {asset}")
            return
        
        data_manager = self.data_managers[asset]
        vol_state_1m = self.vol_states_1m[asset]
        vol_state_5m = self.vol_states_5m[asset]
        
        # Get last update timestamp for 1-minute state
        last_update_1m = vol_state_1m.last_update_ts
        
        # Calculate target timestamp
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        target_ts = int(target_time.timestamp())
        
        # Calculate which prices to process
        base_ts = int(base_start.timestamp())
        
        # Find start index (first price after last_update_ts)
        if last_update_1m is not None:
            start_idx = max(0, (last_update_1m - base_ts) // time_increment + 1)
        else:
            start_idx = 0
        
        # Find end index (up to target_time)
        end_idx = min(len(prices), (target_ts - base_ts) // time_increment + 1)
        
        if start_idx >= end_idx:
            # No new data to process
            return
        
        logger.debug(
            f"Updating states for {asset}: processing prices [{start_idx}:{end_idx}] "
            f"(last_update={last_update_1m}, target={target_ts})"
        )
        
        # Get previous bar for gap detection
        prev_bar_1m = data_manager.get_latest_bar(resolution=1)
        
        # Process 1-minute bars
        for i in range(start_idx, end_idx):
            if i >= len(prices):
                break
                
            price = prices[i]
            # Skip NaN or invalid prices
            if price is None:
                continue
            try:
                price = float(price)
                if price != price or price <= 0:  # NaN or invalid
                    continue
            except (ValueError, TypeError):
                continue
            
            timestamp = base_ts + i * time_increment
            
            # Skip if beyond target time
            if timestamp > target_ts:
                break
            
            # Create OHLC bar from close price
            # For testing: use close as all OHLC (or estimate from previous)
            if prev_bar_1m is not None:
                prev_close = prev_bar_1m.close
                # Estimate OHLC: use previous close as open, current as close
                # High/low are estimated as max/min of open and close
                bar_1m = OHLCBar(
                    timestamp=timestamp,
                    open=prev_close,
                    high=max(prev_close, price),
                    low=min(prev_close, price),
                    close=price
                )
            else:
                # First bar: use price for all OHLC
                bar_1m = OHLCBar(
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price
                )
            
            # Add to data manager and update volatility
            data_manager.add_1m_bar(bar_1m)
            vol_state_1m.update_with_bar(bar_1m, prev_bar_1m)
            prev_bar_1m = bar_1m
        
        # Update 5-minute state with new 5-minute bars
        latest_5m = vol_state_5m.last_update_ts
        if latest_5m is not None:
            bars_5m = data_manager.get_5m_bars(start_ts=latest_5m + 1)
        else:
            bars_5m = data_manager.get_5m_bars()
        
        # Filter bars up to target time
        bars_5m = [b for b in bars_5m if b.timestamp <= target_ts]
        
        if bars_5m:
            prev_bar_5m = data_manager.get_prev_bar(resolution=5)
            for bar_5m in bars_5m:
                vol_state_5m.update_with_bar(bar_5m, prev_bar_5m)
                prev_bar_5m = bar_5m
        
        logger.debug(
            f"Updated states for {asset}: processed {end_idx - start_idx} 1m bars, "
            f"{len(bars_5m)} 5m bars up to {target_time}"
        )
