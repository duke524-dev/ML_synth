"""
Rolling cache manager for price data
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict
import threading
import time

from data_fetcher import BenchmarksFetcher, parse_benchmarks_data
from config import (
    HF_CACHE_MIN_HOURS, LF_CACHE_MIN_DAYS,
    HF_LOOKBACK_HOURS, LF_LOOKBACK_DAYS,
    HF_REFRESH_INTERVAL_MIN, HF_REFRESH_INTERVAL_MAX,
    LF_REFRESH_INTERVAL_MIN, LF_REFRESH_INTERVAL_MAX,
)

logger = logging.getLogger(__name__)


class PriceCache:
    """Per-asset rolling cache for price data"""
    
    def __init__(self, asset: str, resolution: int, is_hf: bool):
        self.asset = asset
        self.resolution = resolution  # 1 or 5 minutes
        self.is_hf = is_hf
        self.fetcher = BenchmarksFetcher()
        
        # Cache storage: list of (timestamp, price) tuples, sorted by timestamp
        self.data: List[Tuple[int, float]] = []
        self.lock = threading.RLock()
        
        # Last refresh time
        self.last_refresh: Optional[datetime] = None
        
        # Refresh interval
        if is_hf:
            self.refresh_interval_min = HF_REFRESH_INTERVAL_MIN
            self.refresh_interval_max = HF_REFRESH_INTERVAL_MAX
        else:
            self.refresh_interval_min = LF_REFRESH_INTERVAL_MIN
            self.refresh_interval_max = LF_REFRESH_INTERVAL_MAX
    
    def _snap_to_grid(self, dt: datetime) -> datetime:
        """Snap datetime to resolution grid"""
        if self.resolution == 1:
            # Snap to minute
            return dt.replace(second=0, microsecond=0)
        else:  # 5 minutes
            # Snap to 5-minute boundary
            minute = (dt.minute // 5) * 5
            return dt.replace(minute=minute, second=0, microsecond=0)
    
    def _get_required_coverage(self) -> timedelta:
        """Get minimum required time coverage"""
        if self.is_hf:
            return timedelta(hours=HF_CACHE_MIN_HOURS)
        else:
            return timedelta(days=LF_CACHE_MIN_DAYS)
    
    def _get_lookback_window(self) -> timedelta:
        """Get lookback window for features"""
        if self.is_hf:
            return timedelta(hours=HF_LOOKBACK_HOURS)
        else:
            return timedelta(days=LF_LOOKBACK_DAYS)
    
    def _merge_data(self, new_data: List[Tuple[int, float]]):
        """Merge new data into cache, removing duplicates"""
        with self.lock:
            # Combine and deduplicate
            all_data = self.data + new_data
            # Sort by timestamp
            all_data.sort(key=lambda x: x[0])
            # Remove duplicates (keep last)
            seen = {}
            for ts, price in all_data:
                seen[ts] = price
            self.data = sorted(seen.items())
    
    def _fetch_backfill(self, end_time: datetime) -> bool:
        """Fetch backfill data to ensure coverage"""
        if self.is_hf:
            # Fetch last 30 hours
            start_time = end_time - timedelta(hours=30)
        else:
            # Fetch last 23 days
            start_time = end_time - timedelta(days=23)
        
        from_ts = int(start_time.timestamp())
        to_ts = int(end_time.timestamp())
        
        logger.info(f"Backfilling {self.asset} from {start_time} to {end_time}")
        data = self.fetcher.fetch_history(
            self.asset, self.resolution, from_ts, to_ts
        )
        
        if data is None:
            return False
        
        timestamps, closes = parse_benchmarks_data(data)
        if not timestamps:
            return False
        
        new_data = list(zip(timestamps, closes))
        self._merge_data(new_data)
        return True
    
    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed"""
        if self.last_refresh is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_refresh).total_seconds()
        
        if self.is_hf:
            # Randomize refresh interval to avoid thundering herd
            import random
            interval = random.uniform(
                self.refresh_interval_min, self.refresh_interval_max
            )
            return elapsed >= interval
        else:
            interval = self.refresh_interval_max
            return elapsed >= interval
    
    def refresh(self):
        """Refresh cache with recent data"""
        if not self._should_refresh():
            return
        
        now = datetime.now(timezone.utc)
        
        if self.is_hf:
            # Fetch last ~3 hours
            start_time = now - timedelta(hours=3)
        else:
            # Fetch last ~2 days
            start_time = now - timedelta(days=2)
        
        from_ts = int(start_time.timestamp())
        to_ts = int(now.timestamp())
        
        logger.debug(f"Refreshing cache for {self.asset}")
        data = self.fetcher.fetch_history(
            self.asset, self.resolution, from_ts, to_ts
        )
        
        if data is None:
            logger.warning(f"Failed to refresh cache for {self.asset}")
            return
        
        timestamps, closes = parse_benchmarks_data(data)
        if not timestamps:
            logger.warning(f"No data in refresh for {self.asset}")
            return
        
        new_data = list(zip(timestamps, closes))
        self._merge_data(new_data)
        self.last_refresh = now
        
        # Trim old data beyond required coverage
        self._trim_old_data()
    
    def _trim_old_data(self):
        """Remove data older than required coverage"""
        with self.lock:
            if not self.data:
                return
            
            now = datetime.now(timezone.utc)
            cutoff = now - self._get_required_coverage()
            cutoff_ts = int(cutoff.timestamp())
            
            self.data = [(ts, p) for ts, p in self.data if ts >= cutoff_ts]
    
    def get_features_window(
        self, start_time: datetime, required_points: int
    ) -> Optional[Tuple[List[int], List[float]]]:
        """
        Get data window for feature extraction
        
        Args:
            start_time: Start time (will be snapped to grid)
            required_points: Number of points needed
            
        Returns:
            (timestamps, prices) or None if insufficient data
        """
        with self.lock:
            # Ensure cache has coverage
            self._ensure_coverage(start_time)
            
            # Snap to grid
            snapped_start = self._snap_to_grid(start_time)
            start_ts = int(snapped_start.timestamp())
            
            # Calculate end timestamp
            resolution_seconds = self.resolution * 60
            end_ts = start_ts + (required_points - 1) * resolution_seconds
            
            # Extract window
            window = [
                (ts, price) for ts, price in self.data
                if start_ts <= ts <= end_ts
            ]
            
            if not window:
                return None
            
            # Check for missing data
            expected_timestamps = set(
                range(start_ts, end_ts + 1, resolution_seconds)
            )
            actual_timestamps = {ts for ts, _ in window}
            missing = len(expected_timestamps - actual_timestamps)
            missing_pct = missing / len(expected_timestamps) if expected_timestamps else 0
            
            threshold = 0.05 if self.is_hf else 0.02
            if missing_pct > threshold:
                logger.warning(
                    f"Too many missing points for {self.asset}: "
                    f"{missing_pct:.2%} missing"
                )
                return None
            
            # Sort and return
            window.sort(key=lambda x: x[0])
            timestamps, prices = zip(*window)
            return list(timestamps), list(prices)
    
    def _ensure_coverage(self, start_time: datetime):
        """Ensure cache has sufficient coverage"""
        if not self.data:
            # Initial backfill
            self._fetch_backfill(start_time)
            return
        
        # Check if we have enough coverage
        oldest_ts = self.data[0][0] if self.data else None
        if oldest_ts is None:
            self._fetch_backfill(start_time)
            return
        
        oldest_time = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
        required_coverage = self._get_required_coverage()
        
        if start_time - oldest_time > required_coverage:
            # Need more data
            self._fetch_backfill(start_time)


class CacheManager:
    """Manages caches for all assets"""
    
    def __init__(self):
        self.caches: Dict[str, Dict[int, PriceCache]] = {}
        self.lock = threading.Lock()
    
    def get_cache(self, asset: str, resolution: int, is_hf: bool) -> PriceCache:
        """Get or create cache for asset/resolution"""
        with self.lock:
            if asset not in self.caches:
                self.caches[asset] = {}
            
            if resolution not in self.caches[asset]:
                self.caches[asset][resolution] = PriceCache(
                    asset, resolution, is_hf
                )
            
            return self.caches[asset][resolution]
    
    def refresh_all(self):
        """Refresh all caches"""
        with self.lock:
            for asset_caches in self.caches.values():
                for cache in asset_caches.values():
                    try:
                        cache.refresh()
                    except Exception as e:
                        logger.error(f"Error refreshing cache: {e}")
