"""
Data fetching from Pyth Benchmarks and Hermes APIs with OHLC support
"""
import logging
import time
import os
import json
import msgpack
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Set
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import (
    BASE_URL, HERMES_BASE_URL, TOKEN_MAP, HERMES_FEED_IDS,
    FALLBACK_PRICES
)

# Default training years (can be overridden)
TRAINING_YEARS = 1

logger = logging.getLogger(__name__)


class DataCache:
    """Manages cache of fetched data ranges to avoid re-fetching"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_file(self, asset: str, resolution: int) -> str:
        """Get cache file path for an asset/resolution"""
        return os.path.join(self.cache_dir, f"{asset}_r{resolution}_ranges.json")
    
    def get_data_file(self, asset: str, resolution: int) -> str:
        """Get data file path for an asset/resolution"""
        return os.path.join(self.cache_dir, f"{asset}_r{resolution}_data.msgpack")
    
    def load_ranges(self, asset: str, resolution: int) -> Dict:
        """Load cached date ranges for an asset"""
        cache_file = self.get_cache_file(asset, resolution)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache for {asset}: {e}")
        return {"ranges": [], "last_updated": None}
    
    def save_ranges(self, asset: str, resolution: int, ranges: List[Tuple[int, int]], 
                   last_updated: Optional[datetime] = None):
        """Save cached date ranges for an asset"""
        cache_file = self.get_cache_file(asset, resolution)
        data = {
            "ranges": ranges,
            "last_updated": last_updated.isoformat() if last_updated else None
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache for {asset}: {e}")
    
    def get_missing_ranges(self, asset: str, resolution: int, 
                          from_time: datetime, to_time: datetime,
                          chunk_days: int) -> List[Tuple[datetime, datetime]]:
        """
        Determine which date ranges need to be fetched
        
        Returns:
            List of (start, end) datetime tuples for missing ranges
        """
        cached = self.load_ranges(asset, resolution)
        cached_ranges = set()
        for range_item in cached.get("ranges", []):
            # Convert list to tuple (lists are not hashable for sets)
            if isinstance(range_item, list) and len(range_item) == 2:
                start_ts, end_ts = tuple(range_item)
                cached_ranges.add((start_ts, end_ts))
            elif isinstance(range_item, tuple) and len(range_item) == 2:
                cached_ranges.add(range_item)
        
        # Generate all required chunks
        required_ranges = []
        current = from_time
        while current < to_time:
            chunk_end = min(current + timedelta(days=chunk_days), to_time)
            start_ts = int(current.timestamp())
            end_ts = int(chunk_end.timestamp())
            required_ranges.append((current, chunk_end, start_ts, end_ts))
            current = chunk_end
        
        # Find missing ranges
        missing = []
        for start_dt, end_dt, start_ts, end_ts in required_ranges:
            if (start_ts, end_ts) not in cached_ranges:
                missing.append((start_dt, end_dt))
        
        return missing
    
    def update_cache(self, asset: str, resolution: int, 
                    fetched_ranges: List[Tuple[datetime, datetime]]):
        """Update cache with newly fetched ranges"""
        cached = self.load_ranges(asset, resolution)
        existing_ranges = set()
        
        # Load existing ranges (convert lists to tuples)
        for range_item in cached.get("ranges", []):
            if isinstance(range_item, list) and len(range_item) == 2:
                existing_ranges.add(tuple(range_item))
            elif isinstance(range_item, tuple) and len(range_item) == 2:
                existing_ranges.add(range_item)
        
        # Add new ranges
        for start_dt, end_dt in fetched_ranges:
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            existing_ranges.add((start_ts, end_ts))
        
        # Convert back to list of lists (for JSON serialization) and sort
        ranges_list = sorted([list(r) for r in existing_ranges])
        self.save_ranges(asset, resolution, ranges_list, datetime.now(timezone.utc))
    
    def save_data(self, asset: str, resolution: int, data: Dict):
        """Save fetched OHLC data to disk"""
        data_file = self.get_data_file(asset, resolution)
        try:
            with open(data_file, 'wb') as f:
                msgpack.pack(data, f)
            logger.debug(f"Saved {len(data.get('t', []))} data points for {asset} to {data_file}")
        except Exception as e:
            logger.warning(f"Error saving data for {asset}: {e}")
    
    def load_data(self, asset: str, resolution: int) -> Optional[Dict]:
        """Load saved OHLC data from disk"""
        data_file = self.get_data_file(asset, resolution)
        if not os.path.exists(data_file):
            return None
        try:
            with open(data_file, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
            logger.info(f"Loaded {len(data.get('t', []))} data points for {asset} from cache")
            return data
        except Exception as e:
            logger.warning(f"Error loading data for {asset}: {e}")
            return None
    
    def merge_data(self, existing: Dict, new: Dict) -> Dict:
        """Merge new data with existing data, removing duplicates"""
        if not existing:
            return new
        
        # Combine all data
        all_timestamps = list(existing.get('t', [])) + list(new.get('t', []))
        all_opens = list(existing.get('o', [])) + list(new.get('o', []))
        all_highs = list(existing.get('h', [])) + list(new.get('h', []))
        all_lows = list(existing.get('l', [])) + list(new.get('l', []))
        all_closes = list(existing.get('c', [])) + list(new.get('c', []))
        
        # Remove duplicates by timestamp (keep most recent)
        data_dict = {}
        for i, ts in enumerate(all_timestamps):
            if ts not in data_dict or ts > data_dict[ts]['timestamp']:
                data_dict[ts] = {
                    'timestamp': ts,
                    'open': all_opens[i] if i < len(all_opens) else all_closes[i],
                    'high': all_highs[i] if i < len(all_highs) else all_closes[i],
                    'low': all_lows[i] if i < len(all_lows) else all_closes[i],
                    'close': all_closes[i] if i < len(all_closes) else all_opens[i],
                }
        
        # Sort by timestamp
        sorted_data = sorted(data_dict.values(), key=lambda x: x['timestamp'])
        
        return {
            's': 'ok',
            't': [d['timestamp'] for d in sorted_data],
            'o': [d['open'] for d in sorted_data],
            'h': [d['high'] for d in sorted_data],
            'l': [d['low'] for d in sorted_data],
            'c': [d['close'] for d in sorted_data],
        }


class BenchmarksFetcher:
    """Fetches historical OHLC data from Pyth Benchmarks TradingView API"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache = DataCache(cache_dir)
    
    @retry(
        stop=stop_after_attempt(2),  # Reduced retries for faster failure on network issues
        wait=wait_random_exponential(multiplier=1, min=0.5, max=2),
        reraise=True
    )
    def fetch_history(
        self,
        asset: str,
        resolution: int,  # 1 or 5 (minutes)
        from_time: int,  # Unix timestamp
        to_time: int,  # Unix timestamp
    ) -> Optional[Dict]:
        """
        Fetch historical OHLC data from Benchmarks API
        
        Returns:
            Dict with keys: 's' (status), 't' (timestamps), 'o', 'h', 'l', 'c' (OHLC)
            or None if error
        """
        if asset not in TOKEN_MAP:
            logger.error(f"Unknown asset: {asset}")
            return None
            
        symbol = TOKEN_MAP[asset]
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_time,
            "to": to_time,
        }
        
        try:
            logger.debug(
                f"Making API request for {asset}: "
                f"symbol={params['symbol']}, resolution={params['resolution']}, "
                f"from={params['from']}, to={params['to']}"
            )
            response = requests.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"API response for {asset}: status={data.get('s')}, data_points={len(data.get('t', []))}")

            status = data.get("s")
            if status != "ok":
                error_msg = data.get("errmsg", "No error message")
                logger.warning(
                    f"Benchmarks status for {asset}: {status}. "
                    f"Error: {error_msg}. Params: {params}"
                )
                # Still try to extract data even if status is not "ok"
                # Sometimes the API returns data with status "error" for partial results

            timestamps = data.get("t") or []
            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])

            if not timestamps or not closes:
                logger.warning(
                    f"No Benchmarks data points for {asset} with params={params}. "
                    f"Status: {status}, Error: {data.get('errmsg', 'N/A')}"
                )
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Benchmarks data: {e}")
            return None
    
    def fetch_training_data(
        self,
        asset: str,
        resolution: int = 1,
        end_time: Optional[datetime] = None,
        chunk_days: int = 3,  # Fetch in 3-day chunks (EWMA warmup uses ~72h)
        training_days: Optional[int] = None,  # Override default training period
        use_cache: bool = True  # Check cache and only fetch missing ranges
    ) -> Optional[Dict]:
        """
        Fetch training data in chunks, checking cache to avoid re-fetching
        
        Args:
            asset: Asset symbol
            resolution: Resolution in minutes (1 or 5)
            end_time: End time for data (default: current time)
            chunk_days: Number of days per API request (default: 3)
            training_days: Number of days to fetch (default: TRAINING_YEARS * 365)
            use_cache: If True, check cache and only fetch missing ranges
        
        Returns:
            Dict with OHLC data or None
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        # Calculate total time range
        if training_days is None:
            training_days = int(TRAINING_YEARS * 365)
        from_time = end_time - timedelta(days=training_days)
        total_days = (end_time - from_time).days
        
        # Check cache for existing data
        existing_data = None
        missing_ranges = None
        if use_cache:
            # Try to load existing saved data first
            existing_data = self.cache.load_data(asset, resolution)
            if existing_data:
                existing_timestamps = existing_data.get('t', [])
                existing_count = len(existing_timestamps)
                logger.info(
                    f"[{asset}] Found existing saved data: {existing_count} data points"
                )
                
                # Check if existing data covers the requested time range
                if existing_timestamps:
                    existing_start_ts = min(existing_timestamps)
                    existing_end_ts = max(existing_timestamps)
                    requested_start_ts = int(from_time.timestamp())
                    requested_end_ts = int(end_time.timestamp())
                    
                    # If existing data covers the requested range, use it directly
                    # Allow small tolerance (1 day) for timestamp matching
                    tolerance = 86400  # 1 day in seconds
                    covers_range = (
                        existing_start_ts <= (requested_start_ts + tolerance) and 
                        existing_end_ts >= (requested_end_ts - tolerance)
                    )
                    
                    if covers_range:
                        logger.info(
                            f"[{asset}] ✓ Existing data covers requested range! "
                            f"Using cached data (NO API FETCHING NEEDED)"
                        )
                        logger.info(
                            f"[{asset}] Cached data range: "
                            f"{datetime.fromtimestamp(existing_start_ts, tz=timezone.utc).date()} "
                            f"to {datetime.fromtimestamp(existing_end_ts, tz=timezone.utc).date()} "
                            f"({existing_count} points)"
                        )
                        logger.info(
                            f"[{asset}] Requested range: "
                            f"{from_time.date()} to {end_time.date()}"
                        )
                        # Filter to requested range if needed
                        if existing_start_ts < requested_start_ts or existing_end_ts > requested_end_ts:
                            filtered_indices = [
                                i for i, ts in enumerate(existing_timestamps)
                                if requested_start_ts <= ts <= requested_end_ts
                            ]
                            filtered_data = {
                                's': 'ok',
                                't': [existing_timestamps[i] for i in filtered_indices],
                                'o': [existing_data['o'][i] for i in filtered_indices],
                                'h': [existing_data['h'][i] for i in filtered_indices],
                                'l': [existing_data['l'][i] for i in filtered_indices],
                                'c': [existing_data['c'][i] for i in filtered_indices],
                            }
                            logger.info(
                                f"[{asset}] Filtered cached data to requested range: "
                                f"{len(filtered_data['t'])} points "
                                f"(from {existing_count} total)"
                            )
                            return filtered_data
                        else:
                            logger.info(
                                f"[{asset}] Using cached data as-is: {existing_count} points"
                            )
                            return existing_data
            
            # Only check for missing ranges if we don't have complete data in the file
            # (Skip this check if we already returned cached data above)
            if existing_data:
                existing_timestamps_check = existing_data.get('t', [])
                if existing_timestamps_check:
                    existing_start_ts = min(existing_timestamps_check)
                    existing_end_ts = max(existing_timestamps_check)
                    requested_start_ts = int(from_time.timestamp())
                    requested_end_ts = int(end_time.timestamp())
                    tolerance = 86400
                    if (existing_start_ts <= (requested_start_ts + tolerance) and 
                        existing_end_ts >= (requested_end_ts - tolerance)):
                        # Already returned above, but just in case - skip range check
                        missing_ranges = []
                    else:
                        logger.info(
                            f"[{asset}] Cached data exists but doesn't fully cover requested range. "
                            f"Checking for missing ranges..."
                        )
                        missing_ranges = self.cache.get_missing_ranges(
                            asset, resolution, from_time, end_time, chunk_days
                        )
                else:
                    logger.info(
                        f"[{asset}] Cached data file exists but is empty. "
                        f"Checking for missing ranges..."
                    )
                    missing_ranges = self.cache.get_missing_ranges(
                        asset, resolution, from_time, end_time, chunk_days
                    )
            else:
                logger.info(
                    f"[{asset}] No cached data file found. "
                    f"Will fetch all required ranges..."
                )
                missing_ranges = self.cache.get_missing_ranges(
                    asset, resolution, from_time, end_time, chunk_days
                )
            cached = self.cache.load_ranges(asset, resolution)
            cached_ranges_count = len(cached.get("ranges", []))
            
            if missing_ranges:
                logger.info(
                    f"[{asset}] Cache check: {cached_ranges_count} ranges already cached, "
                    f"{len(missing_ranges)} ranges need fetching"
                )
                if len(missing_ranges) > 0:
                    logger.info(
                        f"[{asset}] Missing ranges: "
                        f"{missing_ranges[0][0].date()} to {missing_ranges[-1][1].date()} "
                        f"({len(missing_ranges)} chunks)"
                    )
                    # Show first few missing ranges for visibility
                    if len(missing_ranges) <= 5:
                        for i, (start, end) in enumerate(missing_ranges, 1):
                            logger.info(
                                f"[{asset}]   Missing {i}: {start.date()} to {end.date()} "
                                f"({(end - start).days} days)"
                            )
                    else:
                        for i, (start, end) in enumerate(missing_ranges[:3], 1):
                            logger.info(
                                f"[{asset}]   Missing {i}: {start.date()} to {end.date()} "
                                f"({(end - start).days} days)"
                            )
                        logger.info(
                            f"[{asset}]   ... and {len(missing_ranges) - 3} more missing ranges"
                        )
            else:
                logger.info(
                    f"[{asset}] Cache check: All {cached_ranges_count} ranges already cached! "
                    f"Using cached data"
                )
                # All ranges are cached - return existing data if we have it
                if existing_data:
                    return existing_data
        else:
            missing_ranges = None
            logger.info(f"[{asset}] Cache disabled: fetching all data")
        
        # Calculate total expected chunks
        total_chunks = (total_days + chunk_days - 1) // chunk_days  # Ceiling division
        
        logger.info(
            f"Fetching {total_days} days of training data for {asset} "
            f"({from_time.date()} to {end_time.date()}) in {chunk_days}-day chunks "
            f"(expecting ~{total_chunks} chunks)"
        )
        
        # Fetch data in chunks
        all_timestamps = []
        all_opens = []
        all_highs = []
        all_lows = []
        all_closes = []
        
        # Determine which chunks to fetch
        if use_cache and missing_ranges:
            # Only fetch missing ranges
            chunks_to_fetch = missing_ranges
            logger.info(
                f"[{asset}] Fetching only {len(chunks_to_fetch)} missing chunks "
                f"(skipping {total_chunks - len(chunks_to_fetch)} cached chunks)"
            )
        else:
            # Fetch all chunks
            chunks_to_fetch = []
            current = from_time
            while current < end_time:
                chunk_end = min(current + timedelta(days=chunk_days), end_time)
                chunks_to_fetch.append((current, chunk_end))
                current = chunk_end
        
        successful_chunks = 0
        failed_chunks = 0
        fetched_ranges = []  # Track successfully fetched ranges for cache update
        
        for chunk_num, (current_from, current_to) in enumerate(chunks_to_fetch, 1):
            from_ts = int(current_from.timestamp())
            to_ts = int(current_to.timestamp())
            
            # Check if this range is cached (double-check in case cache was updated)
            if use_cache:
                cached = self.cache.load_ranges(asset, resolution)
                cached_ranges = set()
                for range_item in cached.get("ranges", []):
                    # Convert list to tuple (lists are not hashable for sets)
                    if isinstance(range_item, list) and len(range_item) == 2:
                        cached_ranges.add(tuple(range_item))
                    elif isinstance(range_item, tuple) and len(range_item) == 2:
                        cached_ranges.add(range_item)
                if (from_ts, to_ts) in cached_ranges:
                    logger.info(
                        f"[{asset}] Chunk {chunk_num}/{len(chunks_to_fetch)} SKIPPED (cached): "
                        f"{current_from.date()} to {current_to.date()}"
                    )
                    continue
            
            logger.info(
                f"[{asset}] Chunk {chunk_num}/{len(chunks_to_fetch)}: "
                f"Fetching {current_from.date()} to {current_to.date()} "
                f"({chunk_days} days, timestamps {from_ts} to {to_ts})"
            )
            
            try:
                data = self.fetch_history(asset, resolution, from_ts, to_ts)
            except Exception as e:
                failed_chunks += 1
                logger.error(
                    f"[{asset}] Chunk {chunk_num}/{total_chunks} FAILED with exception: "
                    f"{type(e).__name__}: {str(e)}"
                )
                logger.error(
                    f"[{asset}] Chunk {chunk_num} details: "
                    f"from={current_from.date()}, to={current_to.date()}, "
                    f"from_ts={from_ts}, to_ts={to_ts}"
                )
                # Continue with next chunk instead of failing completely
                current_from = current_to
                continue
            
            if data is None:
                failed_chunks += 1
                logger.warning(
                    f"[{asset}] Chunk {chunk_num}/{total_chunks} FAILED: "
                    f"fetch_history returned None for {current_from.date()} to {current_to.date()}"
                )
                # Continue with next chunk instead of failing completely
                current_from = current_to
                continue
            
            # Check status
            status = data.get("s")
            if status != "ok":
                failed_chunks += 1
                error_msg = data.get("errmsg", "No error message")
                logger.warning(
                    f"[{asset}] Chunk {chunk_num}/{total_chunks} FAILED: "
                    f"API returned status '{status}' with error: {error_msg}"
                )
                logger.warning(
                    f"[{asset}] Chunk {chunk_num} request params: "
                    f"symbol={TOKEN_MAP.get(asset)}, resolution={resolution}, "
                    f"from={from_ts}, to={to_ts}"
                )
                current_from = current_to
                continue
            
            # Extract data
            timestamps = data.get("t", [])
            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            
            if timestamps and closes:
                points_fetched = len(timestamps)
                all_timestamps.extend(timestamps)
                all_opens.extend(opens)
                all_highs.extend(highs)
                all_lows.extend(lows)
                all_closes.extend(closes)
                successful_chunks += 1
                fetched_ranges.append((current_from, current_to))  # Track for cache
                logger.info(
                    f"[{asset}] Chunk {chunk_num}/{len(chunks_to_fetch)} SUCCESS: "
                    f"fetched {points_fetched} data points "
                    f"(total so far: {len(all_timestamps)} points, "
                    f"{successful_chunks} successful, {failed_chunks} failed)"
                )
            else:
                failed_chunks += 1
                logger.warning(
                    f"[{asset}] Chunk {chunk_num}/{total_chunks} FAILED: "
                    f"returned empty data (timestamps={len(timestamps) if timestamps else 0}, "
                    f"closes={len(closes) if closes else 0})"
                )
            
            # Small delay between chunks to avoid rate limiting
            time.sleep(0.5)
        
        # Merge with existing data if available
        if existing_data and all_timestamps:
            logger.info(
                f"[{asset}] Merging {len(all_timestamps)} new points with "
                f"{len(existing_data.get('t', []))} existing points"
            )
            merged_data = self.cache.merge_data(existing_data, {
                's': 'ok',
                't': all_timestamps,
                'o': all_opens,
                'h': all_highs,
                'l': all_lows,
                'c': all_closes,
            })
            all_timestamps = merged_data['t']
            all_opens = merged_data['o']
            all_highs = merged_data['h']
            all_lows = merged_data['l']
            all_closes = merged_data['c']
            logger.info(
                f"[{asset}] After merge: {len(all_timestamps)} total data points"
            )
        
        # Save data to disk
        if use_cache and all_timestamps:
            final_data = {
                's': 'ok',
                't': all_timestamps,
                'o': all_opens,
                'h': all_highs,
                'l': all_lows,
                'c': all_closes,
            }
            self.cache.save_data(asset, resolution, final_data)
            logger.info(f"[{asset}] Saved {len(all_timestamps)} data points to disk")
        
        # Update cache with successfully fetched ranges
        if use_cache and fetched_ranges:
            self.cache.update_cache(asset, resolution, fetched_ranges)
            logger.info(
                f"[{asset}] Updated cache with {len(fetched_ranges)} new ranges"
            )
        
        # Final summary
        logger.info("=" * 80)
        logger.info(f"[{asset}] DATA FETCHING COMPLETE - Summary:")
        if use_cache:
            logger.info(f"  Cache Status: ENABLED")
            if existing_data:
                existing_count = len(existing_data.get('t', []))
                logger.info(f"  Cached data loaded: {existing_count:,} points")
            else:
                logger.info(f"  Cached data: None (first time fetching)")
            if missing_ranges:
                logger.info(f"  Missing ranges identified: {len(missing_ranges)} chunks")
            else:
                logger.info(f"  Missing ranges: None (all data was cached)")
        else:
            logger.info(f"  Cache Status: DISABLED")
        logger.info(f"  Chunks attempted: {len(chunks_to_fetch)}")
        logger.info(f"  Chunks fetched successfully: {successful_chunks}")
        logger.info(f"  Chunks failed: {failed_chunks}")
        if use_cache and existing_data:
            skipped = len(chunks_to_fetch) - successful_chunks - failed_chunks
            if skipped > 0:
                logger.info(f"  Chunks skipped (from cache): {skipped}")
        logger.info(f"  Total data points available: {len(all_timestamps):,}")
        if all_timestamps:
            data_start = datetime.fromtimestamp(min(all_timestamps), tz=timezone.utc)
            data_end = datetime.fromtimestamp(max(all_timestamps), tz=timezone.utc)
            logger.info(f"  Data range: {data_start.date()} to {data_end.date()}")
        logger.info("=" * 80)
        
        if not all_timestamps:
            logger.error(
                f"[{asset}] CRITICAL: No data fetched after {chunk_num} chunks. "
                f"Cannot proceed with training."
            )
            return None
        
        # Calculate success rate
        attempted = len(chunks_to_fetch)
        success_rate = (successful_chunks / attempted * 100) if attempted > 0 else 0
        
        if successful_chunks < attempted:
            logger.warning(
                f"[{asset}] WARNING: Only {successful_chunks}/{attempted} chunks succeeded "
                f"({success_rate:.1f}% success rate). "
                f"Training will proceed with partial data ({len(all_timestamps)} points)."
            )
        else:
            logger.info(
                f"[{asset}] SUCCESS: All {successful_chunks} chunks fetched successfully "
                f"({len(all_timestamps)} total data points)"
            )
        
        return {
            "s": "ok",
            "t": all_timestamps,
            "o": all_opens,
            "h": all_highs,
            "l": all_lows,
            "c": all_closes,
        }


class HermesFetcher:
    """Fetches current anchor price from Hermes API"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=1, max=5),
        reraise=True
    )
    def fetch_anchor_price(self, asset: str) -> Optional[float]:
        """Fetch current anchor price from Hermes"""
        if asset not in HERMES_FEED_IDS:
            logger.error(f"No Hermes feed ID for asset: {asset}")
            return None
            
        feed_id = HERMES_FEED_IDS[asset]
        params = {"ids[]": [feed_id]}
        
        try:
            response = requests.get(HERMES_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            parsed = data.get("parsed", [])
            if not parsed:
                logger.warning(f"No parsed data for {asset}")
                return None
                
            price_info = parsed[0].get("price", {})
            if not price_info:
                logger.warning(f"No price info for {asset}")
                return None
                
            price = int(price_info.get("price", 0))
            expo = int(price_info.get("expo", 0))
            
            anchor_price = price * (10 ** expo)
            return float(anchor_price)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Hermes price: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Hermes response: {e}")
            return None
    
    def get_anchor_price_with_fallback(self, asset: str) -> float:
        """Get anchor price, falling back to default if Hermes fails"""
        price = self.fetch_anchor_price(asset)
        if price is None:
            fallback = FALLBACK_PRICES.get(asset, 100.0)
            logger.warning(f"Using fallback price for {asset}: {fallback}")
            return fallback
        return price


def parse_benchmarks_ohlc(data: Dict) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Parse Benchmarks API response into OHLC data
    
    Returns:
        (timestamps, opens, highs, lows, closes) as lists
    """
    timestamps = data.get("t", [])
    opens = data.get("o", [])
    highs = data.get("h", [])
    lows = data.get("l", [])
    closes = data.get("c", [])
    
    # Filter out None/missing values and ensure all arrays have same length
    valid_data = []
    max_len = max(len(timestamps), len(opens), len(highs), len(lows), len(closes))
    
    for i in range(max_len):
        t = timestamps[i] if i < len(timestamps) else None
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        
        # Use close as fallback for missing OHLC
        if c is not None and c > 0:
            o_val = o if o is not None and o > 0 else c
            h_val = h if h is not None and h > 0 else c
            l_val = l if l is not None and l > 0 else c
            
            if t is not None:
                valid_data.append((t, o_val, h_val, l_val, c))
    
    if not valid_data:
        return [], [], [], [], []
    
    timestamps_clean, opens_clean, highs_clean, lows_clean, closes_clean = zip(*valid_data)
    return list(timestamps_clean), list(opens_clean), list(highs_clean), list(lows_clean), list(closes_clean)