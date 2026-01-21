"""
Data fetching from Pyth Benchmarks and Hermes APIs
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import (
    BASE_URL, HERMES_BASE_URL, TOKEN_MAP, HERMES_FEED_IDS,
    FALLBACK_PRICES
)

logger = logging.getLogger(__name__)


class BenchmarksFetcher:
    """Fetches historical price data from Pyth Benchmarks TradingView API"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=1, max=5),
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
        Fetch historical data from Benchmarks API
        
        Returns:
            Dict with keys: 's' (status), 't' (timestamps), 'c' (closes)
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
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # The Benchmarks TradingView shim uses an "s" field for status, but in practice
            # some non-"ok" values (e.g. "no_data") can still be returned alongside a
            # well-formed payload. Our training pipeline only cares about having valid
            # timestamps/prices, so we prefer to rely on the presence of "t"/"c" instead
            # of strictly gating on data["s"] == "ok".
            status = data.get("s")
            if status != "ok":
                logger.warning(f"Benchmarks status for {asset}: {status}, raw response: {data}")

            timestamps = data.get("t") or []
            closes = data.get("c") or []

            if not timestamps or not closes:
                logger.warning(f"No Benchmarks data points for {asset} with params={params}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Benchmarks data: {e}")
            return None


class HermesFetcher:
    """Fetches current anchor price from Hermes API"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=1, max=5),
        reraise=True
    )
    def fetch_anchor_price(self, asset: str) -> Optional[float]:
        """
        Fetch current anchor price from Hermes
        
        Returns:
            Price as float, or None if error
        """
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
        """
        Get anchor price, falling back to default if Hermes fails
        """
        price = self.fetch_anchor_price(asset)
        if price is None:
            fallback = FALLBACK_PRICES.get(asset, 100.0)
            logger.warning(f"Using fallback price for {asset}: {fallback}")
            return fallback
        return price


def parse_benchmarks_data(data: Dict) -> Tuple[List[int], List[float]]:
    """
    Parse Benchmarks API response into timestamps and closes
    
    Returns:
        (timestamps, closes) as lists
    """
    timestamps = data.get("t", [])
    closes = data.get("c", [])
    
    # Filter out None/missing values
    valid_pairs = [
        (t, c) for t, c in zip(timestamps, closes)
        if t is not None and c is not None and c > 0
    ]
    
    if not valid_pairs:
        return [], []
    
    timestamps_clean, closes_clean = zip(*valid_pairs)
    return list(timestamps_clean), list(closes_clean)
