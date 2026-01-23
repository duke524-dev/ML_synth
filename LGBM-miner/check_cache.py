#!/usr/bin/env python3
"""
Utility script to check for missing data in cached files.

Usage:
    python check_cache.py BTC 1                    # Check BTC 1-minute data
    python check_cache.py BTC 1 --request-days 365  # Check against 365 days from now
    python check_cache.py --all                     # Check all cached assets
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import msgpack
import json

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS


class CacheChecker:
    """Check for missing data in cached files"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data_file(self, asset: str, resolution: int) -> str:
        """Get data file path for an asset/resolution"""
        return os.path.join(self.cache_dir, f"{asset}_r{resolution}_data.msgpack")
    
    def get_ranges_file(self, asset: str, resolution: int) -> str:
        """Get ranges file path for an asset/resolution"""
        return os.path.join(self.cache_dir, f"{asset}_r{resolution}_ranges.json")
    
    def load_data(self, asset: str, resolution: int) -> Optional[Dict]:
        """Load cached data file"""
        data_file = self.get_data_file(asset, resolution)
        if not os.path.exists(data_file):
            return None
        
        try:
            with open(data_file, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
                return data
        except Exception as e:
            print(f"Error loading data for {asset} r{resolution}: {e}")
            return None
    
    def load_ranges(self, asset: str, resolution: int) -> Dict:
        """Load cached ranges metadata"""
        ranges_file = self.get_ranges_file(asset, resolution)
        if not os.path.exists(ranges_file):
            return {"ranges": []}
        
        try:
            with open(ranges_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ranges for {asset} r{resolution}: {e}")
            return {"ranges": []}
    
    def analyze_data(self, asset: str, resolution: int, 
                    requested_start: Optional[datetime] = None,
                    requested_end: Optional[datetime] = None) -> Dict:
        """
        Analyze cached data for gaps and coverage
        
        Returns:
            Dict with analysis results
        """
        data = self.load_data(asset, resolution)
        ranges_meta = self.load_ranges(asset, resolution)
        
        if data is None:
            return {
                "exists": False,
                "message": f"No cached data file found for {asset} r{resolution}"
            }
        
        timestamps = data.get('t', [])
        if not timestamps:
            return {
                "exists": True,
                "message": f"Cached data file exists but is empty for {asset} r{resolution}"
            }
        
        # Sort timestamps (should already be sorted, but just in case)
        timestamps = sorted(timestamps)
        
        # Calculate expected interval based on resolution (in seconds)
        expected_interval = resolution * 60
        
        # Find gaps in the data
        gaps = []
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > expected_interval * 2:  # More than 2x expected interval = gap
                gap_start = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
                gap_end = datetime.fromtimestamp(timestamps[i + 1], tz=timezone.utc)
                gap_duration = gap_end - gap_start
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "duration_seconds": gap,
                    "duration_hours": gap / 3600,
                    "missing_points": int(gap / expected_interval) - 1
                })
        
        # Calculate coverage
        data_start = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        data_end = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)
        data_span = data_end - data_start
        total_points = len(timestamps)
        
        # Check against requested range if provided
        coverage_info = {}
        if requested_start and requested_end:
            requested_span = requested_end - requested_start
            expected_points = int(requested_span.total_seconds() / expected_interval)
            
            # Find missing ranges in requested period
            missing_ranges = []
            current = requested_start
            
            # Check if data starts before requested
            if data_start > requested_start:
                missing_ranges.append({
                    "start": requested_start,
                    "end": data_start,
                    "type": "before_data"
                })
            
            # Check gaps within data range
            for gap in gaps:
                if gap["start"] >= requested_start and gap["end"] <= requested_end:
                    missing_ranges.append({
                        "start": gap["start"],
                        "end": gap["end"],
                        "type": "gap"
                    })
            
            # Check if data ends before requested
            if data_end < requested_end:
                missing_ranges.append({
                    "start": data_end,
                    "end": requested_end,
                    "type": "after_data"
                })
            
            coverage_info = {
                "requested_start": requested_start,
                "requested_end": requested_end,
                "requested_span_days": requested_span.days,
                "expected_points": expected_points,
                "actual_points": total_points,
                "coverage_percent": (total_points / expected_points * 100) if expected_points > 0 else 0,
                "missing_ranges": missing_ranges,
                "missing_ranges_count": len(missing_ranges)
            }
        
        # Get ranges metadata
        cached_ranges = ranges_meta.get("ranges", [])
        ranges_count = len(cached_ranges)
        
        return {
            "exists": True,
            "asset": asset,
            "resolution": resolution,
            "data_start": data_start,
            "data_end": data_end,
            "data_span_days": data_span.days,
            "total_points": total_points,
            "expected_interval_seconds": expected_interval,
            "gaps": gaps,
            "gaps_count": len(gaps),
            "cached_ranges_count": ranges_count,
            "coverage": coverage_info if coverage_info else None
        }
    
    def print_analysis(self, analysis: Dict):
        """Print analysis results in a readable format"""
        if not analysis.get("exists"):
            print(f"❌ {analysis['message']}")
            return
        
        asset = analysis["asset"]
        resolution = analysis["resolution"]
        
        print(f"\n{'='*70}")
        print(f"📊 Cache Analysis: {asset} (Resolution: {resolution} minute{'s' if resolution > 1 else ''})")
        print(f"{'='*70}")
        
        # Basic info
        print(f"\n📁 Data File:")
        print(f"   Start:  {analysis['data_start'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   End:    {analysis['data_end'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   Span:   {analysis['data_span_days']} days")
        print(f"   Points: {analysis['total_points']:,}")
        print(f"   Expected interval: {analysis['expected_interval_seconds']} seconds")
        
        # Gaps
        if analysis['gaps_count'] > 0:
            print(f"\n⚠️  Found {analysis['gaps_count']} gap(s) in data:")
            for i, gap in enumerate(analysis['gaps'][:10], 1):  # Show first 10
                print(f"   Gap {i}:")
                print(f"      From:  {gap['start'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"      To:    {gap['end'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"      Duration: {gap['duration_hours']:.1f} hours ({gap['duration_seconds']/3600:.2f} hours)")
                print(f"      Missing points: ~{gap['missing_points']:,}")
            if analysis['gaps_count'] > 10:
                print(f"   ... and {analysis['gaps_count'] - 10} more gaps")
        else:
            print(f"\n✅ No gaps found in data")
        
        # Ranges metadata
        print(f"\n📋 Cached Ranges Metadata:")
        print(f"   Ranges count: {analysis['cached_ranges_count']}")
        
        # Coverage info
        if analysis.get('coverage'):
            cov = analysis['coverage']
            print(f"\n🎯 Coverage Analysis (Requested Range):")
            print(f"   Requested: {cov['requested_start'].strftime('%Y-%m-%d')} to {cov['requested_end'].strftime('%Y-%m-%d')}")
            print(f"   Span: {cov['requested_span_days']} days")
            print(f"   Expected points: {cov['expected_points']:,}")
            print(f"   Actual points: {cov['actual_points']:,}")
            print(f"   Coverage: {cov['coverage_percent']:.1f}%")
            
            if cov['missing_ranges_count'] > 0:
                print(f"\n   ⚠️  Missing ranges ({cov['missing_ranges_count']}):")
                for i, missing in enumerate(cov['missing_ranges'][:10], 1):
                    print(f"      {i}. {missing['type']}: {missing['start'].strftime('%Y-%m-%d %H:%M')} to {missing['end'].strftime('%Y-%m-%d %H:%M')}")
                if cov['missing_ranges_count'] > 10:
                    print(f"      ... and {cov['missing_ranges_count'] - 10} more")
            else:
                print(f"\n   ✅ Full coverage of requested range")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check for missing data in cached files"
    )
    parser.add_argument(
        "asset",
        nargs="?",
        help="Asset symbol (e.g., BTC, ETH). Use --all to check all assets."
    )
    parser.add_argument(
        "resolution",
        type=int,
        nargs="?",
        default=1,
        help="Resolution in minutes (default: 1)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all cached assets"
    )
    parser.add_argument(
        "--request-days",
        type=int,
        help="Number of days back from now to check coverage for"
    )
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help="Cache directory (default: data_cache)"
    )
    
    args = parser.parse_args()
    
    checker = CacheChecker(cache_dir=args.cache_dir)
    
    # Determine requested range
    requested_start = None
    requested_end = None
    if args.request_days:
        requested_end = datetime.now(timezone.utc)
        requested_start = requested_end - timedelta(days=args.request_days)
    
    # Get assets to check
    if args.all:
        # Check all assets that might be cached
        all_assets = set(HF_ASSETS) | set(LF_CRYPTO_ASSETS) | set(LF_EQUITY_ASSETS)
        assets_to_check = []
        for asset in sorted(all_assets):
            data_file = checker.get_data_file(asset, args.resolution)
            if os.path.exists(data_file):
                assets_to_check.append((asset, args.resolution))
    elif args.asset:
        assets_to_check = [(args.asset, args.resolution)]
    else:
        parser.print_help()
        return
    
    # Analyze each asset
    for asset, resolution in assets_to_check:
        analysis = checker.analyze_data(
            asset, resolution,
            requested_start=requested_start,
            requested_end=requested_end
        )
        checker.print_analysis(analysis)


if __name__ == "__main__":
    main()
