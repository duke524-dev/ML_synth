import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np

# Ensure project root (which contains the 'synth' package) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from synth.miner.simulations import generate_simulations
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.price_data_provider import PriceDataProvider
from synth.validator import prompt_config
from synth.utils.helpers import adjust_predictions
from synth.db.models import ValidatorRequest


PromptResult = Dict[str, Any]


DayPriceSeries = Dict[str, Any]
PriceCache = Dict[Tuple[str, str], DayPriceSeries]


def prefetch_day_prices(
    asset: str, day_start: datetime, provider: PriceDataProvider
) -> DayPriceSeries:
    """
    Fetch 1-minute resolution prices for a single asset and UTC day,
    save to a JSON file, and return the loaded object.

    File name pattern: offline_CRPS_test/pyth_1m_{asset}_{YYYY-MM-DD}.json
    """
    day_str = day_start.date().isoformat()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, f"pyth_1m_{asset}_{day_str}.json")

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        # Console log for reuse
        print(f"[PREFETCH] Using cached 1m data for {asset} on {day_str}")
        return data

    print(f"[PREFETCH] Fetching 1m data for {asset} on {day_str} from Pyth...")
    # One full day at 1-minute resolution
    vreq = ValidatorRequest(
        asset=asset,
        start_time=day_start,
        time_increment=60,
        time_length=24 * 3600,
    )
    prices = provider.fetch_data(vreq)

    obj: DayPriceSeries = {
        "asset": asset,
        "day": day_str,
        "start_time_iso": day_start.isoformat(),
        "time_increment": 60,
        "prices": prices,
    }
    with open(filename, "w") as f:
        json.dump(obj, f)

    return obj


def build_price_cache(start_dt: datetime, num_days: int) -> PriceCache:
    """
    Build a cache of 1-minute price series for all assets and days
    in the test range.
    """
    provider = PriceDataProvider()
    low_assets = set(prompt_config.LOW_FREQUENCY.asset_list)
    high_assets = set(prompt_config.HIGH_FREQUENCY.asset_list)
    all_assets = sorted(low_assets | high_assets)

    cache: PriceCache = {}

    for day_offset in range(num_days):
        day_dt = start_dt + timedelta(days=day_offset)
        for asset in all_assets:
            series = prefetch_day_prices(asset, day_dt, provider)
            cache[(asset, day_dt.date().isoformat())] = series

    return cache


def run_daily_baseline_crps_for_prompt(
    prompt_cfg: prompt_config.PromptConfig,
    prompt_label: str,
    day_start: datetime,
    all_prompt_results: List[PromptResult],
    price_cache: PriceCache,
) -> None:
    """
    Simulate one day of prompts for the built-in baseline miner (synth.miner.simulations.generate_simulations),
    without instantiating any Bittensor neuron.

    For each asset in the prompt config and each hourly prompt in the given day:
      - Build "fake" request params (asset, start_time, time_increment, time_length, num_simulations)
      - Call generate_simulations(...)
      - Fetch real prices from Pyth via PriceDataProvider
      - Compute CRPS with calculate_crps_for_miner
      - Append a per-prompt record into all_prompt_results
    """

    scoring_intervals = prompt_cfg.scoring_intervals

    if day_start.tzinfo is None:
        day_start = day_start.replace(tzinfo=timezone.utc)

    # Number of checking cycles per day is determined by total_cycle_minutes
    minutes_per_day = 24 * 60
    cycles_per_day = minutes_per_day // prompt_cfg.total_cycle_minutes

    # Use EWMA state-update function when test_wrapper is already loaded (e.g. run_crps_test)
    update_states_fn = None
    if "test_wrapper" in sys.modules:
        mod = sys.modules["test_wrapper"]
        update_states_fn = getattr(mod, "update_states_from_price_data", None)

    for asset in prompt_cfg.asset_list:
        for cycle_idx in range(cycles_per_day):
            # Real validator starts a prompt every total_cycle_minutes
            prompt_start = day_start + timedelta(
                minutes=cycle_idx * prompt_cfg.total_cycle_minutes
            )
            start_iso = prompt_start.isoformat()

            # Log the "fake request" parameters explicitly
            request_info: PromptResult = {
                "asset": asset,
                "prompt_type": prompt_label,
                "prompt_start": start_iso,
                "prompt_index": cycle_idx,
                "time_increment": prompt_cfg.time_increment,
                "time_length": prompt_cfg.time_length,
                "num_simulations": prompt_cfg.num_simulations,
            }

            # Console log for each request
            print(
                f"[{prompt_label}] request "
                f"asset={asset} "
                f"cycle={cycle_idx + 1}/{cycles_per_day} "
                f"start={start_iso} "
                f"dt={prompt_cfg.time_increment}s "
                f"len={prompt_cfg.time_length}s "
                f"sims={prompt_cfg.num_simulations}"
            )

            # Load cached 1-minute price data (used for both starting price and CRPS)
            day_key = prompt_start.date().isoformat()
            cache_key = (asset, day_key)
            if cache_key not in price_cache:
                rec = dict(request_info)
                rec["error"] = f"no_prefetched_data_for_{asset}_{day_key}"
                all_prompt_results.append(rec)
                continue

            day_data = price_cache[cache_key]
            base_start = datetime.fromisoformat(day_data["start_time_iso"])
            series_1m = day_data["prices"]
            dt_1m = int(day_data.get("time_increment", 60))

            # Extract starting price from cached 1-minute series at prompt_start
            offset_minutes = int(
                (prompt_start - base_start).total_seconds() // dt_1m
            )
            if 0 <= offset_minutes < len(series_1m):
                current_price = float(series_1m[offset_minutes])
                if np.isnan(current_price):
                    # Fallback: find first non-NaN price in the day
                    for p in series_1m:
                        if not np.isnan(p):
                            current_price = float(p)
                            break
                    else:
                        current_price = None
            else:
                current_price = None

            if current_price is None:
                rec = dict(request_info)
                rec["error"] = f"no_valid_price_at_{start_iso}"
                all_prompt_results.append(rec)
                continue

            # 0) Update state from price data up to prompt_start (if supported by miner)
            if update_states_fn is not None:
                try:
                    update_states_fn(
                        asset=asset,
                        prices=series_1m,
                        base_start=base_start,
                        target_time=prompt_start,
                    )
                except Exception:
                    pass

            # 1) Call baseline simulation generator with patched get_asset_price
            # This avoids the live HTTP call to hermes.pyth.network
            print(
                f"[{prompt_label}] fetch real prices "
                f"asset={asset} "
                f"start={start_iso} "
                f"len={prompt_cfg.time_length}s "
                f"dt={prompt_cfg.time_increment}s"
            )
            try:
                with patch(
                    "synth.miner.simulations.get_asset_price",
                    return_value=current_price,
                ):
                    predictions = generate_simulations(
                        asset=asset,
                        start_time=start_iso,
                        time_increment=prompt_cfg.time_increment,
                        time_length=prompt_cfg.time_length,
                        num_simulations=prompt_cfg.num_simulations,
                    )
            except Exception as e:
                rec = dict(request_info)
                rec["error"] = f"generate_simulations_failed: {e}"
                all_prompt_results.append(rec)
                continue

            # Baseline returns tuple: (start_ts, time_increment, [price_list_1, ...])
            predictions_list = list(predictions)
            predictions_path = adjust_predictions(predictions_list)
            if predictions_path is None:
                rec = dict(request_info)
                rec["error"] = "invalid_prediction_format"
                all_prompt_results.append(rec)
                continue

            try:
                simulation_runs = np.array(predictions_path).astype(float)
            except Exception as e:
                rec = dict(request_info)
                rec["error"] = f"prediction_to_numpy_failed: {e}"
                all_prompt_results.append(rec)
                continue

            # 2) Extract real prices for CRPS (may span multiple days for 24h predictions)
            points_needed = prompt_cfg.time_length // dt_1m + 1
            
            # Check if prediction spans into next day
            end_minutes = offset_minutes + points_needed
            if end_minutes <= len(series_1m):
                # All data fits in current day
                segment_1m = series_1m[offset_minutes : end_minutes]
            else:
                # Prediction spans into next day - need to load next day's data
                # Get remaining points from current day
                remaining_from_current = len(series_1m) - offset_minutes
                segment_1m = list(series_1m[offset_minutes:])
                
                # Calculate how many points needed from next day
                points_from_next_day = points_needed - remaining_from_current
                
                # Load next day's data
                next_day = prompt_start.date() + timedelta(days=1)
                next_day_key = next_day.isoformat()
                next_cache_key = (asset, next_day_key)
                
                if next_cache_key not in price_cache:
                    rec = dict(request_info)
                    rec["error"] = f"prefetch_range_too_short: need next day {next_day_key} data"
                    all_prompt_results.append(rec)
                    continue
                
                next_day_data = price_cache[next_cache_key]
                next_series_1m = next_day_data["prices"]
                
                # Append next day's data
                segment_1m.extend(next_series_1m[:points_from_next_day])
            
            if len(segment_1m) < points_needed:
                rec = dict(request_info)
                rec["error"] = "prefetch_range_too_short"
                all_prompt_results.append(rec)
                continue

            # Downsample from 1-minute to the prompt's increment
            factor = prompt_cfg.time_increment // dt_1m
            if factor <= 0:
                factor = 1
            segment = segment_1m[::factor]
            real_prices_array = np.array(segment, dtype=float)

            # 3) Compute CRPS for this prompt
            try:
                total_crps, detailed = calculate_crps_for_miner(
                    simulation_runs=simulation_runs,
                    real_price_path=real_prices_array,
                    time_increment=prompt_cfg.time_increment,
                    scoring_intervals=scoring_intervals,
                )
            except Exception as e:
                rec = dict(request_info)
                rec["error"] = f"crps_calculation_failed: {e}"
                all_prompt_results.append(rec)
                continue

            intervals_totals: Dict[str, float] = {}
            overall_crps: float | None = None
            for entry in detailed:
                if entry.get("Increment") == "Total":
                    interval_name = entry.get("Interval")
                    if interval_name == "Overall":
                        overall_crps = float(entry["CRPS"])
                    else:
                        intervals_totals[interval_name] = float(entry["CRPS"])

            if overall_crps is None:
                overall_crps = float(total_crps)

            rec = dict(request_info)
            rec["overall_crps"] = overall_crps
            rec["intervals"] = intervals_totals
            all_prompt_results.append(rec)


def compute_summaries(
    all_prompt_results: List[PromptResult],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute total and average CRPS per asset and per prompt_type.

    Returns a nested dict:
      summaries[asset][prompt_type] = {
        "overall": {...},
        "intervals": { interval_name: {...}, ... },
      }
    """

    asset_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pr in all_prompt_results:
        if "overall_crps" not in pr or "intervals" not in pr:
            # Error record, skip from stats
            continue

        asset = pr["asset"]
        prompt_type = pr["prompt_type"]
        overall_val = float(pr["overall_crps"])
        intervals: Dict[str, float] = pr["intervals"]

        if asset not in asset_stats:
            asset_stats[asset] = {}
        if prompt_type not in asset_stats[asset]:
            asset_stats[asset][prompt_type] = {
                "overall": {"sum": 0.0, "count": 0},
                "intervals": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            }

        cur = asset_stats[asset][prompt_type]
        cur["overall"]["sum"] += overall_val
        cur["overall"]["count"] += 1

        for interval_name, crps_val in intervals.items():
            cur["intervals"][interval_name]["sum"] += float(crps_val)
            cur["intervals"][interval_name]["count"] += 1

    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for asset, per_type in asset_stats.items():
        summaries[asset] = {}
        for prompt_type, stats in per_type.items():
            overall_sum = stats["overall"]["sum"]
            overall_count = stats["overall"]["count"]
            overall_avg = overall_sum / overall_count if overall_count > 0 else float(
                "nan"
            )

            intervals_summary: Dict[str, Dict[str, Any]] = {}
            for name, istats in stats["intervals"].items():
                isum = istats["sum"]
                icnt = istats["count"]
                iavg = isum / icnt if icnt > 0 else float("nan")
                intervals_summary[name] = {
                    "total_crps": isum,
                    "avg_crps": iavg,
                    "num_prompts": icnt,
                }

            summaries[asset][prompt_type] = {
                "overall": {
                    "total_crps": overall_sum,
                    "avg_crps": overall_avg,
                    "num_prompts": overall_count,
                },
                "intervals": intervals_summary,
            }

    return summaries


def write_results(
    all_prompt_results: List[PromptResult],
    summaries: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
    start_day: str,
    end_day: str,
) -> None:
    """
    Write per-asset average CRPS summary to CSV and console.

    File names include:
      - test run date (UTC now, YYYYMMDD)
      - tested date range (start_day and end_day)
    """

    os.makedirs(output_dir, exist_ok=True)

    run_date = datetime.utcnow().strftime("%Y%m%d")
    range_tag = f"{start_day}_to_{end_day}"

    # Simple per-asset summary: average CRPS for LOW and HIGH frequency
    asset_avg_csv = os.path.join(
        output_dir,
        f"baseline_crps_asset_avg_run-{run_date}_range-{range_tag}.csv",
    )
    with open(asset_avg_csv, "w", newline="") as f:
        fieldnames = ["asset", "LOW_FREQUENCY_avg_crps", "HIGH_FREQUENCY_avg_crps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Collect all assets from both prompt types
        all_assets = set()
        for asset in summaries.keys():
            all_assets.add(asset)

        for asset in sorted(all_assets):
            low_avg = summaries.get(asset, {}).get("LOW_FREQUENCY", {}).get(
                "overall", {}
            ).get("avg_crps", float("nan"))
            high_avg = summaries.get(asset, {}).get("HIGH_FREQUENCY", {}).get(
                "overall", {}
            ).get("avg_crps", float("nan"))

            writer.writerow(
                {
                    "asset": asset,
                    "LOW_FREQUENCY_avg_crps": low_avg if not np.isnan(low_avg) else "",
                    "HIGH_FREQUENCY_avg_crps": high_avg
                    if not np.isnan(high_avg)
                    else "",
                }
            )

    # Console output: per-asset average CRPS
    print("\n" + "=" * 70)
    print("AVERAGE CRPS PER ASSET")
    print("=" * 70)
    print(f"{'Asset':<10} {'LOW_FREQUENCY':<20} {'HIGH_FREQUENCY':<20}")
    print("-" * 70)

    all_assets = set()
    for asset in summaries.keys():
        all_assets.add(asset)

    for asset in sorted(all_assets):
        low_avg = summaries.get(asset, {}).get("LOW_FREQUENCY", {}).get(
            "overall", {}
        ).get("avg_crps", float("nan"))
        high_avg = summaries.get(asset, {}).get("HIGH_FREQUENCY", {}).get(
            "overall", {}
        ).get("avg_crps", float("nan"))

        low_str = f"{low_avg:.4f}" if not np.isnan(low_avg) else "N/A"
        high_str = f"{high_avg:.4f}" if not np.isnan(high_avg) else "N/A"
        print(f"{asset:<10} {low_str:<20} {high_str:<20}")

    print("=" * 70 + "\n")

    print(f"Baseline offline CRPS results written to: {asset_avg_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simplified offline CRPS tester for the baseline miner "
            "(synth.miner.simulations.generate_simulations). "
            "No bittensor neuron or wallet is instantiated; we just make fake "
            "requests, call the simulation function, and score CRPS."
        )
    )
    parser.add_argument(
        "--start-day",
        type=str,
        required=True,
        help="Start day in YYYY-MM-DD (UTC), e.g. 2025-02-01",
    )
    parser.add_argument(
        "--num-days",
        type=int,
        required=True,
        help="Number of days to test from start-day (e.g. 1, 3, 7)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma-separated list of assets to test (e.g. 'BTC,ETH'). If not specified, tests all assets from prompt config.",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["low", "high", "both"],
        default="both",
        help="Which prompt type(s) to test: 'low' (LOW_FREQUENCY), 'high' (HIGH_FREQUENCY), or 'both' (default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    start_day = args.start_day
    num_days = args.num_days

    start_dt = datetime.fromisoformat(start_day).replace(tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=max(num_days - 1, 0))
    end_day = end_dt.date().isoformat()

    all_prompt_results: List[PromptResult] = []

    # Parse asset filter if provided
    asset_filter = None
    if args.assets:
        asset_filter = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
        print(f"[FILTER] Testing only assets: {', '.join(asset_filter)}")

    # Prefetch 1-minute prices for all assets and days in the range
    # Calculate warm-up period (same as real miner: 72h for crypto, 5 days for equity)
    try:
        import sys
        import os
        ewma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EWMA-miner")
        if ewma_dir not in sys.path:
            sys.path.insert(0, ewma_dir)
        import config as ewma_config
        max_warmup_days = max(ewma_config.WARMUP_1M_HOURS // 24, ewma_config.WARMUP_5M_DAYS)
    except ImportError:
        # Fallback to 1 day if EWMA config not available
        max_warmup_days = 1
    
    # Fetch: warmup days + num_days + 1 day after (for cross-day predictions)
    prefetch_days = max_warmup_days + num_days + 1
    prefetch_start_dt = start_dt - timedelta(days=max_warmup_days)
    prefetch_end_dt = prefetch_start_dt + timedelta(days=prefetch_days - 1)
    print(
        f"[PREFETCH] Building 1m price cache for days "
        f"{prefetch_start_dt.date().isoformat()} to {prefetch_end_dt.date().isoformat()} "
        f"({max_warmup_days} warmup days + {num_days} test days + 1 day after for cross-day predictions)..."
    )
    price_cache = build_price_cache(prefetch_start_dt, prefetch_days)

    # Filter prompt configs by asset if specified
    low_freq_cfg = prompt_config.LOW_FREQUENCY
    high_freq_cfg = prompt_config.HIGH_FREQUENCY
    
    if asset_filter:
        # Filter asset lists to only include specified assets
        low_assets_filtered = [a for a in low_freq_cfg.asset_list if a in asset_filter]
        high_assets_filtered = [a for a in high_freq_cfg.asset_list if a in asset_filter]
        
        if not low_assets_filtered and not high_assets_filtered:
            print(f"[ERROR] No assets from filter {asset_filter} found in prompt configs!")
            return 1
        
        low_freq_cfg = replace(low_freq_cfg, asset_list=low_assets_filtered)
        high_freq_cfg = replace(high_freq_cfg, asset_list=high_assets_filtered)
        
        print(f"[FILTER] LOW_FREQUENCY assets: {low_assets_filtered}")
        print(f"[FILTER] HIGH_FREQUENCY assets: {high_assets_filtered}")

    # Filter prompt configs by prompt type if specified
    prompt_cfgs = []
    if args.prompt_type in ("low", "both"):
        prompt_cfgs.append(("LOW_FREQUENCY", low_freq_cfg))
    if args.prompt_type in ("high", "both"):
        prompt_cfgs.append(("HIGH_FREQUENCY", high_freq_cfg))
    
    if not prompt_cfgs:
        print(f"[ERROR] No prompt types selected!")
        return 1
    
    if args.prompt_type != "both":
        print(f"[FILTER] Testing only prompt type: {args.prompt_type.upper()}_FREQUENCY")

    # Get state update function if available (for EWMA miner warm-up)
    update_states_fn = None
    if "test_wrapper" in sys.modules:
        mod = sys.modules["test_wrapper"]
        update_states_fn = getattr(mod, "update_states_from_price_data", None)

    # Get all assets for warm-up (use filtered configs if asset filter was applied)
    low_assets = set(low_freq_cfg.asset_list)
    high_assets = set(high_freq_cfg.asset_list)
    all_assets = sorted(low_assets | high_assets)

    for day_offset in range(num_days):
        day_dt = start_dt + timedelta(days=day_offset)
        prev_day_dt = day_dt - timedelta(days=1)
        prev_day_key = prev_day_dt.date().isoformat()
        
        print(f"Running baseline offline CRPS test for day {day_dt.date().isoformat()}...")

        # Warm up state using historical data (same as real miner's warmup_states)
        # Use proper warm-up period: 72 hours for crypto, 5 days for equity
        if update_states_fn is not None:
            # Import config to get warm-up periods
            try:
                import sys
                import os
                ewma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EWMA-miner")
                if ewma_dir not in sys.path:
                    sys.path.insert(0, ewma_dir)
                import config as ewma_config
                
                print(f"  [WARM-UP] Initializing volatility state with historical data...")
                for asset in all_assets:
                    # Determine warm-up period based on asset type (same as real miner)
                    if asset in ewma_config.HF_ASSETS or asset in ewma_config.LF_CRYPTO_ASSETS:
                        warmup_hours = ewma_config.WARMUP_1M_HOURS  # 72 hours (3 days) for crypto
                    else:
                        warmup_hours = ewma_config.WARMUP_5M_DAYS * 24  # 5 days (120 hours) for equity
                    
                    # Calculate warm-up start time
                    warmup_start_dt = day_dt - timedelta(hours=warmup_hours)
                    
                    # Collect all warm-up days
                    warmup_day_list = []
                    current_warmup_dt = warmup_start_dt
                    while current_warmup_dt < day_dt:
                        warmup_day_key = current_warmup_dt.date().isoformat()
                        cache_key = (asset, warmup_day_key)
                        if cache_key in price_cache:
                            warmup_day_list.append((current_warmup_dt, price_cache[cache_key]))
                        current_warmup_dt += timedelta(days=1)
                    
                    # Process warm-up data in chronological order (same as real miner)
                    if warmup_day_list:
                        print(f"    Warming up {asset} with {warmup_hours}h ({len(warmup_day_list)} days) of historical data...")
                        for warmup_dt, warmup_data in warmup_day_list:
                            warmup_base_start = datetime.fromisoformat(warmup_data["start_time_iso"])
                            warmup_series = warmup_data["prices"]
                            try:
                                # Update state up to end of this warm-up day
                                warmup_day_end = warmup_dt + timedelta(days=1)
                                update_states_fn(
                                    asset=asset,
                                    prices=warmup_series,
                                    base_start=warmup_base_start,
                                    target_time=min(warmup_day_end, day_dt),  # Don't exceed test day start
                                )
                            except Exception as e:
                                print(f"      Warning: Failed to warm up {asset} for {warmup_dt.date()}: {e}")
                    else:
                        print(f"    Warning: No warm-up data for {asset} (need {warmup_hours}h before {day_dt.date()})")
            except ImportError:
                # Fallback to simple 1-day warm-up if EWMA config not available
                print(f"  [WARM-UP] Initializing volatility state from {prev_day_key} data (fallback: 1 day)...")
                for asset in all_assets:
                    cache_key = (asset, prev_day_key)
                    if cache_key in price_cache:
                        prev_day_data = price_cache[cache_key]
                        prev_base_start = datetime.fromisoformat(prev_day_data["start_time_iso"])
                        prev_series = prev_day_data["prices"]
                        try:
                            update_states_fn(
                                asset=asset,
                                prices=prev_series,
                                base_start=prev_base_start,
                                target_time=day_dt,
                            )
                        except Exception as e:
                            print(f"    Warning: Failed to warm up state for {asset}: {e}")
                    else:
                        print(f"    Warning: No data for {asset} on {prev_day_key} for warm-up")

        for label, cfg in prompt_cfgs:
            print(f"  Prompt type: {label}")
            run_daily_baseline_crps_for_prompt(
                prompt_cfg=cfg,
                prompt_label=label,
                day_start=day_dt,
                all_prompt_results=all_prompt_results,
                price_cache=price_cache,
            )

    summaries = compute_summaries(all_prompt_results)

    # All files must live inside offline_CRPS_test folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = base_dir

    write_results(
        all_prompt_results=all_prompt_results,
        summaries=summaries,
        output_dir=output_dir,
        start_day=start_day,
        end_day=end_day,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

