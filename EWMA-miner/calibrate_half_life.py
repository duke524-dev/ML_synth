#!/usr/bin/env python3
"""
Grid-search EWMA half-life (i.e., lambda decay) per asset to minimize offline CRPS.

This script reuses the existing offline CRPS harness in `offline_CRPS_test/offline_crps_simple.py`,
but patches `synth.miner.simulations.generate_simulations` to use the EWMA miner via `test_wrapper`.

Notes
-----
- In EWMA-miner, decay is controlled by half-life values in `config.py`:
  lambda = exp(-ln(2) * dt / half_life_seconds)
  where dt is 60s (high frequency) or 300s (low frequency).
- HIGH_FREQUENCY prompts (dt=60) mainly depend on `HALF_LIFE_1M[asset]`
- LOW_FREQUENCY prompts (dt=300) mainly depend on `HALF_LIFE_5M[asset]`
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from typing import Iterable
from datetime import timedelta


def _parse_grid_minutes(grid: str) -> list[int]:
    vals: list[int] = []
    for part in grid.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("empty grid")
    return vals


def _objective_from_results(all_prompt_results: list[dict]) -> tuple[float, int]:
    total = 0.0
    n = 0
    for r in all_prompt_results:
        v = r.get("overall_crps")
        if v is None:
            continue
        total += float(v)
        n += 1
    return total, n


def _run_for_asset(
    *,
    asset: str,
    prompt_cfg,
    prompt_label: str,
    start_dt,
    num_days: int,
    price_cache,
) -> tuple[float, int]:
    # Import locally so the patching in main() is already active
    from offline_CRPS_test.offline_crps_simple import run_daily_baseline_crps_for_prompt
    from datetime import datetime, timezone
    import sys

    # Keep everything identical to validator config but restrict to one asset
    cfg_one = replace(prompt_cfg, asset_list=[asset])

    # Get state update function for warm-up (same as main() does)
    update_states_fn = None
    if "test_wrapper" in sys.modules:
        mod = sys.modules["test_wrapper"]
        update_states_fn = getattr(mod, "update_states_from_price_data", None)

    # Determine warm-up period based on asset type (same as real miner)
    import config as ewma_config
    
    # Check if asset is crypto (uses 72h warmup) or equity (uses 5 days warmup)
    if asset in ewma_config.HF_ASSETS or asset in ewma_config.LF_CRYPTO_ASSETS:
        warmup_hours = ewma_config.WARMUP_1M_HOURS  # 72 hours (3 days) for crypto
    else:
        warmup_hours = ewma_config.WARMUP_5M_DAYS * 24  # 5 days (120 hours) for equity

    all_prompt_results: list[dict] = []
    for day_offset in range(num_days):
        day_dt = start_dt + timedelta(days=day_offset)
        
        # Warm up state using historical data (same as real miner's warmup_states)
        # Use warmup_hours of historical data, not just one day
        if update_states_fn is not None and day_offset == 0:
            # Only warm up once at the start (for first day)
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
                print(f"[calibrate] Warming up {asset} with {warmup_hours}h ({len(warmup_day_list)} days) of historical data...")
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
                        # Silently continue if warm-up fails
                        pass
        
        # For subsequent days, warm up with previous day (incremental)
        if update_states_fn is not None and day_offset > 0:
            prev_day_dt = day_dt - timedelta(days=1)
            prev_day_key = prev_day_dt.date().isoformat()
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
                        target_time=day_dt,  # End of previous day = start of current day
                    )
                except Exception as e:
                    pass
        
        run_daily_baseline_crps_for_prompt(
            prompt_cfg=cfg_one,
            prompt_label=prompt_label,
            day_start=day_dt,
            all_prompt_results=all_prompt_results,
            price_cache=price_cache,
        )

    return _objective_from_results(all_prompt_results)


def _iter_assets(which: str) -> Iterable[str]:
    if which.lower() == "all":
        import config as ewma_config

        yield from list(ewma_config.TOKEN_MAP.keys())
        return
    for a in which.split(","):
        a = a.strip()
        if a:
            yield a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate EWMA half-life per asset (offline CRPS).")
    p.add_argument("--start-day", type=str, required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--num-days", type=int, required=True, help="Number of days to test")
    p.add_argument(
        "--assets",
        type=str,
        default="BTC,ETH,SOL,XAU,SPYX,NVDAX,TSLAX,AAPLX,GOOGLX",
        help='Comma list (or "all")',
    )
    p.add_argument(
        "--prompt-type",
        choices=["high", "low", "both"],
        default="both",
        help="Which prompt(s) to optimize against",
    )
    p.add_argument(
        "--grid-minutes",
        type=str,
        default="15,30,60,120,240,480,720,1440",
        help="Comma-separated half-life candidates in minutes",
    )
    p.add_argument(
        "--state-root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_calib"),
        help="Directory to store per-trial state (kept for debugging/reuse)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure project root + EWMA-miner are importable
    ewma_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(ewma_dir)
    sys.path.insert(0, ewma_dir)
    sys.path.insert(0, project_root)

    print(
        "[calibrate] start "
        f"start_day={args.start_day} num_days={args.num_days} "
        f"prompt_type={args.prompt_type} assets={args.assets} "
        f"grid_minutes={args.grid_minutes} state_root={args.state_root}",
        flush=True,
    )

    # Patch synth.miner.simulations to use EWMA miner
    import test_wrapper
    from test_wrapper import generate_simulations, get_asset_price

    import synth.miner.simulations as sim_mod

    sim_mod.generate_simulations = generate_simulations
    sim_mod.get_asset_price = get_asset_price

    # Load offline CRPS helpers
    from datetime import datetime, timezone

    from offline_CRPS_test.offline_crps_simple import build_price_cache
    from synth.validator import prompt_config
    import config as ewma_config

    grid_minutes = _parse_grid_minutes(args.grid_minutes)
    os.makedirs(args.state_root, exist_ok=True)

    start_dt = datetime.fromisoformat(args.start_day).replace(tzinfo=timezone.utc)
    
    # Calculate how many days of warm-up data we need (max across all assets)
    import config as ewma_config
    max_warmup_days = max(ewma_config.WARMUP_1M_HOURS // 24, ewma_config.WARMUP_5M_DAYS)
    # Fetch: warmup days + num_days + 1 day after (for cross-day predictions)
    prefetch_days = max_warmup_days + args.num_days + 1
    prefetch_start_dt = start_dt - timedelta(days=max_warmup_days)
    
    print(f"[calibrate] Fetching price cache: {max_warmup_days} warmup days + {args.num_days} test days + 1 day after")
    price_cache = build_price_cache(prefetch_start_dt, prefetch_days)

    def eval_trial(asset: str, half_life_seconds: int, prompt_kind: str) -> tuple[float, int]:
        dt = 60 if prompt_kind == "high" else 300
        lambda_val = math.exp(-math.log(2) * dt / float(half_life_seconds))

        # Apply candidate by mutating the shared dict in-place
        if prompt_kind == "high":
            ewma_config.HALF_LIFE_1M[asset] = half_life_seconds
        elif prompt_kind == "low":
            ewma_config.HALF_LIFE_5M[asset] = half_life_seconds
        else:
            raise ValueError(prompt_kind)

        # Use isolated persisted state per (asset, prompt_kind, candidate)
        trial_state_dir = os.path.join(
            args.state_root,
            f"{asset}_{prompt_kind}_hl_{half_life_seconds}",
        )
        os.makedirs(trial_state_dir, exist_ok=True)
        test_wrapper.set_state_dir(trial_state_dir, reset=True)

        t0 = time.monotonic()
        print(
            f"[calibrate] trial_start asset={asset} kind={prompt_kind} "
            f"half_life_s={half_life_seconds} dt={dt} lambda={lambda_val:.8f} "
            f"state_dir={trial_state_dir}",
            flush=True,
        )

        if prompt_kind == "high":
            total, n = _run_for_asset(
                asset=asset,
                prompt_cfg=prompt_config.HIGH_FREQUENCY,
                prompt_label="HIGH_FREQUENCY",
                start_dt=start_dt,
                num_days=args.num_days,
                price_cache=price_cache,
            )
        else:
            total, n = _run_for_asset(
                asset=asset,
                prompt_cfg=prompt_config.LOW_FREQUENCY,
                prompt_label="LOW_FREQUENCY",
                start_dt=start_dt,
                num_days=args.num_days,
                price_cache=price_cache,
            )

        dt_s = time.monotonic() - t0
        avg = (total / n) if n else float("nan")
        print(
            f"[calibrate] trial_done  asset={asset} kind={prompt_kind} "
            f"half_life_s={half_life_seconds} lambda={lambda_val:.8f} "
            f"avg_crps={avg:.6f} n={n} elapsed_s={dt_s:.1f}",
            flush=True,
        )
        return total, n

    assets = list(_iter_assets(args.assets))
    print(f"[calibrate] assets_resolved n_assets={len(assets)} assets={assets}", flush=True)

    # Calculate total number of trials for progress tracking
    num_prompt_types = 2 if args.prompt_type == "both" else 1
    total_trials = len(assets) * num_prompt_types * len(grid_minutes)
    current_trial = 0
    
    print(f"[calibrate] Total trials to run: {total_trials} (={len(assets)} assets × {num_prompt_types} prompt_types × {len(grid_minutes)} candidates)", flush=True)

    # Collect results for file output
    results_summary = {
        "start_day": args.start_day,
        "num_days": args.num_days,
        "prompt_type": args.prompt_type,
        "grid_minutes": grid_minutes,
        "best_1m": {},  # HALF_LIFE_1M results
        "best_5m": {},  # HALF_LIFE_5M results
    }

    # Print results in a config-friendly way (copy/paste into EWMA-miner/config.py)
    for asset_idx, asset in enumerate(assets, start=1):
        print(f"[calibrate] asset_start {asset_idx}/{len(assets)} asset={asset}", flush=True)
        if args.prompt_type in ("high", "both"):
            best = None
            for m_idx, m in enumerate(grid_minutes, start=1):
                current_trial += 1
                progress_pct = (current_trial / total_trials) * 100
                print(
                    f"[calibrate] progress [{progress_pct:5.1f}%] asset={asset} kind=high "
                    f"candidate={m_idx}/{len(grid_minutes)} half_life_min={m} "
                    f"({current_trial}/{total_trials} trials)",
                    flush=True,
                )
                hl = m * 60
                total, n = eval_trial(asset, hl, "high")
                if n == 0:
                    continue
                avg = total / n
                if best is None or avg < best["avg"]:
                    best = {"hl": hl, "avg": avg, "n": n}
                print(f"[high] asset={asset} half_life_min={m:>5} avg_crps={avg:.6f} n={n}")
            if best is not None:
                results_summary["best_1m"][asset] = best
                print(
                    f"[high] BEST asset={asset} -> HALF_LIFE_1M['{asset}'] = {best['hl']}  "
                    f"(avg_crps={best['avg']:.6f}, n={best['n']})"
                )

        if args.prompt_type in ("low", "both"):
            best = None
            for m_idx, m in enumerate(grid_minutes, start=1):
                current_trial += 1
                progress_pct = (current_trial / total_trials) * 100
                print(
                    f"[calibrate] progress [{progress_pct:5.1f}%] asset={asset} kind=low  "
                    f"candidate={m_idx}/{len(grid_minutes)} half_life_min={m} "
                    f"({current_trial}/{total_trials} trials)",
                    flush=True,
                )
                hl = m * 60
                total, n = eval_trial(asset, hl, "low")
                if n == 0:
                    continue
                avg = total / n
                if best is None or avg < best["avg"]:
                    best = {"hl": hl, "avg": avg, "n": n}
                print(f"[low ] asset={asset} half_life_min={m:>5} avg_crps={avg:.6f} n={n}")
            if best is not None:
                results_summary["best_5m"][asset] = best
                print(
                    f"[low ] BEST asset={asset} -> HALF_LIFE_5M['{asset}'] = {best['hl']}  "
                    f"(avg_crps={best['avg']:.6f}, n={best['n']})"
                )

        print(f"[calibrate] asset_done  {asset_idx}/{len(assets)} asset={asset}", flush=True)

    # Save results to file
    from datetime import datetime as dt
    timestamp = dt.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        ewma_dir,
        f"calibration_results_{args.start_day.replace('-', '')}_{args.prompt_type}_{timestamp}.txt"
    )
    
    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EWMA HALF-LIFE CALIBRATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Calibration Date: {dt.utcnow().isoformat()}Z\n")
        f.write(f"Test Period: {args.start_day} ({args.num_days} day(s))\n")
        f.write(f"Prompt Type: {args.prompt_type}\n")
        f.write(f"Grid (minutes): {', '.join(map(str, grid_minutes))}\n")
        f.write(f"Assets: {', '.join(assets)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Summary of best values
        f.write("BEST HALF-LIFE VALUES (in seconds)\n")
        f.write("-" * 80 + "\n\n")
        
        if results_summary["best_1m"]:
            f.write("HALF_LIFE_1M (for HIGH_FREQUENCY prompts, dt=60s):\n")
            f.write("-" * 80 + "\n")
            for asset in sorted(results_summary["best_1m"].keys()):
                best = results_summary["best_1m"][asset]
                f.write(f"  '{asset}': {best['hl']},  # avg_crps={best['avg']:.6f}, n={best['n']}\n")
            f.write("\n")
        
        if results_summary["best_5m"]:
            f.write("HALF_LIFE_5M (for LOW_FREQUENCY prompts, dt=300s):\n")
            f.write("-" * 80 + "\n")
            for asset in sorted(results_summary["best_5m"].keys()):
                best = results_summary["best_5m"][asset]
                f.write(f"  '{asset}': {best['hl']},  # avg_crps={best['avg']:.6f}, n={best['n']}\n")
            f.write("\n")
        
        # Config-ready format
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFIG.PY READY FORMAT (copy/paste into EWMA-miner/config.py)\n")
        f.write("=" * 80 + "\n\n")
        
        if results_summary["best_1m"]:
            f.write("# EWMA half-lives (in seconds, converted to lambda per dt)\n")
            f.write("HALF_LIFE_1M = {\n")
            for asset in sorted(results_summary["best_1m"].keys()):
                best = results_summary["best_1m"][asset]
                f.write(f"    \"{asset}\": {best['hl']},  # {best['hl']//60} minutes, "
                       f"lambda={math.exp(-math.log(2)*60/best['hl']):.6f}, "
                       f"avg_crps={best['avg']:.6f}\n")
            f.write("}\n\n")
        
        if results_summary["best_5m"]:
            f.write("HALF_LIFE_5M = {\n")
            for asset in sorted(results_summary["best_5m"].keys()):
                best = results_summary["best_5m"][asset]
                f.write(f"    \"{asset}\": {best['hl']},  # {best['hl']//3600} hours, "
                       f"lambda={math.exp(-math.log(2)*300/best['hl']):.6f}, "
                       f"avg_crps={best['avg']:.6f}\n")
            f.write("}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Results saved to: {results_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n[calibrate] Results saved to: {results_file}", flush=True)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

