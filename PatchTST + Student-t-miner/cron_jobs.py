"""
Cron job scripts for calibration and training
"""
import os
import sys
import subprocess
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_with_low_priority(cmd: list, cwd: str = None):
    """Run command with low priority (nice/ionice)"""
    try:
        # Use nice and ionice for low priority
        full_cmd = ["nice", "-n", "19", "ionice", "-c", "3", "-n", "7"] + cmd
        result = subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return False
        
        logger.info(f"Command succeeded: {' '.join(cmd)}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {' '.join(cmd)}")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def calibrate_hf(artifacts_dir: str):
    """Calibrate HF model"""
    logger.info("Running HF calibration")
    cmd = [sys.executable, "-m", "PatchTST + Student-t-miner.calibrate", "--model", "HF", "--artifacts-dir", artifacts_dir]
    return run_with_low_priority(cmd)


def calibrate_lf(artifacts_dir: str):
    """Calibrate LF models"""
    logger.info("Running LF calibration")
    
    # Calibrate both LF models
    success_crypto = run_with_low_priority([
        sys.executable, "-m", "PatchTST + Student-t-miner.calibrate",
        "--model", "LF-crypto", "--artifacts-dir", artifacts_dir
    ])
    
    success_equity = run_with_low_priority([
        sys.executable, "-m", "PatchTST + Student-t-miner.calibrate",
        "--model", "LF-equity", "--artifacts-dir", artifacts_dir
    ])
    
    return success_crypto and success_equity


def train_models(artifacts_dir: str):
    """Train all models"""
    logger.info("Running model training")
    
    models = ["HF", "LF-crypto", "LF-equity"]
    success = True
    
    for model in models:
        logger.info(f"Training {model} model")
        cmd = [
            sys.executable, "-m", "PatchTST + Student-t-miner.train",
            "--model", model, "--artifacts-dir", artifacts_dir,
            "--epochs", "50"
        ]
        if not run_with_low_priority(cmd):
            logger.warning(f"Training {model} failed")
            success = False
    
    return success


def publish_artifacts(artifacts_dir: str):
    """Atomically publish staging artifacts to current"""
    staging_dir = os.path.join(artifacts_dir, "staging")
    current_dir = os.path.join(artifacts_dir, "current")
    
    if not os.path.exists(staging_dir):
        logger.warning("No staging directory found")
        return False
    
    os.makedirs(current_dir, exist_ok=True)
    
    # Move files atomically
    for filename in os.listdir(staging_dir):
        staging_path = os.path.join(staging_dir, filename)
        current_path = os.path.join(current_dir, filename)
        
        try:
            if os.path.exists(current_path):
                os.remove(current_path)
            os.rename(staging_path, current_path)
            logger.info(f"Published {filename}")
        except Exception as e:
            logger.error(f"Error publishing {filename}: {e}")
            return False
    
    return True


def restart_miner():
    """Restart miner service (if using PM2)"""
    try:
        # Check if PM2 is available
        result = subprocess.run(
            ["pm2", "list"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Restart miner
            subprocess.run(["pm2", "restart", "duke-miner"], check=False)
            logger.info("Restarted miner via PM2")
            return True
        else:
            logger.info("PM2 not available, skipping restart")
            return True
            
    except Exception as e:
        logger.warning(f"Could not restart miner: {e}")
        return True  # Don't fail the job


def should_restart_miner(last_restart_file: str, min_interval_minutes: int = 30) -> bool:
    """Check if miner should be restarted (avoid frequent restarts)"""
    if not os.path.exists(last_restart_file):
        return True
    
    try:
        with open(last_restart_file, 'r') as f:
            last_restart = float(f.read().strip())
        
        elapsed = (time.time() - last_restart) / 60  # minutes
        return elapsed >= min_interval_minutes
        
    except Exception as e:
        logger.warning(f"Error reading last restart time: {e}")
        return True


def update_last_restart(last_restart_file: str):
    """Update last restart timestamp"""
    os.makedirs(os.path.dirname(last_restart_file), exist_ok=True)
    with open(last_restart_file, 'w') as f:
        f.write(str(time.time()))


def main():
    """Main cron job handler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cron job handler")
    parser.add_argument("--job", choices=["calibrate-hf", "calibrate-lf", "train"], required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--restart-miner", action="store_true")
    args = parser.parse_args()
    
    success = False
    
    if args.job == "calibrate-hf":
        success = calibrate_hf(args.artifacts_dir)
    elif args.job == "calibrate-lf":
        success = calibrate_lf(args.artifacts_dir)
    elif args.job == "train":
        success = train_models(args.artifacts_dir)
    
    if success:
        # Publish artifacts
        if publish_artifacts(args.artifacts_dir):
            logger.info("Artifacts published successfully")
            
            # Restart miner if requested and enough time has passed
            if args.restart_miner:
                last_restart_file = os.path.join(args.artifacts_dir, ".last_restart")
                if should_restart_miner(last_restart_file):
                    if restart_miner():
                        update_last_restart(last_restart_file)
                else:
                    logger.info("Skipping restart (too soon since last restart)")
        else:
            logger.error("Failed to publish artifacts")
            sys.exit(1)
    else:
        logger.error(f"Job {args.job} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
