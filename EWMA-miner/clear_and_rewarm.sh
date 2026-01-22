#!/bin/bash
# Clear old state files and re-warm with corrected variance

cd "$(dirname "$0")"

echo "Clearing old state files..."
rm -f state/*.msgpack

echo "Re-warming states for 2026-01-15..."
python state_updater.py --warmup-date 2026-01-15 --one-shot --state-dir state --log-level INFO

echo "Done! States have been re-warmed with corrected variance values."
