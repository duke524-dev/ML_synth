module.exports = {
    apps: [
      {
        name: "calibrate-low",
        interpreter: "python3",
        script: "./EWMA-miner/calibrate_half_life.py",
        args: "--start-day 2026-01-08 --num-days 7 --prompt-type low --grid-minutes 60,120,240,480,720,1440",
        env: {
          PYTHONPATH: ".",
        },
      },
    ],
  };