module.exports = {
    apps: [
      {
        name: "calibrate-high",
        interpreter: "python3",
        script: "./EWMA-miner/calibrate_half_life.py",
        args: "--start-day 2026-01-08 --num-days 7 --assets BTC,ETH,SOL,XAU --prompt-type high --grid-minutes 15,30,60,120,240",
        env: {
          PYTHONPATH: ".",
        },
      },
    ],
  };