module.exports = {
    apps: [
      {
        name: "calibrate-nu",
        interpreter: "python3",
        script: "./EWMA-miner/calibrate_nu.py",
        args: "--start-day 2026-01-08 --num-days 7 --assets BTC,ETH,SOL,XAU --prompt-type both --grid-nu 2.5,3.0,3.5,4.0,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0",
        env: {
          PYTHONPATH: ".",
        },
      },
    ],
  };