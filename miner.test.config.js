module.exports = {
  apps: [
    {
      name: "base miner",
      interpreter: "python3",
      script: "./neurons/miner.py",
      args: "--netuid 50 --logging.debug --logging.trace --subtensor.network test --wallet.name duke524 --wallet.hotkey test_miner --axon.port 9001 --blacklist.validator.min_stake 0",
      env: {
        PYTHONPATH: ".",
      },
    },
  ],
};
