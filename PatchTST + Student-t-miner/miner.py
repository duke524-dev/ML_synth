"""
Miner integration: duke_miner_1 class
"""
import time
import typing
import logging
import bittensor as bt

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from synth.base.miner import BaseMinerNeuron
from synth.protocol import Simulation

# Import from current directory (handle space in folder name)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import using importlib to handle folder name with spaces
import importlib.util
spec = importlib.util.spec_from_file_location("forecast_engine", os.path.join(current_dir, "forecast_engine.py"))
forecast_engine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_engine_module)
ForecastEngine = forecast_engine_module.ForecastEngine

logger = logging.getLogger(__name__)


class DukeMiner1(BaseMinerNeuron):
    """
    Custom miner using PatchTST + Student-t models
    """
    
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Initialize forecast engine
        artifacts_dir = getattr(config, 'artifacts_dir', 'artifacts') if config else 'artifacts'
        self.forecast_engine = ForecastEngine(artifacts_dir=artifacts_dir)
        
        bt.logging.info("DukeMiner1 initialized with PatchTST + Student-t models")
    
    async def forward_miner(self, synapse: Simulation) -> Simulation:
        """
        Handle incoming simulation request
        """
        simulation_input = synapse.simulation_input
        
        bt.logging.info(
            f"[DukeMiner1] Request from {synapse.dendrite.hotkey} "
            f"asset={simulation_input.asset} "
            f"start={simulation_input.start_time} "
            f"increment={simulation_input.time_increment} "
            f"length={simulation_input.time_length} "
            f"sims={simulation_input.num_simulations}"
        )
        
        try:
            # Generate paths using ForecastEngine
            result = self.forecast_engine.generate_paths(
                asset=simulation_input.asset,
                start_time=simulation_input.start_time,
                time_increment=simulation_input.time_increment,
                time_length=simulation_input.time_length,
                num_simulations=simulation_input.num_simulations,
            )
            
            # Set output
            synapse.simulation_output = result
            
            bt.logging.info(
                f"[DukeMiner1] Generated {simulation_input.num_simulations} paths "
                f"for {simulation_input.asset}"
            )
            
        except Exception as e:
            bt.logging.error(f"[DukeMiner1] Error in forward_miner: {e}", exc_info=True)
            # Set empty output (will be scored 0)
            synapse.simulation_output = None
        
        return synapse
    
    async def blacklist(self, synapse: Simulation) -> typing.Tuple[bool, str]:
        """
        Blacklist logic (reuse base implementation)
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received request without dendrite or hotkey")
            return True, "Missing dendrite or hotkey"
        
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"
        
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = self.metagraph.S[uid]
        
        if stake <= self.config.blacklist.validator_min_stake:
            bt.logging.info(
                f"Hotkey {synapse.dendrite.hotkey}: stake below minimum "
                f"threshold of {self.config.blacklist.validator_min_stake}"
            )
            return True, "Stake below minimum threshold"
        
        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"
        
        return False, "Hotkey recognized!"
    
    async def priority(self, synapse: Simulation) -> float:
        """
        Priority based on stake (reuse base implementation)
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority
    
    def save_state(self):
        """Save miner state (if needed)"""
        pass
    
    def load_state(self):
        """Load miner state (if needed)"""
        pass
    
    def set_weights(self):
        """Set weights (miners don't set weights)"""
        pass
    
    def forward_validator(self):
        """Not used for miners"""
        pass
    
    def print_info(self):
        """Print miner info"""
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        
        log = (
            "DukeMiner1 | "
            f"Step:{self.step} | "
            f"UID:{self.uid} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"Trust:{metagraph.T[self.uid]:.4f} | "
            f"Incentive:{metagraph.I[self.uid]:.4f} | "
            f"Emission:{metagraph.E[self.uid]:.4f}"
        )
        bt.logging.info(log)


# Main entry point
if __name__ == "__main__":
    with DukeMiner1() as miner:
        while True:
            miner.print_info()
            time.sleep(20)
