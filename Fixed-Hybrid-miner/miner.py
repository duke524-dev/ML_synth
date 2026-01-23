"""
Fixed Hybrid Miner: Main miner class
LightGBM + GARCH + EWMA + Student-t
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

# Import from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from forecast_engine import ForecastEngine

logger = logging.getLogger(__name__)


class FixedHybridMiner(BaseMinerNeuron):
    """
    Fixed Hybrid miner using LightGBM (center path) + GARCH + EWMA + Student-t
    """
    
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Initialize forecast engine
        state_dir = getattr(config, 'state_dir', 'state') if config else 'state'
        model_dir = getattr(config, 'model_dir', 'models') if config else 'models'
        self.forecast_engine = ForecastEngine(state_dir=state_dir, model_dir=model_dir)
        
        bt.logging.info("FixedHybridMiner initialized with LightGBM + GARCH + EWMA + Student-t")
    
    async def forward_miner(self, synapse: Simulation) -> Simulation:
        """
        Handle incoming simulation request
        """
        simulation_input = synapse.simulation_input
        
        bt.logging.info(
            f"[FixedHybridMiner] Request from {synapse.dendrite.hotkey} "
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
            
            if result is None:
                bt.logging.error(f"[FixedHybridMiner] Failed to generate paths for {simulation_input.asset}")
                synapse.simulation_output = None
            else:
                # Set output
                synapse.simulation_output = result
                
                bt.logging.info(
                    f"[FixedHybridMiner] Generated {simulation_input.num_simulations} paths "
                    f"for {simulation_input.asset}"
                )
            
        except Exception as e:
            bt.logging.error(f"[FixedHybridMiner] Error in forward_miner: {e}", exc_info=True)
            # Set empty output (will be scored 0)
            synapse.simulation_output = None
        
        return synapse
    
    async def blacklist(self, synapse: Simulation) -> typing.Tuple[bool, str]:
        """Blacklist logic (reuse base implementation)"""
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
            return True, "Stake below minimum"
        
        return False, ""
    
    async def priority(self, synapse: Simulation) -> float:
        """Priority function based on stake"""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return 0.0
        
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = float(self.metagraph.S[uid])
        
        return stake


if __name__ == "__main__":
    from neurons.miner import Miner as BaseMiner
    
    # Create and run miner
    miner = FixedHybridMiner()
    miner.run()
