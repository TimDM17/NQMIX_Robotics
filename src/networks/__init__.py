"""
Neural network architectures for MARL algorithms.

Networks:
- AgentNetwork: Per-agent actor-critic with GRU
- MixingNetwork: Combines individual Q-values
"""

from src.networks.agent_network import AgentNetwork
from src.networks.mixer_network import MixingNetwork

__all__ = ['AgentNetwork', 'MixingNetwork']