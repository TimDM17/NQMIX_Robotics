"""
MARL agents module.

Available algorithms:
- NQMIX: Non-monotonic Q-value mixing for continuous actions
- FACMAC: Factored Multi-Agent Centralised Policy Gradients
"""


from .nqmix import NQMIX
from .facmac import FACMAC

__all__ = ['NQMIX', 'FACMAC']