"""
Abstract base class for all MARL agents.

All algorithms must implement this interface to work with the training system.
This enables fair comparison between different MARL algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base for multi-agent RL algorithms.
    
    Design principles:
    - Algorithm-agnostic interface
    - Works with any MARL algorithm 
    - Enables fair comparison between algorithms
    
    Required methods:
    - select_actions: Get actions for all agents
    - train_step: Single training update
    - init_hidden_states: Initialize recurrent states
    - save/load: Model checkpointing
    
    Required properties:
    - replay_buffer: Access to experience storage
    """
    
    @abstractmethod
    def select_actions(
        self,
        observations: List[np.ndarray],
        last_actions: List[np.ndarray],
        hiddens: List[torch.Tensor],
        explore: bool = True,
        noise_scale: float = 0.1
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Select actions for all agents (single environment).

        Args:
            observations: List of observations per agent
            last_actions: List of previous actions per agent
            hiddens: List of recurrent hidden states (if applicable)
            explore: Whether to add exploration noise
            noise_scale: Scale of exploration noise

        Returns:
            actions: List of selected actions per agent
            new_hiddens: Updated hidden states
        """
        pass

    def select_actions_batched(
        self,
        observations_batch: List[List[np.ndarray]],
        last_actions_batch: List[List[np.ndarray]],
        hiddens_batch: List[List[torch.Tensor]],
        explore: bool = True,
        noise_scale: float = 0.1
    ) -> Tuple[List[List[np.ndarray]], List[List[torch.Tensor]]]:
        """
        Select actions for all agents across multiple environments (batched).

        This is a performance optimization that processes all environments
        in parallel on the GPU instead of sequentially.

        Args:
            observations_batch: [n_envs][n_agents] observations
            last_actions_batch: [n_envs][n_agents] last actions
            hiddens_batch: [n_envs][n_agents] hidden states
            explore: Whether to add exploration noise
            noise_scale: Scale of exploration noise

        Returns:
            actions_batch: [n_envs][n_agents] actions
            new_hiddens_batch: [n_envs][n_agents] new hidden states

        Default implementation calls select_actions sequentially.
        Override for GPU-batched implementation.
        """
        # Default: sequential processing (override for batched)
        actions_batch = []
        new_hiddens_batch = []

        for env_idx in range(len(observations_batch)):
            actions, hiddens = self.select_actions(
                observations_batch[env_idx],
                last_actions_batch[env_idx],
                hiddens_batch[env_idx],
                explore=explore,
                noise_scale=noise_scale
            )
            actions_batch.append(actions)
            new_hiddens_batch.append(hiddens)

        return actions_batch, new_hiddens_batch
    
    @abstractmethod
    def train_step(self, batch_size: int) -> Optional[float]:
        """
        Perform single training step.
        
        Args:
            batch_size: Number of episodes/transitions to sample
        
        Returns:
            loss: Training loss (for monitoring), None if not enough data
        """
        pass
    
    @abstractmethod
    def init_hidden_states(self) -> List[torch.Tensor]:
        """
        Initialize recurrent hidden states for all agents.
        
        Returns:
            List of initialized hidden states (one per agent)
        """
        pass
    
    def store_episode(self, episode_data: Dict) -> None:
        """
        Store episode in replay buffer.
        
        Default implementation uses self.replay_buffer.push().
        Override if custom storage logic is needed.
        
        Args:
            episode_data: Dictionary containing episode information
                {
                    'observations': [[obs_t0, obs_t1, ...], ...],  # Per agent
                    'actions': [[act_t0, act_t1, ...], ...],       # Per agent
                    'last_actions': [[last_act_t0, ...], ...],     # Per agent
                    'states': [state_t0, state_t1, ...],           # Global
                    'rewards': [reward_t0, reward_t1, ...]         # Shared
                }
        """
        self.replay_buffer.push(episode_data)
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: File path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: File path to load checkpoint from
        """
        pass
    
    @property
    @abstractmethod
    def replay_buffer(self):
        """
        Access to replay buffer.
        
        Returns:
            ReplayBuffer instance or None if agent doesn't use replay buffer
            
        Note:
            On-policy algorithms (like MAPPO) may return None.
            Off-policy algorithms (like QMIX, MADDPG) should return a buffer.
        """
        pass


