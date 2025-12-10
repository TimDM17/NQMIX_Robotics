"""
Observation normalization utilities for MARL training.

Purpose:
    Normalizes observations to zero mean and unit variance using running statistics.
    Critical for MuJoCo environments where observations have very different scales.

Key Concepts:
    - Running mean/std: Incrementally updated without storing all observations
    - Per-agent normalization: Each agent has independent statistics
    - Welford's algorithm: Numerically stable incremental variance computation

Reference:
    - SAC, TD3, PPO all use observation normalization for MuJoCo
    - facmac-main uses running mean/std normalization
"""

import numpy as np
from typing import List, Optional, Dict
import pickle


class RunningMeanStd:
    """
    Running mean and standard deviation using Welford's algorithm.

    Numerically stable incremental computation that doesn't require
    storing all observations in memory.

    Algorithm:
        For each new observation x:
            n = n + 1
            delta = x - mean
            mean = mean + delta / n
            M2 = M2 + delta * (x - mean)
            var = M2 / n
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        """
        Initialize running statistics.

        Args:
            shape: Shape of observations (e.g., (obs_dim,))
            epsilon: Small value for numerical stability (prevents div by zero)
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Start with small count to prevent div by zero
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new observation(s).

        Args:
            x: Single observation [obs_dim] or batch [batch, obs_dim]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Single obs -> batch of 1

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ) -> None:
        """
        Update statistics from batch moments (mean, var, count).

        Uses parallel algorithm for combining statistics:
            delta = batch_mean - mean
            total_count = count + batch_count
            new_mean = mean + delta * batch_count / total_count
            M2 = var * count + batch_var * batch_count + delta^2 * count * batch_count / total_count
            new_var = M2 / total_count
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # New mean: weighted average
        new_mean = self.mean + delta * batch_count / total_count

        # New M2: combine variances using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize observation to zero mean and unit variance.

        Args:
            x: Observation(s) to normalize

        Returns:
            Normalized observation(s): (x - mean) / sqrt(var + epsilon)
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Convert normalized observation back to original scale.

        Args:
            x: Normalized observation(s)

        Returns:
            Original scale observation(s): x * sqrt(var + epsilon) + mean
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean


class ObservationNormalizer:
    """
    Multi-agent observation normalizer.

    Maintains separate running statistics for each agent since they
    may have different observation dimensions and distributions.

    Usage:
        normalizer = ObservationNormalizer(obs_dims=[242, 170])

        # During training (updates statistics)
        normalized_obs = normalizer.normalize(observations, update=True)

        # During evaluation (use frozen statistics)
        normalized_obs = normalizer.normalize(observations, update=False)
    """

    def __init__(
        self,
        obs_dims: List[int],
        epsilon: float = 1e-4,
        clip_range: float = 10.0
    ):
        """
        Initialize multi-agent observation normalizer.

        Args:
            obs_dims: List of observation dimensions per agent
            epsilon: Numerical stability constant
            clip_range: Clip normalized observations to [-clip_range, clip_range]
        """
        self.n_agents = len(obs_dims)
        self.obs_dims = obs_dims
        self.clip_range = clip_range

        # Create separate running stats for each agent
        self.running_stats = [
            RunningMeanStd(shape=(dim,), epsilon=epsilon)
            for dim in obs_dims
        ]

    def normalize(
        self,
        observations: List[np.ndarray],
        update: bool = True
    ) -> List[np.ndarray]:
        """
        Normalize observations for all agents.

        Args:
            observations: List of observations per agent
            update: Whether to update running statistics (True for training)

        Returns:
            List of normalized observations per agent
        """
        normalized = []
        for i, obs in enumerate(observations):
            if update:
                self.running_stats[i].update(obs)

            norm_obs = self.running_stats[i].normalize(obs)

            # Clip to prevent extreme values
            norm_obs = np.clip(norm_obs, -self.clip_range, self.clip_range)

            normalized.append(norm_obs)

        return normalized

    def normalize_batch(
        self,
        observations_batch: List[List[np.ndarray]],
        update: bool = True
    ) -> List[List[np.ndarray]]:
        """
        Normalize observations from multiple environments.

        Args:
            observations_batch: [n_envs][n_agents] observations
            update: Whether to update running statistics

        Returns:
            [n_envs][n_agents] normalized observations
        """
        n_envs = len(observations_batch)

        # If updating, collect all observations for batch update
        if update:
            for agent_idx in range(self.n_agents):
                agent_obs = np.stack([
                    observations_batch[env][agent_idx]
                    for env in range(n_envs)
                ])
                self.running_stats[agent_idx].update(agent_obs)

        # Normalize all observations
        normalized_batch = []
        for env_idx in range(n_envs):
            normalized = []
            for agent_idx in range(self.n_agents):
                obs = observations_batch[env_idx][agent_idx]
                norm_obs = self.running_stats[agent_idx].normalize(obs)
                norm_obs = np.clip(norm_obs, -self.clip_range, self.clip_range)
                normalized.append(norm_obs)
            normalized_batch.append(normalized)

        return normalized_batch

    def save(self, path: str) -> None:
        """Save normalizer statistics to file."""
        stats = {
            'n_agents': self.n_agents,
            'obs_dims': self.obs_dims,
            'clip_range': self.clip_range,
            'running_stats': [
                {
                    'mean': rs.mean,
                    'var': rs.var,
                    'count': rs.count,
                    'epsilon': rs.epsilon
                }
                for rs in self.running_stats
            ]
        }
        with open(path, 'wb') as f:
            pickle.dump(stats, f)

    def load(self, path: str) -> None:
        """Load normalizer statistics from file."""
        with open(path, 'rb') as f:
            stats = pickle.load(f)

        assert stats['n_agents'] == self.n_agents
        assert stats['obs_dims'] == self.obs_dims

        self.clip_range = stats['clip_range']

        for i, rs_stats in enumerate(stats['running_stats']):
            self.running_stats[i].mean = rs_stats['mean']
            self.running_stats[i].var = rs_stats['var']
            self.running_stats[i].count = rs_stats['count']
            self.running_stats[i].epsilon = rs_stats['epsilon']

    def get_stats(self) -> Dict:
        """Get current statistics for logging."""
        return {
            f'agent_{i}_mean_norm': float(np.mean(np.abs(rs.mean)))
            for i, rs in enumerate(self.running_stats)
        }
