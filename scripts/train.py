"""
Main training script.

Usage:
    python scripts/train.py --config configs/nqmix_humanoid.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src import MaMuJoCoWrapper
from src import NQMIX, FACMAC
from src import Trainer
from src import Logger


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set seeds
    set_seeds(config['seed'])
    
    # Create directories
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    env = MaMuJoCoWrapper(config['env_name'], config['partitioning'])
    agent = create_agent(config, env)
    logger = Logger(log_file=str(save_dir / 'train.log'))
    
    # Log configuration
    logger.info(f"Training {config['algorithm']} on {config['env_name']}")
    logger.info(f"Seed: {config['seed']}, Device: {agent.device}")
    logger.info(f"Config: {args.config}\n")
    
    # Create trainer and run
    trainer = Trainer(agent, env, config, logger, save_dir=str(save_dir))
    trainer.train()


def create_agent(config, env):
    """Factory function to create agent based on config"""
    if config['algorithm'] == 'nqmix':
        return NQMIX(
            n_agents=env.n_agents,
            obs_dims=env.obs_dims,
            action_dims=env.action_dims,
            state_dim=env.state_dim,
            **config['agent_params']
        )
    elif config['algorithm'] == 'facmac':
        return FACMAC(
            n_agents=env.n_agents,
            obs_dims=env.obs_dims,
            action_dims=env.action_dims,
            state_dim=env.state_dim,
            **config['agent_params']
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")


def set_seeds(seed):
    """Set all random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()