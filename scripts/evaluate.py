"""
Standalone evaluation script for trained agents.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/nqmix_best.pth --config configs/nqmix_humanoid.yaml
    python scripts/evaluate.py --checkpoint results/checkpoints/nqmix_best.pth --config configs/nqmix_humanoid.yaml --render --n_episodes 5
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src import MaMuJoCoWrapper
from src import Evaluator
from src import Logger


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate trained MARL agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=999,
                       help='Random seed for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render environment (if supported)')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save evaluation results (optional)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds
    set_seeds(args.seed)

    # Initialize environment
    env = MaMuJoCoWrapper(config['env_name'], config['partitioning'])

    # Create agent
    agent = create_agent(config, env)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(str(checkpoint_path))
    print(f"Checkpoint loaded successfully!")
    print(f"{'='*70}\n")

    # Create evaluator
    evaluator = Evaluator(agent, env, config)

    # Run evaluation
    print(f"Evaluating for {args.n_episodes} episodes...")
    print(f"Environment: {config['env_name']}")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Seed: {args.seed}\n")

    if args.render:
        # Evaluate with rendering (slower)
        avg_reward, avg_length = evaluate_with_render(
            agent, env, config, args.n_episodes, args.seed
        )
    else:
        # Fast evaluation without rendering
        avg_reward, avg_length = evaluator.evaluate(
            n_episodes=args.n_episodes,
            seed=args.seed
        )

    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Number of episodes: {args.n_episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"{'='*70}\n")

    # Save results if requested
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'checkpoint': str(checkpoint_path),
            'config': args.config,
            'algorithm': config['algorithm'],
            'environment': config['env_name'],
            'n_episodes': args.n_episodes,
            'seed': args.seed,
            'avg_reward': float(avg_reward),
            'avg_length': float(avg_length)
        }

        with open(save_path, 'w') as f:
            yaml.dump(results, f)

        print(f"Results saved to: {save_path}\n")


def evaluate_with_render(agent, env, config, n_episodes, seed):
    """
    Evaluate with rendering enabled (if environment supports it)
    Note: MaMuJoCo rendering may require additional setup
    """
    print("Note: Rendering may not be available for MaMuJoCo environments")
    print("Falling back to evaluation without rendering...\n")

    # Use standard evaluator (rendering support can be added later)
    evaluator = Evaluator(agent, env, config)
    return evaluator.evaluate(n_episodes=n_episodes, seed=seed)


def create_agent(config, env):
    """Factory function to create agent based on config"""
    if config['algorithm'] == 'nqmix':
        from src import NQMIX
        return NQMIX(
            n_agents=env.n_agents,
            obs_dims=env.obs_dims,
            action_dims=env.action_dims,
            state_dim=env.state_dim,
            **config['agent_params']
        )
    elif config['algorithm'] == 'facmac':
        from src import FACMAC
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
