"""
Plot training results from log files.

Purpose:
    Parses training log files and generates visualization plots for
    analyzing training progress. Creates reward curves, loss plots,
    and episode length charts.

Usage:
    python scripts/plot_results.py --log results/nqmix_humanoid/training.log
    python scripts/plot_results.py --log results/nqmix_humanoid/training.log --output results/nqmix_humanoid/plots

Output:
    - reward.png: Training rewards over episodes
    - eval_reward.png: Evaluation rewards over episodes
    - loss.png: Training loss over episodes
    - length.png: Episode lengths over episodes
    - summary.png: Combined 2x2 plot of all metrics
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_log_file(log_path: str) -> dict:
    """
    Parse training log file and extract metrics.

    The log file contains lines like:
        Training: "Ep  100 | R:   -50.3 | R̄10:   -45.2 | Len:  234 | Loss:  0.0123 | Buf:  100 | T:   2.5m"
        Evaluation: "EVAL @ Ep  100 | R:   -42.1 | Len:  245.3 | Best:   -40.5"

    We use regex patterns to extract numeric values from these formatted strings.

    Args:
        log_path: Path to the training log file

    Returns:
        Dictionary with lists of metrics for plotting
    """
    metrics = {
        'episodes': [],       # Episode numbers
        'rewards': [],        # Per-episode rewards
        'avg_rewards': [],    # Running average rewards (R̄10)
        'lengths': [],        # Episode lengths
        'losses': [],         # Training losses
        'buffer_sizes': [],   # Replay buffer sizes
        'times': [],          # Training times in minutes
        'eval_episodes': [],  # Episodes where evaluation was run
        'eval_rewards': [],   # Evaluation mean rewards
        'best_rewards': []    # Best evaluation rewards so far
    }

    # ================================================================
    # REGEX PATTERNS FOR LOG PARSING
    # ================================================================

    # Pattern for training log lines
    # Matches: "Ep  100 | R:   -50.3 | R̄10:   -45.2 | Len:  234 | Loss:  0.0123 | Buf:  100 | T:   2.5m"
    # Note: Using R10 instead of R̄10 because the macron may not be in all logs
    train_pattern = re.compile(
        r'Ep\s+(\d+)\s+\|\s+'           # Episode number
        r'R:\s+([-\d.]+)\s+\|\s+'        # Reward (can be negative)
        r'R10:\s+([-\d.]+)\s+\|\s+'      # Average reward (R̄10 or R10)
        r'Len:\s+(\d+)\s+\|\s+'          # Episode length
        r'Loss:\s+([-\d.]+)\s+\|\s+'     # Loss value
        r'Buf:\s+(\d+)\s+\|\s+'          # Buffer size
        r'T:\s+([\d.]+)m'                # Time in minutes
    )

    # Pattern for evaluation log lines
    # Matches: "EVAL @ Ep  100 | R:   -42.1 | Len:  245.3 | Best:   -40.5"
    eval_pattern = re.compile(
        r'EVAL @ Ep\s+(\d+)\s+\|\s+'     # Evaluation episode
        r'R:\s+([-\d.]+)\s+\|\s+'        # Mean evaluation reward
        r'Len:\s+([\d.]+)\s+\|\s+'       # Mean episode length
        r'Best:\s+([-\d.]+)'             # Best reward so far
    )

    # Parse file line by line
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Try to match training log
            train_match = train_pattern.search(line)
            if train_match:
                metrics['episodes'].append(int(train_match.group(1)))
                metrics['rewards'].append(float(train_match.group(2)))
                metrics['avg_rewards'].append(float(train_match.group(3)))
                metrics['lengths'].append(int(train_match.group(4)))
                metrics['losses'].append(float(train_match.group(5)))
                metrics['buffer_sizes'].append(int(train_match.group(6)))
                metrics['times'].append(float(train_match.group(7)))
                continue

            # Try to match evaluation log
            eval_match = eval_pattern.search(line)
            if eval_match:
                metrics['eval_episodes'].append(int(eval_match.group(1)))
                metrics['eval_rewards'].append(float(eval_match.group(2)))
                metrics['best_rewards'].append(float(eval_match.group(4)))

    return metrics


def smooth(data: list, window: int = 10) -> np.ndarray:
    """
    Apply moving average smoothing to noisy data.

    Moving average helps visualize trends by reducing noise:
        smoothed[i] = mean(data[i-window+1:i+1])

    Args:
        data: Raw data to smooth
        window: Number of points to average (default 10)

    Returns:
        Smoothed data (shorter by window-1 points)
    """
    if len(data) < window:
        return np.array(data)
    # np.convolve with ones/window computes moving average
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_training_curves(metrics: dict, output_dir: Path, show: bool = True):
    """
    Generate training plots from parsed metrics.

    Creates 5 plots:
        1. reward.png - Training rewards with smoothing
        2. eval_reward.png - Evaluation rewards
        3. loss.png - Training loss
        4. length.png - Episode lengths
        5. summary.png - Combined 2x2 overview

    Args:
        metrics: Dictionary with training metrics from parse_log_file()
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use seaborn style for cleaner plots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_size = (10, 6)

    # ================================================================
    # PLOT 1: Training Reward
    # ================================================================
    # Shows learning progress - reward should increase over time
    fig, ax = plt.subplots(figsize=fig_size)
    episodes = metrics['episodes']
    rewards = metrics['rewards']

    # Raw rewards (transparent to show variance)
    ax.plot(episodes, rewards, alpha=0.3, label='Episode Reward')

    # Smoothed rewards for clearer trend
    if len(rewards) >= 10:
        smoothed = smooth(rewards, 10)
        ax.plot(episodes[9:], smoothed, label='Smoothed (10 ep)')

    # Running average from training (already computed during training)
    ax.plot(episodes, metrics['avg_rewards'], label='Running Avg (10 ep)', linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'reward.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    # ================================================================
    # PLOT 2: Evaluation Reward
    # ================================================================
    # Shows true policy performance (no exploration noise)
    if metrics['eval_episodes']:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(metrics['eval_episodes'], metrics['eval_rewards'],
                marker='o', label='Eval Reward')
        ax.plot(metrics['eval_episodes'], metrics['best_rewards'],
                linestyle='--', label='Best Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Evaluation Reward')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'eval_reward.png', dpi=150)
        if show:
            plt.show()
        plt.close()

    # ================================================================
    # PLOT 3: Training Loss
    # ================================================================
    # Shows learning stability - should decrease and stabilize
    fig, ax = plt.subplots(figsize=fig_size)
    losses = metrics['losses']
    ax.plot(episodes, losses, alpha=0.5)
    if len(losses) >= 10:
        smoothed_loss = smooth(losses, 10)
        ax.plot(episodes[9:], smoothed_loss, label='Smoothed', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    # ================================================================
    # PLOT 4: Episode Length
    # ================================================================
    # Shows survival time - longer is better for Humanoid
    fig, ax = plt.subplots(figsize=fig_size)
    lengths = metrics['lengths']
    ax.plot(episodes, lengths, alpha=0.5)
    if len(lengths) >= 10:
        smoothed_len = smooth(lengths, 10)
        ax.plot(episodes[9:], smoothed_len, label='Smoothed', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Length')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'length.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    # ================================================================
    # PLOT 5: Combined Summary
    # ================================================================
    # 2x2 grid with all key metrics for quick overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Training reward
    ax = axes[0, 0]
    ax.plot(episodes, metrics['avg_rewards'], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Training Reward')

    # Top-right: Evaluation reward
    ax = axes[0, 1]
    if metrics['eval_episodes']:
        ax.plot(metrics['eval_episodes'], metrics['eval_rewards'],
                marker='o', markersize=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Evaluation Reward')

    # Bottom-left: Loss
    ax = axes[1, 0]
    if len(losses) >= 10:
        smoothed_loss = smooth(losses, 10)
        ax.plot(episodes[9:], smoothed_loss, linewidth=2)
    else:
        ax.plot(episodes, losses, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')

    # Bottom-right: Episode length
    ax = axes[1, 1]
    if len(lengths) >= 10:
        smoothed_len = smooth(lengths, 10)
        ax.plot(episodes[9:], smoothed_len, linewidth=2)
    else:
        ax.plot(episodes, lengths, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Length')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    # ================================================================
    # ARGUMENT PARSING
    # ================================================================
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to training log file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: same as log)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    args = parser.parse_args()

    # Validate log file exists
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    # ================================================================
    # PARSE LOG FILE
    # ================================================================
    print(f"Parsing log file: {log_path}")
    metrics = parse_log_file(str(log_path))

    if not metrics['episodes']:
        print("Error: No training data found in log file")
        sys.exit(1)

    print(f"Found {len(metrics['episodes'])} training entries")
    print(f"Found {len(metrics['eval_episodes'])} evaluation entries")

    # ================================================================
    # GENERATE PLOTS
    # ================================================================
    # Default output: plots/ subdirectory next to log file
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = log_path.parent / 'plots'

    plot_training_curves(metrics, output_dir, show=not args.no_show)


if __name__ == '__main__':
    main()
