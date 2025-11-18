"""
Script to plot training results and create comparison visualizations.

Usage:
    # Plot single training run
    python scripts/plot_results.py --log results/checkpoints/train.log --output results/plots/

    # Compare multiple algorithms
    python scripts/plot_results.py --logs results/nqmix/train.log results/facmac/train.log --labels NQMIX FACMAC --output results/plots/comparison.png
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style
sns.set_style("whitegrid")
sns.set_palette("husl")


def parse_log_file(log_path: str) -> Dict[str, List]:
    """
    Parse training log file to extract metrics.

    Expected log format:
    Episode 100 | Reward: 1234.56 | Avg(100): 1200.00 | Length: 500 | Loss: 0.123 | Buffer: 100 | Time: 12.3s

    Returns:
        Dictionary with lists of episodes, rewards, avg_rewards, lengths, losses
    """
    episodes = []
    rewards = []
    avg_rewards = []
    lengths = []
    losses = []

    with open(log_path, 'r') as f:
        for line in f:
            # Skip info lines
            if not line.startswith('Episode'):
                continue

            # Extract metrics using regex
            episode_match = re.search(r'Episode (\d+)', line)
            reward_match = re.search(r'Reward: ([-\d.]+)', line)
            avg_match = re.search(r'Avg\(\d+\): ([-\d.]+)', line)
            length_match = re.search(r'Length: (\d+)', line)
            loss_match = re.search(r'Loss: ([-\d.]+)', line)

            if episode_match:
                episodes.append(int(episode_match.group(1)))
            if reward_match:
                rewards.append(float(reward_match.group(1)))
            if avg_match:
                avg_rewards.append(float(avg_match.group(1)))
            if length_match:
                lengths.append(int(length_match.group(1)))
            if loss_match:
                losses.append(float(loss_match.group(1)))

    return {
        'episodes': episodes,
        'rewards': rewards,
        'avg_rewards': avg_rewards,
        'lengths': lengths,
        'losses': losses
    }


def plot_single_run(data: Dict[str, List], output_path: str, title: str = "Training Progress"):
    """Create a comprehensive plot for a single training run"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    episodes = data['episodes']

    # Plot 1: Rewards
    if data['rewards'] and data['avg_rewards']:
        axes[0, 0].plot(episodes, data['rewards'], alpha=0.3, label='Episode Reward')
        axes[0, 0].plot(episodes, data['avg_rewards'], linewidth=2, label='Average Reward (100 eps)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode Length
    if data['lengths']:
        axes[0, 1].plot(episodes, data['lengths'], color='green', alpha=0.6)
        # Add smoothed line
        if len(data['lengths']) > 10:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(data['lengths'], size=min(50, len(data['lengths'])//10))
            axes[0, 1].plot(episodes, smoothed, color='darkgreen', linewidth=2, label='Smoothed')
            axes[0, 1].legend()
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss
    if data['losses']:
        axes[1, 0].plot(episodes, data['losses'], color='red', alpha=0.6)
        # Add smoothed line
        if len(data['losses']) > 10:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(data['losses'], size=min(50, len(data['losses'])//10))
            axes[1, 0].plot(episodes, smoothed, color='darkred', linewidth=2, label='Smoothed')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Learning Curve (cumulative avg reward)
    if data['rewards']:
        cumulative_avg = np.cumsum(data['rewards']) / np.arange(1, len(data['rewards']) + 1)
        axes[1, 1].plot(episodes, cumulative_avg, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Average Reward')
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_comparison(data_list: List[Dict[str, List]], labels: List[str],
                   output_path: str, title: str = "Algorithm Comparison"):
    """Create comparison plots for multiple training runs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = sns.color_palette("husl", len(data_list))

    # Plot 1: Average Rewards Comparison
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data['episodes'] and data['avg_rewards']:
            axes[0].plot(data['episodes'], data['avg_rewards'],
                        linewidth=2, label=label, color=colors[i])

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward (100 episodes)')
    axes[0].set_title('Reward Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Loss Comparison
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if data['episodes'] and data['losses']:
            # Smooth the losses for better visualization
            from scipy.ndimage import uniform_filter1d
            if len(data['losses']) > 10:
                smoothed = uniform_filter1d(data['losses'], size=min(50, len(data['losses'])//10))
                axes[1].plot(data['episodes'], smoothed,
                           linewidth=2, label=label, color=colors[i])

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Training Loss (smoothed)')
    axes[1].set_title('Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def create_summary_table(data_list: List[Dict[str, List]], labels: List[str]):
    """Print a summary table of final performance"""
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Algorithm':<15} {'Final Avg Reward':<20} {'Best Reward':<20} {'Final Loss':<15}")
    print(f"{'-'*70}")

    for data, label in zip(data_list, labels):
        final_avg = data['avg_rewards'][-1] if data['avg_rewards'] else 0
        best_reward = max(data['avg_rewards']) if data['avg_rewards'] else 0
        final_loss = data['losses'][-1] if data['losses'] else 0

        print(f"{label:<15} {final_avg:<20.2f} {best_reward:<20.2f} {final_loss:<15.4f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--log', type=str, default=None,
                       help='Path to single log file')
    parser.add_argument('--logs', type=str, nargs='+', default=None,
                       help='Paths to multiple log files for comparison')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                       help='Labels for each log file in comparison')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for plot (file or directory)')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom title for the plot')
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    if output_path.suffix == '':
        # It's a directory
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # It's a file
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Single log file
    if args.log:
        print(f"Parsing log file: {args.log}")
        data = parse_log_file(args.log)

        if not data['episodes']:
            print("Error: No training data found in log file.")
            return

        print(f"Found {len(data['episodes'])} episodes of training data")

        # Determine output path
        if output_path.suffix == '':
            plot_path = output_path / 'training_progress.png'
        else:
            plot_path = output_path

        title = args.title or "Training Progress"
        plot_single_run(data, str(plot_path), title)

    # Multiple log files for comparison
    elif args.logs:
        print(f"Comparing {len(args.logs)} training runs...")

        data_list = []
        for log_path in args.logs:
            print(f"  Parsing: {log_path}")
            data = parse_log_file(log_path)
            data_list.append(data)

        # Use provided labels or default to log filenames
        if args.labels:
            if len(args.labels) != len(args.logs):
                print("Error: Number of labels must match number of log files")
                return
            labels = args.labels
        else:
            labels = [Path(log).stem for log in args.logs]

        # Determine output path
        if output_path.suffix == '':
            plot_path = output_path / 'comparison.png'
        else:
            plot_path = output_path

        title = args.title or "Algorithm Comparison"
        plot_comparison(data_list, labels, str(plot_path), title)

        # Print summary table
        create_summary_table(data_list, labels)

    else:
        print("Error: Must provide either --log or --logs")
        parser.print_help()


if __name__ == '__main__':
    main()
