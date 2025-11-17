"""Clean logging for training"""

import sys
from typing import Dict


class Logger:
    """
    Minimal logger for MARL training.
    
    Design: One-line logs, essential metrics only
    """
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
    
    def info(self, message: str):
        """General info message"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def log_train(self, metrics: Dict):
        """One-line training log"""
        log_str = (
            f"Ep {metrics['episode']:4d} | "
            f"R: {metrics['reward']:6.1f} | "
            f"R̄10: {metrics['avg_reward_10']:6.1f} | "
            f"L: {metrics['length']:3d} | "
            f"Loss: {metrics['loss']:6.4f} | "
            f"Buf: {metrics['buffer_size']:4d} | "
            f"T: {metrics['time_min']:5.1f}m"
        )
        print(log_str)
    
    def log_eval(self, episode: int, eval_reward: float, 
                 eval_length: float, best_reward: float):
        """One-line evaluation log"""
        print(f"\n{'='*70}")
        print(f"EVAL @ Ep {episode:4d} | "
              f"R: {eval_reward:6.1f} | "
              f"L: {eval_length:5.1f} | "
              f"Best: {best_reward:6.1f}")
        print(f"{'='*70}\n")
    
    def log_summary(self, summary: Dict):
        """Final training summary"""
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Time:         {summary['total_time_min']:6.1f} min")
        print(f"Final reward: {summary['final_avg_reward']:6.1f}")
        print(f"Best reward:  {summary['best_eval_reward']:6.1f}")
        print(f"Episodes:     {summary['total_episodes']:6d}")
        print(f"{'='*70}\n")