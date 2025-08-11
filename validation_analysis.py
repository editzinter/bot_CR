#!/usr/bin/env python3
"""
Validation and Performance Analysis System
This module provides comprehensive analysis of the RL agent's performance
in both simulated and real game environments.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import seaborn as sns
from datetime import datetime
import time

from stable_baselines3 import PPO
import gymnasium
import clash_royale

@dataclass
class ValidationConfig:
    """Configuration for validation and analysis."""
    num_test_episodes: int = 50
    max_steps_per_episode: int = 1000
    confidence_interval: float = 0.95
    performance_threshold: float = 0.0  # Minimum acceptable performance
    
class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for RL agents.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {}
        
    def analyze_simulated_performance(self, model_path: str) -> Dict[str, Any]:
        """Analyze model performance in simulated environment."""
        print("📊 ANALYZING SIMULATED PERFORMANCE")
        print("-" * 40)
        
        try:
            # Load model
            model = PPO.load(model_path)
            print(f"✓ Model loaded: {model_path}")
            
            # Create environment
            env = gymnasium.make("clash-royale", render_mode="rgb_array")
            
            # Run test episodes
            episode_rewards = []
            episode_lengths = []
            action_distributions = []
            
            for episode in range(self.config.num_test_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_actions = []
                
                for step in range(self.config.max_steps_per_episode):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    episode_actions.append(int(action))
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                action_distributions.append(episode_actions)
                
                if (episode + 1) % 10 == 0:
                    print(f"  Completed {episode + 1}/{self.config.num_test_episodes} episodes")
            
            env.close()
            
            # Calculate statistics
            results = {
                "model_path": model_path,
                "num_episodes": len(episode_rewards),
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "min_reward": float(np.min(episode_rewards)),
                "max_reward": float(np.max(episode_rewards)),
                "mean_length": float(np.mean(episode_lengths)),
                "std_length": float(np.std(episode_lengths)),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "action_diversity": self._calculate_action_diversity(action_distributions)
            }
            
            print(f"✓ Simulated analysis complete:")
            print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Simulated analysis failed: {e}")
            return {}
    
    def analyze_real_game_performance(self, data_path: str) -> Dict[str, Any]:
        """Analyze model performance in real game environment."""
        print("🎮 ANALYZING REAL GAME PERFORMANCE")
        print("-" * 40)
        
        try:
            # Load real game data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            episode_data = data.get("episode_data", [])
            performance_metrics = data.get("performance_metrics", {})
            
            if not episode_data:
                print("❌ No real game data found")
                return {}
            
            # Extract metrics
            rewards = [ep["total_reward"] for ep in episode_data]
            lengths = [ep["steps"] for ep in episode_data]
            successes = [ep["success"] for ep in episode_data]
            
            # Calculate statistics
            results = {
                "data_path": data_path,
                "num_episodes": len(episode_data),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "win_rate": float(np.mean(successes)),
                "mean_length": float(np.mean(lengths)),
                "safety_stops": performance_metrics.get("safety_stops", 0),
                "episode_rewards": rewards,
                "episode_lengths": lengths,
                "success_rate": float(np.mean(successes))
            }
            
            print(f"✓ Real game analysis complete:")
            print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Win rate: {results['win_rate']:.1%}")
            print(f"  Safety stops: {results['safety_stops']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Real game analysis failed: {e}")
            return {}
    
    def compare_performance(self, sim_results: Dict, real_results: Dict) -> Dict[str, Any]:
        """Compare simulated vs real game performance."""
        print("⚖️  COMPARING PERFORMANCE")
        print("-" * 40)
        
        if not sim_results or not real_results:
            print("❌ Insufficient data for comparison")
            return {}
        
        comparison = {
            "reward_transfer": {
                "simulated_mean": sim_results["mean_reward"],
                "real_mean": real_results["mean_reward"],
                "transfer_ratio": real_results["mean_reward"] / sim_results["mean_reward"] if sim_results["mean_reward"] != 0 else 0,
                "performance_drop": sim_results["mean_reward"] - real_results["mean_reward"]
            },
            "episode_length": {
                "simulated_mean": sim_results["mean_length"],
                "real_mean": real_results["mean_length"],
                "length_ratio": real_results["mean_length"] / sim_results["mean_length"] if sim_results["mean_length"] != 0 else 0
            },
            "stability": {
                "simulated_std": sim_results["std_reward"],
                "real_std": real_results["std_reward"],
                "stability_change": real_results["std_reward"] - sim_results["std_reward"]
            }
        }
        
        print(f"✓ Performance comparison:")
        print(f"  Transfer ratio: {comparison['reward_transfer']['transfer_ratio']:.2f}")
        print(f"  Performance drop: {comparison['reward_transfer']['performance_drop']:.2f}")
        print(f"  Real game win rate: {real_results.get('win_rate', 0):.1%}")
        
        return comparison
    
    def _calculate_action_diversity(self, action_distributions: List[List[int]]) -> Dict[str, float]:
        """Calculate action diversity metrics."""
        all_actions = []
        for episode_actions in action_distributions:
            all_actions.extend(episode_actions)
        
        if not all_actions:
            return {"entropy": 0.0, "unique_actions": 0, "most_common_frequency": 0.0}
        
        # Calculate entropy
        action_counts = np.bincount(all_actions, minlength=2304)
        action_probs = action_counts / len(all_actions)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        
        # Other metrics
        unique_actions = len(np.unique(all_actions))
        most_common_frequency = np.max(action_counts) / len(all_actions)
        
        return {
            "entropy": float(entropy),
            "unique_actions": int(unique_actions),
            "most_common_frequency": float(most_common_frequency)
        }
    
    def generate_visualizations(self, sim_results: Dict, real_results: Dict, output_dir: str = "analysis_plots"):
        """Generate comprehensive visualizations."""
        print("📈 GENERATING VISUALIZATIONS")
        print("-" * 40)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Reward comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Reward distributions
        if sim_results and real_results:
            axes[0, 0].hist(sim_results["episode_rewards"], alpha=0.7, label="Simulated", bins=20)
            axes[0, 0].hist(real_results["episode_rewards"], alpha=0.7, label="Real Game", bins=20)
            axes[0, 0].set_xlabel("Episode Reward")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Reward Distribution Comparison")
            axes[0, 0].legend()
            
            # Episode length comparison
            axes[0, 1].hist(sim_results["episode_lengths"], alpha=0.7, label="Simulated", bins=20)
            axes[0, 1].hist(real_results["episode_lengths"], alpha=0.7, label="Real Game", bins=20)
            axes[0, 1].set_xlabel("Episode Length")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Episode Length Comparison")
            axes[0, 1].legend()
            
            # Performance over time (real game)
            episodes = range(1, len(real_results["episode_rewards"]) + 1)
            axes[1, 0].plot(episodes, real_results["episode_rewards"], alpha=0.7)
            axes[1, 0].plot(episodes, np.convolve(real_results["episode_rewards"], 
                                                np.ones(5)/5, mode='same'), 'r-', linewidth=2)
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Reward")
            axes[1, 0].set_title("Real Game Performance Over Time")
            
            # Summary statistics
            stats_text = f"""
            Simulated Performance:
            Mean Reward: {sim_results['mean_reward']:.2f}
            Std Reward: {sim_results['std_reward']:.2f}
            
            Real Game Performance:
            Mean Reward: {real_results['mean_reward']:.2f}
            Win Rate: {real_results.get('win_rate', 0):.1%}
            Safety Stops: {real_results.get('safety_stops', 0)}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='center')
            axes[1, 1].set_title("Performance Summary")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to: {output_dir}/")
    
    def generate_report(self, sim_results: Dict, real_results: Dict, comparison: Dict) -> str:
        """Generate comprehensive analysis report."""
        report = f"""
# Clash Royale RL Agent Performance Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report analyzes the performance of the Clash Royale RL agent in both simulated and real game environments.

## Simulated Environment Performance
- **Episodes Tested**: {sim_results.get('num_episodes', 'N/A')}
- **Mean Reward**: {sim_results.get('mean_reward', 0):.2f} ± {sim_results.get('std_reward', 0):.2f}
- **Mean Episode Length**: {sim_results.get('mean_length', 0):.1f} steps
- **Action Diversity**: {sim_results.get('action_diversity', {}).get('unique_actions', 0)} unique actions used

## Real Game Environment Performance
- **Episodes Completed**: {real_results.get('num_episodes', 'N/A')}
- **Mean Reward**: {real_results.get('mean_reward', 0):.2f} ± {real_results.get('std_reward', 0):.2f}
- **Win Rate**: {real_results.get('win_rate', 0):.1%}
- **Safety Stops**: {real_results.get('safety_stops', 0)}
- **Mean Episode Length**: {real_results.get('mean_length', 0):.1f} steps

## Transfer Learning Analysis
- **Performance Transfer Ratio**: {comparison.get('reward_transfer', {}).get('transfer_ratio', 0):.2f}
- **Performance Drop**: {comparison.get('reward_transfer', {}).get('performance_drop', 0):.2f}
- **Stability Change**: {comparison.get('stability', {}).get('stability_change', 0):.2f}

## Recommendations
1. **Model Performance**: {'Good' if real_results.get('win_rate', 0) > 0.5 else 'Needs Improvement'}
2. **Safety**: {'Acceptable' if real_results.get('safety_stops', 0) < 5 else 'Concerning'}
3. **Transfer Quality**: {'Good' if comparison.get('reward_transfer', {}).get('transfer_ratio', 0) > 0.5 else 'Poor'}

## Next Steps
- Continue fine-tuning if performance is below threshold
- Improve safety mechanisms if too many safety stops
- Enhance reward shaping for better real-world adaptation
"""
        
        return report

def run_comprehensive_analysis():
    """Run comprehensive performance analysis."""
    print("🔍 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    config = ValidationConfig()
    analyzer = PerformanceAnalyzer(config)
    
    # Find available models and data
    model_paths = [
        "extended_models/ppo_clash_royale_5M_final",
        "models/ppo_clash_royale_final",
        "test_models/quick_test_ppo"
    ]
    
    real_data_paths = [
        "real_game_models/fine_tuning_data.json"
    ]
    
    # Find best available model
    model_path = None
    for path in model_paths:
        if os.path.exists(path + ".zip"):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found for analysis")
        return
    
    # Analyze simulated performance
    sim_results = analyzer.analyze_simulated_performance(model_path)
    
    # Analyze real game performance if data exists
    real_results = {}
    for data_path in real_data_paths:
        if os.path.exists(data_path):
            real_results = analyzer.analyze_real_game_performance(data_path)
            break
    
    # Compare performance
    comparison = {}
    if sim_results and real_results:
        comparison = analyzer.compare_performance(sim_results, real_results)
    
    # Generate visualizations
    if sim_results or real_results:
        analyzer.generate_visualizations(sim_results, real_results)
    
    # Generate report
    if sim_results:
        report = analyzer.generate_report(sim_results, real_results, comparison)
        
        # Save report
        with open("performance_analysis_report.md", 'w') as f:
            f.write(report)
        
        print("✓ Analysis complete!")
        print("  - Visualizations: analysis_plots/")
        print("  - Report: performance_analysis_report.md")
    
    return {
        "simulated": sim_results,
        "real_game": real_results,
        "comparison": comparison
    }

if __name__ == "__main__":
    run_comprehensive_analysis()
