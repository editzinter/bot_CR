#!/usr/bin/env python3
"""
Training Monitor for Extended PPO Training
Monitors the progress of the 5M timestep training and provides real-time updates.
"""

import os
import time
import json
import subprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def check_training_status():
    """Check if training is currently running."""
    try:
        # Check if there are any python processes running extended_training.py
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        return "python.exe" in result.stdout
    except:
        return False

def get_latest_tensorboard_logs():
    """Get the latest tensorboard log directory."""
    tensorboard_dir = "extended_tensorboard"
    if not os.path.exists(tensorboard_dir):
        return None
    
    subdirs = [d for d in os.listdir(tensorboard_dir) if os.path.isdir(os.path.join(tensorboard_dir, d))]
    if not subdirs:
        return None
    
    # Get the most recent directory
    latest_dir = max(subdirs, key=lambda d: os.path.getctime(os.path.join(tensorboard_dir, d)))
    return os.path.join(tensorboard_dir, latest_dir)

def parse_training_logs():
    """Parse training logs to extract progress information."""
    log_dir = "extended_logs"
    if not os.path.exists(log_dir):
        return None
    
    # Look for monitor.csv files
    monitor_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file == "monitor.csv":
                monitor_files.append(os.path.join(root, file))
    
    if not monitor_files:
        return None
    
    # Parse the most recent monitor file
    latest_file = max(monitor_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:  # Header + at least one data line
            return None
        
        # Parse episodes
        episodes = []
        for line in lines[2:]:  # Skip header lines
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    reward = float(parts[0])
                    length = int(parts[1])
                    timestamp = float(parts[2])
                    episodes.append({
                        'reward': reward,
                        'length': length,
                        'timestamp': timestamp
                    })
                except ValueError:
                    continue
        
        return episodes
    except Exception as e:
        print(f"Error parsing logs: {e}")
        return None

def get_model_checkpoints():
    """Get information about saved model checkpoints."""
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith('.zip'):
            filepath = os.path.join(checkpoints_dir, file)
            timestamp = os.path.getctime(filepath)
            size = os.path.getsize(filepath)
            checkpoints.append({
                'filename': file,
                'timestamp': timestamp,
                'size_mb': size / (1024 * 1024)
            })
    
    return sorted(checkpoints, key=lambda x: x['timestamp'])

def estimate_completion_time(episodes):
    """Estimate training completion time based on current progress."""
    if not episodes or len(episodes) < 2:
        return None
    
    # Calculate timesteps per second
    recent_episodes = episodes[-10:]  # Use last 10 episodes
    if len(recent_episodes) < 2:
        return None
    
    time_span = recent_episodes[-1]['timestamp'] - recent_episodes[0]['timestamp']
    total_timesteps = sum(ep['length'] for ep in recent_episodes)
    
    if time_span <= 0:
        return None
    
    timesteps_per_second = total_timesteps / time_span
    
    # Estimate remaining time (assuming 5M total timesteps)
    total_target = 5_000_000
    current_timesteps = sum(ep['length'] for ep in episodes)
    remaining_timesteps = total_target - current_timesteps
    
    if remaining_timesteps <= 0:
        return "Training Complete!"
    
    remaining_seconds = remaining_timesteps / timesteps_per_second
    return str(timedelta(seconds=int(remaining_seconds)))

def create_progress_plot(episodes):
    """Create a progress plot showing training metrics."""
    if not episodes or len(episodes) < 5:
        return
    
    rewards = [ep['reward'] for ep in episodes]
    lengths = [ep['length'] for ep in episodes]
    timestamps = [ep['timestamp'] for ep in episodes]
    
    # Convert timestamps to relative time in hours
    start_time = timestamps[0]
    hours = [(t - start_time) / 3600 for t in timestamps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(hours, rewards, 'b-', alpha=0.7, label='Episode Rewards')
    if len(rewards) > 10:
        # Add moving average
        window = min(10, len(rewards) // 4)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        avg_hours = hours[window-1:]
        ax1.plot(avg_hours, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')
    
    ax1.set_xlabel('Training Time (hours)')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress - Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(hours, lengths, 'g-', alpha=0.7, label='Episode Length')
    if len(lengths) > 10:
        window = min(10, len(lengths) // 4)
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        avg_hours = hours[window-1:]
        ax2.plot(avg_hours, moving_avg, 'orange', linewidth=2, label=f'Moving Average ({window})')
    
    ax2.set_xlabel('Training Time (hours)')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Progress - Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_status_report():
    """Print a comprehensive status report."""
    print("🎮 CLASH ROYALE EXTENDED TRAINING MONITOR")
    print("=" * 60)
    print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if training is running
    is_running = check_training_status()
    status_icon = "🟢" if is_running else "🔴"
    print(f"Training Status: {status_icon} {'RUNNING' if is_running else 'STOPPED'}")
    print()
    
    # Parse training logs
    episodes = parse_training_logs()
    if episodes:
        print("📊 TRAINING PROGRESS")
        print("-" * 30)
        print(f"Episodes Completed: {len(episodes)}")
        
        total_timesteps = sum(ep['length'] for ep in episodes)
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Progress: {(total_timesteps / 5_000_000) * 100:.2f}% of 5M target")
        
        if len(episodes) >= 5:
            recent_rewards = [ep['reward'] for ep in episodes[-5:]]
            avg_reward = np.mean(recent_rewards)
            print(f"Recent Avg Reward: {avg_reward:.2f}")
            
            recent_lengths = [ep['length'] for ep in episodes[-5:]]
            avg_length = np.mean(recent_lengths)
            print(f"Recent Avg Episode Length: {avg_length:.0f}")
        
        # Estimate completion time
        eta = estimate_completion_time(episodes)
        if eta:
            print(f"Estimated Time Remaining: {eta}")
        
        # Create progress plot
        try:
            create_progress_plot(episodes)
            print("📈 Progress plot saved: analysis_plots/training_progress.png")
        except Exception as e:
            print(f"⚠️  Could not create progress plot: {e}")
    else:
        print("📊 No training logs found yet")
    
    print()
    
    # Check model checkpoints
    checkpoints = get_model_checkpoints()
    if checkpoints:
        print("💾 MODEL CHECKPOINTS")
        print("-" * 30)
        for cp in checkpoints[-3:]:  # Show last 3 checkpoints
            timestamp = datetime.fromtimestamp(cp['timestamp'])
            print(f"  {cp['filename']} ({cp['size_mb']:.1f}MB) - {timestamp.strftime('%H:%M:%S')}")
    else:
        print("💾 No checkpoints found yet")
    
    print()
    
    # TensorBoard info
    tb_dir = get_latest_tensorboard_logs()
    if tb_dir:
        print("📈 TENSORBOARD MONITORING")
        print("-" * 30)
        print(f"Log Directory: {tb_dir}")
        print("To view training metrics:")
        print(f"  tensorboard --logdir {tb_dir}")
        print("  Then open: http://localhost:6006")
    
    print()
    print("=" * 60)

def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Extended PPO Training")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Run continuous monitoring (updates every 30 seconds)")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Update interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    if args.continuous:
        print("Starting continuous monitoring...")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                print_status_report()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print_status_report()

if __name__ == "__main__":
    main()
