#!/usr/bin/env python3
"""
Environment Setup Validation Script
Tests all components needed for the extended training pipeline.
"""

import sys
import os
import time
import traceback
from datetime import datetime

def test_python_version():
    """Test Python version requirement."""
    print("🐍 Testing Python Version...")
    version = sys.version_info
    required = (3, 10)
    
    if version >= required:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (>= 3.10 required)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (>= 3.10 required)")
        return False

def test_core_imports():
    """Test core package imports."""
    print("\n📦 Testing Core Package Imports...")
    
    packages = {
        'clash_royale': 'Clash Royale Environment',
        'gymnasium': 'Gymnasium Framework', 
        'stable_baselines3': 'Stable Baselines3',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    results = {}
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {description}")
            results[package] = True
        except ImportError as e:
            print(f"   ❌ {description}: {e}")
            results[package] = False
    
    return all(results.values()), results

def test_environment_creation():
    """Test Clash Royale environment creation."""
    print("\n🎮 Testing Environment Creation...")
    
    try:
        import clash_royale
        import gymnasium
        
        # Create environment
        env = gymnasium.make("clash-royale", render_mode="rgb_array")
        print("   ✅ Environment created successfully")
        
        # Test basic properties
        print(f"   📊 Action Space: {env.action_space}")
        print(f"   📊 Observation Space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"   📊 Observation Shape: {obs.shape}")
        print(f"   📊 Observation Type: {obs.dtype}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   📊 Sample Action: {action}")
        print(f"   📊 Sample Reward: {reward}")
        
        env.close()
        print("   ✅ Environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_stable_baselines3():
    """Test Stable Baselines3 functionality."""
    print("\n🤖 Testing Stable Baselines3...")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        import clash_royale
        import gymnasium
        
        # Create vectorized environment
        env = make_vec_env("clash-royale", n_envs=1, env_kwargs={"render_mode": "rgb_array"})
        print("   ✅ Vectorized environment created")
        
        # Create PPO model
        model = PPO("CnnPolicy", env, verbose=0, device="auto")
        print("   ✅ PPO model created")
        
        # Test device
        device = model.device
        print(f"   📊 Using device: {device}")
        
        # Quick training test (just a few steps)
        print("   🏃 Running quick training test...")
        model.learn(total_timesteps=10, progress_bar=False)
        print("   ✅ Quick training test passed")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Stable Baselines3 test failed: {e}")
        traceback.print_exc()
        return False

def test_optional_packages():
    """Test optional packages for enhanced functionality."""
    print("\n🔧 Testing Optional Packages...")
    
    optional_packages = {
        'cv2': 'OpenCV (for vision system)',
        'tensorboard': 'TensorBoard (for logging)',
        'wandb': 'Weights & Biases (for experiment tracking)'
    }
    
    results = {}
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {description}")
            results[package] = True
        except ImportError:
            print(f"   ⚠️  {description} - Not installed (optional)")
            results[package] = False
    
    return results

def test_directory_structure():
    """Test and create necessary directories."""
    print("\n📁 Testing Directory Structure...")
    
    directories = [
        'models',
        'extended_models', 
        'logs',
        'extended_logs',
        'tensorboard_logs',
        'extended_tensorboard',
        'real_game_models',
        'analysis_plots'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created directory: {directory}")
        else:
            print(f"   ✅ Directory exists: {directory}")
    
    return True

def generate_setup_report(results):
    """Generate setup validation report."""
    report = f"""# Environment Setup Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Information
- Python Version: {sys.version}
- Platform: {sys.platform}

## Core Requirements
"""
    
    core_tests = [
        ("Python Version", results.get('python_version', False)),
        ("Core Imports", results.get('core_imports', False)),
        ("Environment Creation", results.get('environment_creation', False)),
        ("Stable Baselines3", results.get('stable_baselines3', False))
    ]
    
    for test_name, passed in core_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        report += f"- **{test_name}**: {status}\n"
    
    # Optional packages
    if 'optional_packages' in results:
        report += "\n## Optional Packages\n"
        for package, installed in results['optional_packages'].items():
            status = "✅ Installed" if installed else "⚠️ Not Installed"
            report += f"- **{package}**: {status}\n"
    
    # Overall status
    all_core_passed = all(results.get(key, False) for key in 
                         ['python_version', 'core_imports', 'environment_creation', 'stable_baselines3'])
    
    report += f"""
## Overall Status: {'🎉 READY FOR TRAINING' if all_core_passed else '❌ SETUP INCOMPLETE'}

## Next Steps:
{'- Proceed with extended training' if all_core_passed else '- Install missing dependencies'}
- Run master pipeline: `python master_pipeline.py`
- Monitor training with TensorBoard
"""
    
    return report

def main():
    """Main validation function."""
    print("🔍 CLASH ROYALE RL ENVIRONMENT SETUP VALIDATION")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['python_version'] = test_python_version()
    results['core_imports'], package_results = test_core_imports()
    results['environment_creation'] = test_environment_creation()
    results['stable_baselines3'] = test_stable_baselines3()
    results['optional_packages'] = test_optional_packages()
    results['directory_structure'] = test_directory_structure()
    
    # Generate report
    report = generate_setup_report(results)
    
    # Save report
    with open('environment_setup_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_core_passed = all(results.get(key, False) for key in 
                         ['python_version', 'core_imports', 'environment_creation', 'stable_baselines3'])
    
    if all_core_passed:
        print("🎉 Environment setup validation PASSED!")
        print("✅ Ready for extended training")
        print("📄 Report saved: environment_setup_report.md")
        return 0
    else:
        print("❌ Environment setup validation FAILED!")
        print("⚠️  Please address the issues above before proceeding")
        print("📄 Report saved: environment_setup_report.md")
        return 1

if __name__ == "__main__":
    exit(main())
