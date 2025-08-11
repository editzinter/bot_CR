# 🎮 Clash Royale RL Transfer Learning Pipeline

A comprehensive reinforcement learning system that trains an AI agent in a simulated Clash Royale environment and transfers the learned policy to real gameplay using BlueStacks automation.

## 🎯 Project Overview

This project implements a complete pipeline for:
1. **Extended Simulation Training** - Train PPO agent for 5M+ timesteps
2. **Model Analysis** - Document and analyze trained models
3. **Real Game Integration** - Transfer to BlueStacks with computer vision
4. **Fine-tuning** - Adapt the model to real game dynamics
5. **Validation** - Comprehensive performance analysis

## 🏗️ Architecture

```
Simulated Environment → Extended Training → Model Analysis
                                              ↓
Real Game Validation ← Fine-tuning ← BlueStacks Integration
```

## 📋 Prerequisites

### System Requirements
- **Python 3.10+**
- **Windows** (for BlueStacks integration)
- **8GB+ RAM** (16GB recommended for training)
- **GPU** (optional but recommended for faster training)

### Software Dependencies
```bash
# Core RL packages
pip install stable-baselines3[extra]
pip install gymnasium
pip install torch torchvision

# Computer vision
pip install opencv-python
pip install pillow

# Data analysis
pip install matplotlib seaborn pandas numpy

# BlueStacks automation
# Install BlueStacks and ADB separately
```

### BlueStacks Setup
1. Install [BlueStacks](https://www.bluestacks.com/)
2. Install [Android Debug Bridge (ADB)](https://developer.android.com/studio/command-line/adb)
3. Enable ADB debugging in BlueStacks
4. Install Clash Royale in BlueStacks

## 🚀 Quick Start

### Option 1: Full Pipeline
```bash
# Run complete pipeline (5+ hours)
python master_pipeline.py
```

### Option 2: Step-by-Step
```bash
# 1. Extended training (3-5 hours)
python extended_training.py

# 2. Model analysis
python model_analysis.py

# 3. BlueStacks setup
python bluestacks_integration.py

# 4. Vision system test
python vision_system.py

# 5. Real game fine-tuning (CAUTION: Use test account)
python real_game_finetuning.py

# 6. Performance analysis
python validation_analysis.py
```

### Option 3: Selective Phases
```bash
# Run only specific phases
python master_pipeline.py --phases training analysis
python master_pipeline.py --skip-real-game
python master_pipeline.py --skip-training
```

## 📁 Project Structure

```
├── master_pipeline.py              # Main orchestration script
├── extended_training.py            # 5M timestep training
├── model_analysis.py               # Model documentation
├── bluestacks_integration.py       # ADB automation
├── vision_system.py                # Computer vision system
├── real_game_finetuning.py        # Real game adaptation
├── validation_analysis.py          # Performance analysis
├── environment_wrappers.py         # Enhanced env wrappers
├── clash_royale/                   # Enhanced environment
├── extended_models/                # Trained models
├── real_game_models/               # Fine-tuned models
├── analysis_plots/                 # Visualizations
└── logs/                          # Training logs
```

## 🎮 Usage Guide

### Phase 1: Simulation Training
- Trains PPO agent for 5M timesteps
- Uses 8 parallel environments
- Includes comprehensive logging
- Saves checkpoints every 250k steps

**Monitor Progress:**
```bash
tensorboard --logdir extended_tensorboard/
```

### Phase 2: Model Analysis
- Documents input/output formats
- Analyzes action mappings
- Tests inference performance
- Generates transfer guide

### Phase 3: BlueStacks Integration
- Sets up ADB connection
- Tests screen capture
- Validates action mapping
- Configures safety systems

### Phase 4: Vision System
- Real-time frame processing
- Game state detection
- UI element recognition
- Performance optimization

### Phase 5: Real Game Fine-tuning
⚠️ **SAFETY FIRST**: Use test account only!

- Transfers pre-trained model
- Adapts to real game dynamics
- Implements safety mechanisms
- Logs all interactions

### Phase 6: Validation
- Compares sim vs real performance
- Generates comprehensive reports
- Creates visualizations
- Provides recommendations

## 🛡️ Safety Features

### Automated Safety Checks
- **Action Timeout**: Stops if no successful actions for 30s
- **Failure Limit**: Ends episode after 5 consecutive failures
- **Time Limits**: Maximum episode duration (3 minutes)
- **Emergency Stop**: Manual interruption capability

### Best Practices
- Always use test accounts
- Monitor gameplay closely
- Respect game terms of service
- Start with short sessions

## 📊 Performance Metrics

### Simulation Environment
- Episode rewards and lengths
- Action diversity analysis
- Learning curve progression
- Model convergence metrics

### Real Game Environment
- Win/loss rates
- Safety stop frequency
- Action success rates
- Transfer learning effectiveness

## 🔧 Configuration

### Training Configuration
```python
config = {
    "total_timesteps": 5_000_000,
    "n_envs": 8,
    "learning_rate": 2.5e-4,
    "batch_size": 256,
    "n_epochs": 10
}
```

### Fine-tuning Configuration
```python
config = {
    "max_episodes": 100,
    "learning_rate": 1e-5,
    "exploration_rate": 0.1,
    "safety_timeout": 30
}
```

## 📈 Expected Results

### Simulation Performance
- **Training Time**: 3-5 hours (with GPU)
- **Final Model Size**: ~24MB
- **Action Space Coverage**: 500+ unique actions
- **Convergence**: Stable after 2M timesteps

### Transfer Learning
- **Performance Retention**: 50-80% of simulation performance
- **Adaptation Time**: 50-100 real game episodes
- **Win Rate**: 40-60% (depending on opponent strength)
- **Safety**: <5% emergency stops

## 🐛 Troubleshooting

### Common Issues

**Training Fails**
- Check GPU memory availability
- Reduce batch size or n_envs
- Verify environment installation

**BlueStacks Connection Issues**
- Ensure ADB is in PATH
- Check BlueStacks ADB settings
- Restart BlueStacks and try again

**Vision System Errors**
- Verify OpenCV installation
- Check screen resolution settings
- Ensure proper game positioning

**Real Game Issues**
- Use test account only
- Check internet connection
- Verify game is in battle mode

### Debug Mode
```bash
# Enable verbose logging
python master_pipeline.py --verbose

# Test individual components
python -c "from bluestacks_integration import setup_bluestacks_integration; setup_bluestacks_integration()"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## ⚖️ Legal & Ethics

- **Educational Purpose**: This project is for research and learning
- **Terms of Service**: Respect Clash Royale's terms of service
- **Fair Play**: Don't use for competitive advantage
- **Account Safety**: Use test accounts only

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- MSU-AI for the base Clash Royale environment
- Stable Baselines3 team for RL algorithms
- OpenAI Gymnasium for environment standards
- BlueStacks for Android emulation platform

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Review generated logs
3. Open GitHub issue with details
4. Include system specifications

---

**⚠️ Disclaimer**: This project is for educational purposes. Use responsibly and respect game terms of service.
