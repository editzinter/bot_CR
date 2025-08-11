#!/usr/bin/env python3
"""
Model Analysis and Documentation for Transfer Learning
This script analyzes trained models and documents their input/output formats
for transfer to real Clash Royale gameplay.
"""

import os
import json
import numpy as np
import torch
import gymnasium
import clash_royale
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ModelAnalyzer:
    """
    Comprehensive model analysis for transfer learning preparation.
    """
    
    def __init__(self, model_path: str):
        """Initialize analyzer with trained model."""
        self.model_path = model_path
        self.model = None
        self.env = None
        self.analysis_results = {}
        
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = PPO.load(self.model_path)
            print(f"✓ Model loaded from: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_test_environment(self):
        """Create test environment matching training setup."""
        try:
            self.env = gymnasium.make("clash-royale", render_mode="rgb_array")
            print(f"✓ Test environment created")
            return True
        except Exception as e:
            print(f"❌ Error creating environment: {e}")
            return False
    
    def analyze_input_format(self) -> Dict[str, Any]:
        """Analyze model input requirements."""
        print("\n📥 ANALYZING INPUT FORMAT")
        print("-" * 40)
        
        input_analysis = {}
        
        if self.env is None:
            return input_analysis
        
        # Get observation space details
        obs_space = self.env.observation_space
        input_analysis["observation_space"] = {
            "type": str(type(obs_space)),
            "shape": obs_space.shape,
            "dtype": str(obs_space.dtype),
            "low": obs_space.low.tolist() if hasattr(obs_space.low, 'tolist') else str(obs_space.low),
            "high": obs_space.high.tolist() if hasattr(obs_space.high, 'tolist') else str(obs_space.high)
        }
        
        print(f"  Observation Shape: {obs_space.shape}")
        print(f"  Data Type: {obs_space.dtype}")
        print(f"  Value Range: [{obs_space.low.min()}, {obs_space.high.max()}]")
        
        # Test with sample observation
        obs, _ = self.env.reset()
        input_analysis["sample_observation"] = {
            "shape": obs.shape,
            "dtype": str(obs.dtype),
            "min_value": float(obs.min()),
            "max_value": float(obs.max()),
            "mean_value": float(obs.mean())
        }
        
        print(f"  Sample Obs Shape: {obs.shape}")
        print(f"  Sample Value Range: [{obs.min():.2f}, {obs.max():.2f}]")
        
        # Preprocessing requirements
        input_analysis["preprocessing_requirements"] = {
            "normalization": "Values should be in range [0, 255] for RGB images",
            "frame_stacking": "Model may expect 4 stacked frames",
            "resize_needed": f"Input should be exactly {obs_space.shape}",
            "color_format": "RGB format (Height, Width, Channels)"
        }
        
        return input_analysis
    
    def analyze_action_format(self) -> Dict[str, Any]:
        """Analyze model action space and mapping."""
        print("\n📤 ANALYZING ACTION FORMAT")
        print("-" * 40)
        
        action_analysis = {}
        
        if self.env is None:
            return action_analysis
        
        # Get action space details
        action_space = self.env.action_space
        action_analysis["action_space"] = {
            "type": str(type(action_space)),
            "n": action_space.n if hasattr(action_space, 'n') else None,
            "shape": getattr(action_space, 'shape', None)
        }
        
        print(f"  Action Space: {action_space}")
        print(f"  Total Actions: {action_space.n}")
        
        # Action mapping details (from environment implementation)
        action_analysis["action_mapping"] = {
            "total_actions": 2304,
            "dimensions": {
                "x_positions": 18,  # Arena width
                "y_positions": 32,  # Arena height  
                "card_indices": 4   # Cards in hand
            },
            "formula": "action = x * (height * 4) + y * 4 + card_index",
            "reverse_formula": {
                "card_index": "action % 4",
                "y_position": "(action // 4) % height", 
                "x_position": "action // (height * 4)"
            }
        }
        
        print(f"  Action Dimensions: 18 x 32 x 4 = {18*32*4}")
        print(f"  Mapping: (x, y, card) -> discrete action")
        
        # Test action decoding
        test_actions = [0, 1, 4, 100, 1000, 2303]
        decoded_actions = []
        
        for action in test_actions:
            card_idx = action % 4
            y_pos = (action // 4) % 32
            x_pos = action // (32 * 4)
            decoded_actions.append({
                "action": action,
                "x": x_pos,
                "y": y_pos,
                "card": card_idx
            })
        
        action_analysis["sample_decodings"] = decoded_actions
        
        print("  Sample Action Decodings:")
        for decode in decoded_actions[:3]:
            print(f"    Action {decode['action']:4d} -> x={decode['x']:2d}, y={decode['y']:2d}, card={decode['card']}")
        
        return action_analysis
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture and parameters."""
        print("\n🏗️  ANALYZING MODEL ARCHITECTURE")
        print("-" * 40)
        
        arch_analysis = {}
        
        if self.model is None:
            return arch_analysis
        
        # Get policy network details
        policy = self.model.policy
        arch_analysis["policy_type"] = str(type(policy))
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        arch_analysis["parameters"] = {
            "total": total_params,
            "trainable": trainable_params,
            "size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")
        
        # Architecture details
        if hasattr(policy, 'features_extractor'):
            arch_analysis["cnn_architecture"] = str(policy.features_extractor)
            print(f"  CNN Architecture: {type(policy.features_extractor).__name__}")
        
        # Hyperparameters
        arch_analysis["hyperparameters"] = {
            "learning_rate": float(self.model.learning_rate),
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "clip_range": float(self.model.clip_range(1.0)),
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
        }
        
        return arch_analysis
    
    def test_model_inference(self, num_tests: int = 100) -> Dict[str, Any]:
        """Test model inference and analyze outputs."""
        print("\n🧪 TESTING MODEL INFERENCE")
        print("-" * 40)
        
        inference_analysis = {}
        
        if self.model is None or self.env is None:
            return inference_analysis
        
        # Collect inference data
        actions = []
        action_probs = []
        values = []
        inference_times = []
        
        obs, _ = self.env.reset()
        
        for i in range(num_tests):
            start_time = time.time()
            
            # Get action and additional info
            action, _states = self.model.predict(obs, deterministic=False)
            
            # Get action probabilities and value if possible
            if hasattr(self.model.policy, 'predict'):
                try:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        features = self.model.policy.extract_features(obs_tensor)
                        action_logits = self.model.policy.action_net(features)
                        action_prob = torch.softmax(action_logits, dim=-1)
                        value = self.model.policy.value_net(features)
                    
                    action_probs.append(action_prob.cpu().numpy())
                    values.append(float(value.cpu().numpy()))
                except:
                    pass
            
            inference_time = time.time() - start_time
            
            actions.append(int(action))
            inference_times.append(inference_time)
            
            # Step environment for next observation
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset()
        
        # Analyze results
        actions = np.array(actions)
        inference_times = np.array(inference_times)
        
        inference_analysis["performance"] = {
            "mean_inference_time_ms": float(np.mean(inference_times) * 1000),
            "std_inference_time_ms": float(np.std(inference_times) * 1000),
            "fps_capability": float(1.0 / np.mean(inference_times))
        }
        
        inference_analysis["action_distribution"] = {
            "unique_actions": int(len(np.unique(actions))),
            "most_common_action": int(np.bincount(actions).argmax()),
            "action_entropy": float(-np.sum(np.bincount(actions) / len(actions) * 
                                          np.log(np.bincount(actions) / len(actions) + 1e-10)))
        }
        
        print(f"  Mean Inference Time: {np.mean(inference_times)*1000:.2f} ms")
        print(f"  FPS Capability: {1.0/np.mean(inference_times):.1f}")
        print(f"  Unique Actions Used: {len(np.unique(actions))}/2304")
        
        return inference_analysis
    
    def generate_transfer_guide(self) -> Dict[str, Any]:
        """Generate comprehensive transfer learning guide."""
        print("\n📋 GENERATING TRANSFER GUIDE")
        print("-" * 40)
        
        guide = {
            "real_game_requirements": {
                "screen_capture": "Capture 128x128 RGB frames from BlueStacks",
                "preprocessing": "Normalize to [0,255], ensure RGB format",
                "frame_rate": "Model can handle 30+ FPS",
                "action_mapping": "Convert discrete actions to screen coordinates"
            },
            "bluestacks_integration": {
                "adb_commands": [
                    "adb shell input tap <x> <y>  # For card placement",
                    "adb shell screencap -p /sdcard/screen.png  # For capture",
                    "adb pull /sdcard/screen.png  # Download frame"
                ],
                "coordinate_mapping": "Scale model coordinates to screen resolution",
                "timing_considerations": "Add delays for animations"
            },
            "fine_tuning_strategy": {
                "initial_transfer": "Use pre-trained weights as starting point",
                "reward_adaptation": "Adjust rewards for real game outcomes",
                "exploration_strategy": "Reduce exploration in real environment",
                "safety_measures": "Use test account, limit play time"
            }
        }
        
        return guide
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete model analysis."""
        print("🔍 COMPLETE MODEL ANALYSIS")
        print("=" * 60)
        
        if not self.load_model():
            return {}
        
        if not self.create_test_environment():
            return {}
        
        # Run all analyses
        self.analysis_results = {
            "input_format": self.analyze_input_format(),
            "action_format": self.analyze_action_format(), 
            "model_architecture": self.analyze_model_architecture(),
            "inference_testing": self.test_model_inference(),
            "transfer_guide": self.generate_transfer_guide()
        }
        
        # Save results
        output_path = "model_analysis_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ Complete analysis saved to: {output_path}")
        
        if self.env:
            self.env.close()
        
        return self.analysis_results

def analyze_available_models():
    """Analyze all available trained models."""
    print("🔍 ANALYZING AVAILABLE MODELS")
    print("=" * 60)
    
    # Look for models in various directories
    model_dirs = ["models", "test_models", "enhanced_models", "extended_models", "checkpoints"]
    found_models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    model_path = os.path.join(model_dir, file.replace('.zip', ''))
                    found_models.append(model_path)
    
    if not found_models:
        print("❌ No trained models found. Run training first.")
        return
    
    print(f"Found {len(found_models)} trained models:")
    for i, model_path in enumerate(found_models):
        print(f"  {i+1}. {model_path}")
    
    # Analyze the most recent/best model
    best_model = found_models[0]  # You could implement logic to find the best one
    print(f"\nAnalyzing: {best_model}")
    
    analyzer = ModelAnalyzer(best_model)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    import time
    
    # Run analysis on available models
    analyze_available_models()
