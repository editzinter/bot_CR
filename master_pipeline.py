#!/usr/bin/env python3
"""
Master Pipeline for Clash Royale RL Transfer Learning
This script orchestrates the complete pipeline from simulation training
to real game deployment and validation.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎮 CLASH ROYALE RL TRANSFER LEARNING PIPELINE        ║
    ║                                                              ║
    ║  Simulation Training → Model Analysis → Real Game Transfer   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    print("🔍 CHECKING PREREQUISITES")
    print("-" * 40)
    
    checks = {
        "Python Environment": sys.version_info >= (3, 10),
        "Clash Royale Package": False,
        "Stable Baselines3": False,
        "OpenCV": False,
        "BlueStacks/ADB": False
    }
    
    # Check packages
    try:
        import clash_royale
        checks["Clash Royale Package"] = True
    except ImportError:
        pass
    
    try:
        import stable_baselines3
        checks["Stable Baselines3"] = True
    except ImportError:
        pass
    
    try:
        import cv2
        checks["OpenCV"] = True
    except ImportError:
        pass
    
    # Check ADB (simplified check)
    import subprocess
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, timeout=5)
        checks["BlueStacks/ADB"] = result.returncode == 0
    except:
        pass
    
    # Print results
    all_good = True
    for check, status in checks.items():
        status_icon = "✓" if status else "❌"
        print(f"  {status_icon} {check}")
        if not status:
            all_good = False
    
    if not all_good:
        print("\n⚠️  Some prerequisites are missing. Please install required components.")
        return False
    
    print("\n✅ All prerequisites met!")
    return True

def run_extended_training() -> bool:
    """Run extended simulation training."""
    print("\n🚀 PHASE 1: EXTENDED SIMULATION TRAINING")
    print("=" * 60)
    
    try:
        from extended_training import train_extended_ppo
        
        print("Starting 5M timestep training...")
        print("This may take several hours. Monitor with TensorBoard:")
        print("  tensorboard --logdir extended_tensorboard/")
        
        model, metadata = train_extended_ppo()
        
        if model is not None and metadata.get("status") != "error":
            print("✅ Extended training completed successfully!")
            return True
        else:
            print("❌ Extended training failed or was interrupted")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def run_model_analysis() -> bool:
    """Run model analysis and documentation."""
    print("\n📊 PHASE 2: MODEL ANALYSIS")
    print("=" * 60)
    
    try:
        from model_analysis import analyze_available_models
        
        results = analyze_available_models()
        
        if results:
            print("✅ Model analysis completed successfully!")
            return True
        else:
            print("❌ Model analysis failed - no models found")
            return False
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def setup_bluestacks() -> bool:
    """Setup BlueStacks integration."""
    print("\n🔧 PHASE 3: BLUESTACKS SETUP")
    print("=" * 60)
    
    try:
        from bluestacks_integration import setup_bluestacks_integration
        
        success = setup_bluestacks_integration()
        
        if success:
            print("✅ BlueStacks integration setup completed!")
            return True
        else:
            print("❌ BlueStacks setup failed")
            return False
            
    except Exception as e:
        print(f"❌ BlueStacks setup error: {e}")
        return False

def test_vision_system() -> bool:
    """Test the vision system."""
    print("\n👁️  PHASE 4: VISION SYSTEM TEST")
    print("=" * 60)
    
    try:
        from vision_system import test_vision_system
        
        test_vision_system()
        print("✅ Vision system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Vision system error: {e}")
        return False

def run_real_game_finetuning() -> bool:
    """Run real game fine-tuning."""
    print("\n🎮 PHASE 5: REAL GAME FINE-TUNING")
    print("=" * 60)
    
    print("⚠️  IMPORTANT SAFETY NOTICE:")
    print("  - Use a test account, not your main account")
    print("  - Monitor the process closely")
    print("  - Stop if anything seems wrong")
    
    response = input("\nProceed with real game fine-tuning? (y/N): ")
    if response.lower() != 'y':
        print("Real game fine-tuning skipped by user")
        return True
    
    try:
        from real_game_finetuning import run_real_game_finetuning
        
        success = run_real_game_finetuning()
        
        if success:
            print("✅ Real game fine-tuning completed!")
            return True
        else:
            print("❌ Real game fine-tuning failed")
            return False
            
    except Exception as e:
        print(f"❌ Fine-tuning error: {e}")
        return False

def run_validation_analysis() -> bool:
    """Run comprehensive validation and analysis."""
    print("\n📈 PHASE 6: VALIDATION & ANALYSIS")
    print("=" * 60)
    
    try:
        from validation_analysis import run_comprehensive_analysis
        
        results = run_comprehensive_analysis()
        
        if results:
            print("✅ Validation analysis completed!")
            return True
        else:
            print("❌ Validation analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def generate_final_report(pipeline_results: Dict[str, bool]) -> str:
    """Generate final pipeline report."""
    report = f"""
# Clash Royale RL Transfer Learning Pipeline Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Execution Summary

### Phase Results:
"""
    
    phases = [
        ("Extended Training", "extended_training"),
        ("Model Analysis", "model_analysis"),
        ("BlueStacks Setup", "bluestacks_setup"),
        ("Vision System Test", "vision_test"),
        ("Real Game Fine-tuning", "real_game_finetuning"),
        ("Validation Analysis", "validation_analysis")
    ]
    
    for phase_name, phase_key in phases:
        status = "✅ COMPLETED" if pipeline_results.get(phase_key, False) else "❌ FAILED"
        report += f"- **{phase_name}**: {status}\n"
    
    # Overall success
    all_critical_passed = all(pipeline_results.get(key, False) for key in 
                             ["extended_training", "model_analysis"])
    
    report += f"""
## Overall Status: {'🎉 SUCCESS' if all_critical_passed else '⚠️ PARTIAL SUCCESS'}

## Generated Files:
- `extended_models/`: Trained models from simulation
- `model_analysis_results.json`: Model analysis documentation
- `real_game_models/`: Fine-tuned models (if completed)
- `analysis_plots/`: Performance visualizations
- `performance_analysis_report.md`: Detailed analysis report

## Next Steps:
{'- Deploy model for real gameplay testing' if all_critical_passed else '- Address failed phases before deployment'}
- Monitor performance in production
- Collect feedback for further improvements
- Consider additional fine-tuning iterations

## Safety Reminders:
- Always use test accounts for real game testing
- Monitor automated gameplay closely
- Respect game terms of service
- Stop immediately if issues arise
"""
    
    return report

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Clash Royale RL Transfer Learning Pipeline")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip extended training (use existing models)")
    parser.add_argument("--skip-real-game", action="store_true",
                       help="Skip real game fine-tuning")
    parser.add_argument("--phases", nargs="+", 
                       choices=["training", "analysis", "bluestacks", "vision", "finetuning", "validation"],
                       help="Run only specific phases")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Track results
    pipeline_results = {}
    start_time = time.time()
    
    # Define phases
    phases = [
        ("extended_training", "Extended Training", run_extended_training, not args.skip_training),
        ("model_analysis", "Model Analysis", run_model_analysis, True),
        ("bluestacks_setup", "BlueStacks Setup", setup_bluestacks, True),
        ("vision_test", "Vision System Test", test_vision_system, True),
        ("real_game_finetuning", "Real Game Fine-tuning", run_real_game_finetuning, not args.skip_real_game),
        ("validation_analysis", "Validation Analysis", run_validation_analysis, True)
    ]
    
    # Filter phases if specified
    if args.phases:
        phase_map = {
            "training": "extended_training",
            "analysis": "model_analysis", 
            "bluestacks": "bluestacks_setup",
            "vision": "vision_test",
            "finetuning": "real_game_finetuning",
            "validation": "validation_analysis"
        }
        selected_phases = [phase_map[p] for p in args.phases]
        phases = [(key, name, func, enabled) for key, name, func, enabled in phases 
                 if key in selected_phases]
    
    # Execute phases
    for phase_key, phase_name, phase_func, enabled in phases:
        if not enabled:
            print(f"\n⏭️  Skipping {phase_name}")
            pipeline_results[phase_key] = True  # Mark as successful skip
            continue
        
        try:
            success = phase_func()
            pipeline_results[phase_key] = success
            
            if not success:
                print(f"\n⚠️  {phase_name} failed. Continue anyway? (y/N): ", end="")
                response = input()
                if response.lower() != 'y':
                    print("Pipeline stopped by user")
                    break
        
        except KeyboardInterrupt:
            print(f"\n⚠️  {phase_name} interrupted by user")
            pipeline_results[phase_key] = False
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {phase_name}: {e}")
            pipeline_results[phase_key] = False
    
    # Generate final report
    total_time = time.time() - start_time
    report = generate_final_report(pipeline_results)
    
    # Save report
    with open("pipeline_execution_report.md", 'w') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {total_time/3600:.2f} hours")
    print(f"Report saved: pipeline_execution_report.md")
    
    successful_phases = sum(1 for success in pipeline_results.values() if success)
    total_phases = len(pipeline_results)
    print(f"Phases completed: {successful_phases}/{total_phases}")
    
    if successful_phases == total_phases:
        print("🎉 All phases completed successfully!")
        return 0
    else:
        print("⚠️  Some phases failed - check report for details")
        return 1

if __name__ == "__main__":
    exit(main())
