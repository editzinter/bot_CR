#!/usr/bin/env python3
"""
BlueStacks Integration for Real Clash Royale Gameplay
This module handles ADB automation, screen capture, and action mapping
for transferring the trained RL agent to real gameplay.
"""

import os
import time
import subprocess
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import json
from dataclasses import dataclass
from PIL import Image
import threading
import queue

@dataclass
class GameConfig:
    """Configuration for BlueStacks and Clash Royale integration."""
    bluestacks_port: int = 5555
    screen_width: int = 1280
    screen_height: int = 720
    game_area_x: int = 0
    game_area_y: int = 0
    game_area_width: int = 1280
    game_area_height: int = 720
    model_input_size: Tuple[int, int] = (128, 128)
    adb_path: str = "adb"  # Path to ADB executable

class BlueStacksController:
    """
    Controller for BlueStacks automation via ADB.
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.device_id = f"127.0.0.1:{config.bluestacks_port}"
        self.connected = False
        self.last_screenshot = None
        self.screenshot_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.capturing = False
        
    def connect(self) -> bool:
        """Connect to BlueStacks via ADB."""
        try:
            # Kill any existing ADB server
            subprocess.run([self.config.adb_path, "kill-server"], 
                         capture_output=True, timeout=5)
            
            # Start ADB server
            subprocess.run([self.config.adb_path, "start-server"], 
                         capture_output=True, timeout=5)
            
            # Connect to BlueStacks
            result = subprocess.run([self.config.adb_path, "connect", self.device_id], 
                                  capture_output=True, text=True, timeout=10)
            
            if "connected" in result.stdout or "already connected" in result.stdout:
                self.connected = True
                print(f"✓ Connected to BlueStacks at {self.device_id}")
                return True
            else:
                print(f"❌ Failed to connect: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from BlueStacks."""
        if self.capturing:
            self.stop_continuous_capture()
        
        try:
            subprocess.run([self.config.adb_path, "disconnect", self.device_id], 
                         capture_output=True, timeout=5)
            self.connected = False
            print("✓ Disconnected from BlueStacks")
        except Exception as e:
            print(f"⚠️  Disconnect error: {e}")
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture screenshot from BlueStacks."""
        if not self.connected:
            return None
        
        try:
            # Capture screenshot
            result = subprocess.run([
                self.config.adb_path, "-s", self.device_id, 
                "shell", "screencap", "-p"
            ], capture_output=True, timeout=2)
            
            if result.returncode == 0:
                # Convert bytes to image
                image_data = result.stdout
                image = Image.open(io.BytesIO(image_data))
                frame = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                self.last_screenshot = frame
                return frame
            else:
                print(f"❌ Screenshot failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
            return None
    
    def start_continuous_capture(self, fps: int = 30):
        """Start continuous screenshot capture in background thread."""
        if self.capturing:
            return
        
        self.capturing = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop, 
            args=(1.0/fps,), 
            daemon=True
        )
        self.capture_thread.start()
        print(f"✓ Started continuous capture at {fps} FPS")
    
    def stop_continuous_capture(self):
        """Stop continuous screenshot capture."""
        self.capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        print("✓ Stopped continuous capture")
    
    def _capture_loop(self, interval: float):
        """Background capture loop."""
        while self.capturing:
            frame = self.capture_screen()
            if frame is not None:
                try:
                    self.screenshot_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.screenshot_queue.get_nowait()
                        self.screenshot_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            time.sleep(interval)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        try:
            return self.screenshot_queue.get_nowait()
        except queue.Empty:
            return self.last_screenshot
    
    def tap(self, x: int, y: int, duration: int = 100) -> bool:
        """Tap at screen coordinates."""
        if not self.connected:
            return False
        
        try:
            result = subprocess.run([
                self.config.adb_path, "-s", self.device_id,
                "shell", "input", "tap", str(x), str(y)
            ], capture_output=True, timeout=2)
            
            if result.returncode == 0:
                time.sleep(duration / 1000.0)  # Convert ms to seconds
                return True
            else:
                print(f"❌ Tap failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Tap error: {e}")
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """Swipe from (x1,y1) to (x2,y2)."""
        if not self.connected:
            return False
        
        try:
            result = subprocess.run([
                self.config.adb_path, "-s", self.device_id,
                "shell", "input", "swipe", 
                str(x1), str(y1), str(x2), str(y2), str(duration)
            ], capture_output=True, timeout=3)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Swipe error: {e}")
            return False

class ActionMapper:
    """
    Maps model actions to real game coordinates and actions.
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.arena_bounds = self._calculate_arena_bounds()
        
    def _calculate_arena_bounds(self) -> Dict[str, int]:
        """Calculate arena boundaries on screen."""
        # These would need to be calibrated for actual Clash Royale UI
        # This is a placeholder - real implementation would need UI detection
        return {
            "left": int(self.config.screen_width * 0.1),
            "right": int(self.config.screen_width * 0.9),
            "top": int(self.config.screen_height * 0.2),
            "bottom": int(self.config.screen_height * 0.8),
            "width": int(self.config.screen_width * 0.8),
            "height": int(self.config.screen_height * 0.6)
        }
    
    def decode_model_action(self, action: int) -> Dict[str, int]:
        """Decode model action into x, y, card components."""
        card_idx = action % 4
        y_pos = (action // 4) % 32
        x_pos = action // (32 * 4)
        
        return {
            "x": x_pos,
            "y": y_pos,
            "card": card_idx,
            "raw_action": action
        }
    
    def map_to_screen_coordinates(self, model_x: int, model_y: int) -> Tuple[int, int]:
        """Map model coordinates to screen coordinates."""
        # Scale from model space (18x32) to screen arena space
        screen_x = self.arena_bounds["left"] + int(
            (model_x / 17) * self.arena_bounds["width"]
        )
        screen_y = self.arena_bounds["top"] + int(
            (model_y / 31) * self.arena_bounds["height"]
        )
        
        return screen_x, screen_y
    
    def get_card_position(self, card_idx: int) -> Tuple[int, int]:
        """Get screen position of card in hand."""
        # Placeholder - would need to detect actual card positions
        card_area_y = int(self.config.screen_height * 0.9)
        card_spacing = self.config.screen_width // 5
        card_x = card_spacing + (card_idx * card_spacing)
        
        return card_x, card_area_y
    
    def execute_action(self, controller: BlueStacksController, action: int) -> bool:
        """Execute a model action in the real game."""
        decoded = self.decode_model_action(action)
        
        # Get card position and tap it first
        card_x, card_y = self.get_card_position(decoded["card"])
        if not controller.tap(card_x, card_y, duration=50):
            return False
        
        # Small delay for card selection
        time.sleep(0.1)
        
        # Get placement position and tap
        place_x, place_y = self.map_to_screen_coordinates(decoded["x"], decoded["y"])
        return controller.tap(place_x, place_y, duration=100)

class GameStateDetector:
    """
    Detects game state and UI elements for better integration.
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
        
    def preprocess_for_model(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess captured frame for model input."""
        if frame is None:
            return None
        
        # Crop to game area
        game_frame = frame[
            self.config.game_area_y:self.config.game_area_y + self.config.game_area_height,
            self.config.game_area_x:self.config.game_area_x + self.config.game_area_width
        ]
        
        # Resize to model input size
        resized = cv2.resize(game_frame, self.config.model_input_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Ensure correct data type and range
        return rgb_frame.astype(np.uint8)
    
    def detect_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect current game state from frame."""
        # Placeholder for game state detection
        # Real implementation would use computer vision to detect:
        # - Battle phase (preparation, battle, overtime)
        # - Elixir count
        # - Cards in hand
        # - Tower health
        # - etc.
        
        return {
            "in_battle": True,  # Placeholder
            "elixir": 10,       # Placeholder
            "phase": "battle"   # Placeholder
        }

def setup_bluestacks_integration():
    """Setup and test BlueStacks integration."""
    print("🔧 SETTING UP BLUESTACKS INTEGRATION")
    print("=" * 50)
    
    # Create configuration
    config = GameConfig()
    
    # Test ADB connection
    print("1. Testing ADB connection...")
    controller = BlueStacksController(config)
    
    if not controller.connect():
        print("❌ Failed to connect to BlueStacks")
        print("   Make sure BlueStacks is running and ADB is installed")
        return False
    
    # Test screenshot
    print("2. Testing screenshot capture...")
    frame = controller.capture_screen()
    if frame is not None:
        print(f"✓ Screenshot captured: {frame.shape}")
        cv2.imwrite("test_screenshot.png", frame)
        print("   Saved as: test_screenshot.png")
    else:
        print("❌ Screenshot failed")
        return False
    
    # Test action mapping
    print("3. Testing action mapping...")
    mapper = ActionMapper(config)
    test_action = 1000
    decoded = mapper.decode_model_action(test_action)
    screen_x, screen_y = mapper.map_to_screen_coordinates(decoded["x"], decoded["y"])
    
    print(f"   Action {test_action} -> x={decoded['x']}, y={decoded['y']}, card={decoded['card']}")
    print(f"   Screen coordinates: ({screen_x}, {screen_y})")
    
    # Test preprocessing
    print("4. Testing frame preprocessing...")
    detector = GameStateDetector(config)
    processed = detector.preprocess_for_model(frame)
    if processed is not None:
        print(f"✓ Preprocessing successful: {processed.shape}")
        cv2.imwrite("processed_frame.png", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    controller.disconnect()
    
    print("\n✅ BlueStacks integration setup complete!")
    return True

if __name__ == "__main__":
    import io
    
    # Run setup and tests
    setup_bluestacks_integration()
