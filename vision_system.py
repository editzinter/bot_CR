#!/usr/bin/env python3
"""
Real-time Vision System for Clash Royale
This module implements advanced computer vision for game state detection,
frame preprocessing, and real-time analysis of Clash Royale gameplay.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import threading
import queue
import json

@dataclass
class VisionConfig:
    """Configuration for vision system."""
    target_fps: int = 30
    frame_buffer_size: int = 10
    model_input_size: Tuple[int, int] = (128, 128)
    detection_confidence: float = 0.7
    template_match_threshold: float = 0.8
    
class FrameProcessor:
    """
    Advanced frame processing for Clash Royale gameplay.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.frame_buffer = deque(maxsize=config.frame_buffer_size)
        self.templates = {}
        self.load_templates()
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        
        # Kalman filters for tracking
        self.trackers = {}
        
    def load_templates(self):
        """Load template images for UI element detection."""
        # In a real implementation, these would be actual template images
        # For now, we'll create placeholder templates
        template_names = [
            "elixir_bar", "card_slot", "tower", "king_tower",
            "battle_button", "end_battle", "victory", "defeat"
        ]
        
        for name in template_names:
            # Create placeholder templates (would load actual images)
            self.templates[name] = np.zeros((50, 50, 3), dtype=np.uint8)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame to match model training format.
        """
        if frame is None:
            return None
        
        # Resize to model input size
        processed = cv2.resize(frame, self.config.model_input_size)
        
        # Convert BGR to RGB (model expects RGB)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 255] range (ensure uint8)
        processed = processed.astype(np.uint8)
        
        return processed
    
    def detect_ui_elements(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect UI elements using template matching and contour detection.
        """
        detections = {}
        
        if frame is None:
            return detections
        
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Template matching for UI elements
        for template_name, template in self.templates.items():
            if template.size == 0:  # Skip empty templates
                continue
                
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            locations = np.where(result >= self.config.template_match_threshold)
            
            if len(locations[0]) > 0:
                detections[template_name] = {
                    "found": True,
                    "locations": list(zip(locations[1], locations[0])),
                    "confidence": float(np.max(result))
                }
            else:
                detections[template_name] = {"found": False}
        
        return detections
    
    def detect_elixir_count(self, frame: np.ndarray) -> int:
        """
        Detect current elixir count using computer vision.
        """
        # Placeholder implementation
        # Real implementation would:
        # 1. Locate elixir bar region
        # 2. Count purple elixir drops
        # 3. Use OCR for numerical display
        
        # For now, return a mock value
        return np.random.randint(0, 11)
    
    def detect_cards_in_hand(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cards currently in hand.
        """
        cards = []
        
        # Placeholder implementation
        # Real implementation would:
        # 1. Locate card area at bottom of screen
        # 2. Segment individual card regions
        # 3. Use template matching or CNN to identify cards
        # 4. Detect if cards are playable (enough elixir)
        
        for i in range(4):
            cards.append({
                "slot": i,
                "card_type": "unknown",
                "playable": True,
                "elixir_cost": np.random.randint(1, 9)
            })
        
        return cards
    
    def detect_towers(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect tower positions and health.
        """
        towers = {
            "player": {
                "left_tower": {"health": 100, "position": (0, 0), "alive": True},
                "right_tower": {"health": 100, "position": (0, 0), "alive": True},
                "king_tower": {"health": 100, "position": (0, 0), "alive": True}
            },
            "opponent": {
                "left_tower": {"health": 100, "position": (0, 0), "alive": True},
                "right_tower": {"health": 100, "position": (0, 0), "alive": True},
                "king_tower": {"health": 100, "position": (0, 0), "alive": True}
            }
        }
        
        # Placeholder - real implementation would use computer vision
        return towers
    
    def detect_units_on_field(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect units currently on the battlefield.
        """
        units = []
        
        # Use background subtraction to detect moving objects
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                
                units.append({
                    "position": (x + w//2, y + h//2),
                    "bounding_box": (x, y, w, h),
                    "area": area,
                    "type": "unknown",
                    "team": "unknown"
                })
        
        return units
    
    def analyze_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive game state analysis.
        """
        analysis = {
            "timestamp": time.time(),
            "frame_shape": frame.shape if frame is not None else None,
            "ui_elements": self.detect_ui_elements(frame),
            "elixir_count": self.detect_elixir_count(frame),
            "cards_in_hand": self.detect_cards_in_hand(frame),
            "towers": self.detect_towers(frame),
            "units_on_field": self.detect_units_on_field(frame),
            "battle_phase": self.detect_battle_phase(frame)
        }
        
        return analysis
    
    def detect_battle_phase(self, frame: np.ndarray) -> str:
        """
        Detect current battle phase (preparation, battle, overtime, end).
        """
        # Placeholder implementation
        # Real implementation would analyze UI elements and timers
        phases = ["preparation", "battle", "overtime", "victory", "defeat"]
        return np.random.choice(phases)

class RealTimeVisionSystem:
    """
    Real-time vision system with threading and performance optimization.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.processor = FrameProcessor(config)
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Frame statistics
        self.frame_stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "dropped_frames": 0,
            "avg_processing_time": 0
        }
    
    def start_processing(self):
        """Start real-time processing thread."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        print("✓ Real-time vision processing started")
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("✓ Real-time vision processing stopped")
    
    def _processing_loop(self):
        """Main processing loop running in background thread."""
        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process frame
                processed_frame = self.processor.preprocess_frame(frame)
                game_state = self.processor.analyze_game_state(frame)
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.frame_stats["processed_frames"] += 1
                self.frame_stats["avg_processing_time"] = (
                    self.frame_stats["avg_processing_time"] * 0.9 + 
                    processing_time * 0.1
                )
                
                # Put result in output queue
                result = {
                    "processed_frame": processed_frame,
                    "game_state": game_state,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
                
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Processing error: {e}")
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """Add frame for processing."""
        if not self.running:
            return False
        
        self.frame_stats["total_frames"] += 1
        
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            # Drop oldest frame if queue is full
            self.frame_stats["dropped_frames"] += 1
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                return True
            except queue.Empty:
                return False
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get latest processing result."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "current_fps": self.current_fps,
            "target_fps": self.config.target_fps,
            "frame_stats": self.frame_stats.copy(),
            "avg_processing_time_ms": self.frame_stats["avg_processing_time"] * 1000,
            "queue_sizes": {
                "input_queue": self.frame_queue.qsize(),
                "output_queue": self.result_queue.qsize()
            }
        }

def test_vision_system():
    """Test the vision system with sample data."""
    print("🔍 TESTING VISION SYSTEM")
    print("=" * 40)
    
    config = VisionConfig()
    vision_system = RealTimeVisionSystem(config)
    
    # Create test frame
    test_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    
    print("1. Testing frame preprocessing...")
    processed = vision_system.processor.preprocess_frame(test_frame)
    if processed is not None:
        print(f"✓ Preprocessing successful: {processed.shape}")
    
    print("2. Testing game state analysis...")
    game_state = vision_system.processor.analyze_game_state(test_frame)
    print(f"✓ Game state analysis complete: {len(game_state)} components")
    
    print("3. Testing real-time processing...")
    vision_system.start_processing()
    
    # Add test frames
    for i in range(10):
        vision_system.add_frame(test_frame)
        time.sleep(0.1)
    
    # Get results
    time.sleep(1)
    result = vision_system.get_latest_result()
    if result:
        print(f"✓ Real-time processing working")
        print(f"  Processing time: {result['processing_time']*1000:.2f} ms")
    
    # Get performance stats
    stats = vision_system.get_performance_stats()
    print(f"✓ Performance stats: {stats['current_fps']} FPS")
    
    vision_system.stop_processing()
    
    print("\n✅ Vision system test complete!")

if __name__ == "__main__":
    test_vision_system()
