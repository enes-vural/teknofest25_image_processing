import cv2
import numpy as np
import time
from collections import deque
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging
from datetime import datetime
import os

# =================== ENUMS & DATA CLASSES ===================

class ShapeType(Enum):
    """Şekil tipleri"""
    TRIANGLE = "triangle"
    SQUARE = "square"
    HEXAGON = "hexagon"

class ColorType(Enum):
    """Renk tipleri"""
    RED = "red"
    BLUE = "blue"

@dataclass
class DetectedShape:
    """Algılanan şekil bilgileri"""
    shape_type: ShapeType
    color: ColorType
    center_x: int
    center_y: int
    width: int
    height: int
    area: float
    confidence: float
    distance_to_center: float  # Frame merkezine uzaklık
    timestamp: float
    contour: np.ndarray = None
    
    def __str__(self):
        return f"{self.color.value} {self.shape_type.value} at ({self.center_x}, {self.center_y})"

# =================== MAIN SHAPE DETECTOR CLASS ===================

class ShapeDetector:
    """Thread-safe shape detector with easy access to detection results"""
    
    def __init__(self, width=640, height=480, log_enabled=True):
        # Frame dimensions
        self.frame_width = width
        self.frame_height = height
        self.frame_center_x = width // 2
        self.frame_center_y = height // 2
        
        # Logging setup
        self.log_enabled = log_enabled
        if log_enabled:
            self._setup_logging()
        
        # Color ranges in HSV
        self.color_ranges = {
            ColorType.RED: [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            ColorType.BLUE: [
                (np.array([100, 100, 100]), np.array([140, 255, 255]))
            ]
        }
        
        # Detection parameters
        self.min_contour_area = 500
        self.approx_epsilon = 0.025
        self.proximity_threshold = 0.4  # Frame alanının %40'ı
        
        # Morphological kernel
        self.kernel = np.ones((3, 3), np.uint8)
        
        # Thread-safe detection results
        self._lock = threading.Lock()
        self._reset_detection_results()
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=10)
        self.prev_time = time.time()
        self.current_fps = 0.0
        
        # Statistics
        self.total_frames_processed = 0
        self.total_shapes_detected = 0
        
    def _setup_logging(self):
        """Setup logging"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        log_file = f"logs/shape_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Shape Detector initialized")
    
    def _reset_detection_results(self):
        """Reset all detection results"""
        self.detected_shapes = {
            'red_triangle': [],
            'red_square': [],
            'blue_square': [],
            'blue_hexagon': []
        }
        self.red_proximity_alert = False
        self.blue_proximity_alert = False
        self.closest_shape = None
        self.last_update_time = time.time()
    
    def _log(self, message, level=logging.INFO):
        """Safe logging"""
        if self.log_enabled and hasattr(self, 'logger'):
            self.logger.log(level, message)
    
    # =================== MAIN PROCESSING ===================
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a frame and update detection results
        
        Args:
            frame: BGR format OpenCV frame
            
        Returns:
            bool: True if processing successful
        """
        if frame is None:
            return False
        
        start_time = time.time()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color masks
        masks = self._create_color_masks(hsv)
        
        # Check proximity
        self._check_proximity(masks, frame.shape)
        
        # Detect shapes
        detected = self._detect_shapes(masks, frame)
        
        # Update FPS
        self._update_fps()
        
        # Thread-safe update of results
        with self._lock:
            self._reset_detection_results()
            
            # Organize detected shapes
            for shape in detected:
                key = f"{shape.color.value}_{shape.shape_type.value}"
                if key in self.detected_shapes:
                    self.detected_shapes[key].append(shape)
            
            # Find closest shape
            if detected:
                self.closest_shape = min(detected, key=lambda s: s.distance_to_center)
            
            self.last_update_time = time.time()
        
        # Update statistics
        self.total_frames_processed += 1
        self.total_shapes_detected += len(detected)
        
        # Log detections
        if detected:
            self._log(f"Detected {len(detected)} shapes: {[str(s) for s in detected]}")
        
        return True
    
    def _create_color_masks(self, hsv: np.ndarray) -> Dict[ColorType, np.ndarray]:
        """Create color masks"""
        masks = {}
        
        # Red mask (two ranges)
        red_mask1 = cv2.inRange(hsv, *self.color_ranges[ColorType.RED][0])
        red_mask2 = cv2.inRange(hsv, *self.color_ranges[ColorType.RED][1])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        masks[ColorType.RED] = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Blue mask
        blue_mask = cv2.inRange(hsv, *self.color_ranges[ColorType.BLUE][0])
        masks[ColorType.BLUE] = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, self.kernel)
        
        return masks
    
    def _check_proximity(self, masks: Dict[ColorType, np.ndarray], frame_shape: tuple):
        """Check proximity alerts"""
        frame_area = frame_shape[0] * frame_shape[1]
        threshold = frame_area * self.proximity_threshold
        
        with self._lock:
            self.red_proximity_alert = np.sum(masks[ColorType.RED] > 0) > threshold
            self.blue_proximity_alert = np.sum(masks[ColorType.BLUE] > 0) > threshold
        
        if self.red_proximity_alert:
            self._log("RED PROXIMITY ALERT!", logging.WARNING)
        if self.blue_proximity_alert:
            self._log("BLUE PROXIMITY ALERT!", logging.WARNING)
    
    def _detect_shapes(self, masks: Dict[ColorType, np.ndarray], 
                      frame: np.ndarray) -> List[DetectedShape]:
        """Detect shapes in masks"""
        all_detected = []
        
        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                shape = self._analyze_contour(contour, color, frame)
                if shape:
                    all_detected.append(shape)
        
        return all_detected
    
    def _analyze_contour(self, contour: np.ndarray, color: ColorType, 
                        frame: np.ndarray) -> Optional[DetectedShape]:
        """Analyze contour and determine shape type"""
        area = cv2.contourArea(contour)
        if area < self.min_contour_area:
            return None
        
        # Polygon approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
        corners = len(approx)
        
        # Bounding box and center
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Distance to frame center
        distance = np.sqrt((cx - self.frame_center_x)**2 + 
                          (cy - self.frame_center_y)**2)
        
        # Determine shape type
        shape_type = None
        confidence = 0.0
        
        if color == ColorType.RED:
            if corners == 3:
                shape_type = ShapeType.TRIANGLE
                confidence = 0.9
            elif corners == 4:
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3:
                    shape_type = ShapeType.SQUARE
                    confidence = 0.85
        
        elif color == ColorType.BLUE:
            if corners == 4:
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3:
                    shape_type = ShapeType.SQUARE
                    confidence = 0.85
            elif corners == 6:
                if self._is_regular_hexagon(approx, x, y, w, h):
                    shape_type = ShapeType.HEXAGON
                    confidence = 0.9
        
        if shape_type:
            return DetectedShape(
                shape_type=shape_type,
                color=color,
                center_x=cx,
                center_y=cy,
                width=w,
                height=h,
                area=area,
                confidence=confidence,
                distance_to_center=distance,
                timestamp=time.time(),
                contour=contour
            )
        
        return None
    
    def _is_regular_hexagon(self, approx: np.ndarray, x: int, y: int, 
                           w: int, h: int) -> bool:
        """Check if shape is regular hexagon"""
        if len(approx) != 6:
            return False
        
        # Center point
        cx, cy = x + w / 2, y + h / 2
        
        # Check distances from vertices to center
        distances = []
        for point in approx:
            px, py = point[0]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            distances.append(dist)
        
        # Check uniformity
        mean_dist = np.mean(distances)
        dist_tolerance = mean_dist * 0.15
        
        if not all(abs(d - mean_dist) < dist_tolerance for d in distances):
            return False
        
        # Check aspect ratio
        aspect_ratio = max(w, h) / min(w, h)
        return aspect_ratio < 1.3
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_buffer.append(fps)
            self.prev_time = current_time
            self.current_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
    
    # =================== PUBLIC INTERFACE - EASY ACCESS METHODS ===================
    
    def has_red_triangle(self) -> bool:
        """Check if red triangle is detected"""
        with self._lock:
            return len(self.detected_shapes['red_triangle']) > 0
    
    def has_red_square(self) -> bool:
        """Check if red square is detected"""
        with self._lock:
            return len(self.detected_shapes['red_square']) > 0
    
    def has_blue_square(self) -> bool:
        """Check if blue square is detected"""
        with self._lock:
            return len(self.detected_shapes['blue_square']) > 0
    
    def has_blue_hexagon(self) -> bool:
        """Check if blue hexagon is detected"""
        with self._lock:
            return len(self.detected_shapes['blue_hexagon']) > 0
    
    def get_red_triangles(self) -> List[DetectedShape]:
        """Get all detected red triangles"""
        with self._lock:
            return self.detected_shapes['red_triangle'].copy()
    
    def get_red_squares(self) -> List[DetectedShape]:
        """Get all detected red squares"""
        with self._lock:
            return self.detected_shapes['red_square'].copy()
    
    def get_blue_squares(self) -> List[DetectedShape]:
        """Get all detected blue squares"""
        with self._lock:
            return self.detected_shapes['blue_square'].copy()
    
    def get_blue_hexagons(self) -> List[DetectedShape]:
        """Get all detected blue hexagons"""
        with self._lock:
            return self.detected_shapes['blue_hexagon'].copy()
    
    def get_closest_shape(self) -> Optional[DetectedShape]:
        """Get closest shape to frame center"""
        with self._lock:
            return self.closest_shape
    
    def get_all_shapes(self) -> List[DetectedShape]:
        """Get all detected shapes"""
        with self._lock:
            all_shapes = []
            for shapes_list in self.detected_shapes.values():
                all_shapes.extend(shapes_list)
            return all_shapes
    
    def is_red_close(self) -> bool:
        """Check if red object is very close (proximity alert)"""
        with self._lock:
            return self.red_proximity_alert
    
    def is_blue_close(self) -> bool:
        """Check if blue object is very close (proximity alert)"""
        with self._lock:
            return self.blue_proximity_alert
    
    def get_shape_position(self, shape_type: str) -> Optional[Tuple[int, int]]:
        """
        Get position of specific shape type
        Args:
            shape_type: 'red_triangle', 'red_square', 'blue_square', 'blue_hexagon'
        Returns:
            (x, y) position or None
        """
        with self._lock:
            if shape_type in self.detected_shapes and self.detected_shapes[shape_type]:
                shape = self.detected_shapes[shape_type][0]  # First detected
                return (shape.center_x, shape.center_y)
        return None
    
    def get_direction_to_center(self, shape_type: str) -> Optional[Tuple[float, float]]:
        """
        Get normalized direction from shape to frame center
        Returns: (dx, dy) normalized between -1 and 1
        """
        pos = self.get_shape_position(shape_type)
        if pos:
            dx = (pos[0] - self.frame_center_x) / self.frame_center_x
            dy = (pos[1] - self.frame_center_y) / self.frame_center_y
            return (dx, dy)
        return None
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        with self._lock:
            return {
                'total_frames': self.total_frames_processed,
                'total_detections': self.total_shapes_detected,
                'current_fps': self.current_fps,
                'shapes_in_frame': len(self.get_all_shapes()),
                'red_triangles': len(self.detected_shapes['red_triangle']),
                'red_squares': len(self.detected_shapes['red_square']),
                'blue_squares': len(self.detected_shapes['blue_square']),
                'blue_hexagons': len(self.detected_shapes['blue_hexagon']),
                'red_proximity': self.red_proximity_alert,
                'blue_proximity': self.blue_proximity_alert
            }
    
    def draw_overlay(self, frame: np.ndarray, show_info: bool = True) -> np.ndarray:
        """Draw detection results on frame"""
        output = frame.copy()
        
        with self._lock:
            # Draw all shapes
            for shape in self.get_all_shapes():
                color_bgr = (0, 0, 255) if shape.color == ColorType.RED else (255, 0, 0)
                
                # Draw contour
                if shape.contour is not None:
                    cv2.drawContours(output, [shape.contour], -1, color_bgr, 2)
                
                # Draw label
                label = f"{shape.color.value} {shape.shape_type.value}"
                cv2.putText(output, label, (shape.center_x - 30, shape.center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center point
                cv2.circle(output, (shape.center_x, shape.center_y), 3, color_bgr, -1)
        
        # Draw info panel
        if show_info:
            stats = self.get_stats()
            info_text = [
                f"FPS: {stats['current_fps']:.1f}",
                f"Shapes: {stats['shapes_in_frame']}",
            ]
            
            if stats['red_proximity']:
                info_text.append("RED CLOSE!")
            if stats['blue_proximity']:
                info_text.append("BLUE CLOSE!")
            
            y_offset = 30
            for text in info_text:
                cv2.putText(output, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        # Draw center crosshair
        cv2.line(output, (self.frame_center_x - 20, self.frame_center_y), 
                (self.frame_center_x + 20, self.frame_center_y), (0, 255, 0), 1)
        cv2.line(output, (self.frame_center_x, self.frame_center_y - 20), 
                (self.frame_center_x, self.frame_center_y + 20), (0, 255, 0), 1)
        
        return output


# =================== USAGE EXAMPLE ===================

def example_usage():
    """Example usage showing how to use the detector"""
    
    # Initialize detector
    detector = ShapeDetector(width=640, height=480, log_enabled=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)  # or video file path
    
    print("Shape Detection System")
    print("Press 'q' to quit")
    print("\n=== Easy Access Methods ===")
    print("detector.has_red_triangle()")
    print("detector.has_red_square()")
    print("detector.has_blue_square()")
    print("detector.has_blue_hexagon()")
    print("detector.is_red_close()")
    print("detector.is_blue_close()")
    print("detector.get_closest_shape()")
    print("detector.get_shape_position('red_triangle')")
    print("===========================\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detector.process_frame(frame)
        
        # === EASY ACCESS FOR DRONE ALGORITHM ===
        
        # Check specific shapes
        if detector.has_red_triangle():
            print("RED TRIANGLE DETECTED!")
            # Drone action here
        
        if detector.has_blue_hexagon():
            print("BLUE HEXAGON DETECTED!")
            # Drone action here
        
        # Check proximity
        if detector.is_red_close():
            print("WARNING: Red object very close!")
        
        # Get closest shape for targeting
        closest = detector.get_closest_shape()
        if closest:
            print(f"Closest: {closest}")
        
        # Get specific shape position for navigation
        red_tri_pos = detector.get_shape_position('red_triangle')
        if red_tri_pos:
            print(f"Red triangle at: {red_tri_pos}")
        
        # Get direction for drone movement
        direction = detector.get_direction_to_center('blue_hexagon')
        if direction:
            dx, dy = direction
            print(f"Blue hexagon direction: dx={dx:.2f}, dy={dy:.2f}")
        
        # === END OF DRONE ALGORITHM SECTION ===
        
        # Draw overlay for visualization
        display_frame = detector.draw_overlay(frame)
        
        # Show frame
        cv2.imshow('Shape Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\nFinal Statistics:")
    for key, value in detector.get_stats().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()