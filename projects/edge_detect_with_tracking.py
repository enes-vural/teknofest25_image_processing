import cv2
import numpy as np
import time
from collections import deque
import threading
import queue
import os
import atexit
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# Enable OpenCV optimizations (uses NEON SIMD instructions on ARM if available)
cv2.setUseOptimized(True)

# Ensure OpenCV is using optimized code paths
if cv2.useOptimized():
    print("OpenCV optimizations enabled (using hardware acceleration)")
else:
    print("WARNING: OpenCV optimizations not available")

# Set Raspberry Pi to performance mode (requires appropriate permissions)
try:
    os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")
    print("Set CPU to performance mode")
    
    # Reset to ondemand governor on exit
    def reset_governor():
        os.system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")
    
    atexit.register(reset_governor)
except:
    print("Could not set CPU governor (may need sudo)")


@dataclass
class TrackedObject:
    """Data class to store tracked object information."""
    id: int
    shape_type: str
    color: str
    center: Tuple[int, int]
    contour: np.ndarray
    area: float
    last_seen: float
    track_history: deque
    confidence: float = 1.0
    tracking_state: str = "active"  # active, lost, recovered
    lost_frames: int = 0
    
    def update_position(self, new_center: Tuple[int, int], new_contour: np.ndarray, new_area: float):
        """Update object position and tracking history."""
        self.center = new_center
        self.contour = new_contour
        self.area = new_area
        self.last_seen = time.time()
        self.track_history.append(new_center)
        self.lost_frames = 0
        self.tracking_state = "recovered" if self.tracking_state == "lost" else "active"
        
        # Update confidence based on tracking consistency
        if len(self.track_history) >= 2:
            # Calculate movement consistency
            recent_moves = list(self.track_history)[-3:]
            if len(recent_moves) >= 2:
                distances = [math.sqrt((recent_moves[i+1][0] - recent_moves[i][0])**2 + 
                                     (recent_moves[i+1][1] - recent_moves[i][1])**2) 
                           for i in range(len(recent_moves)-1)]
                avg_distance = np.mean(distances)
                # Higher confidence for consistent movement
                self.confidence = min(1.0, self.confidence + 0.1) if avg_distance < 50 else max(0.3, self.confidence - 0.05)
    
    def mark_as_lost(self):
        """Mark object as lost and increment lost frames counter."""
        self.lost_frames += 1
        self.tracking_state = "lost"
        self.confidence = max(0.1, self.confidence - 0.1)


class ObjectTracker:
    """Advanced object tracker with fail-safe mechanisms."""
    
    def __init__(self, max_disappeared_frames=30, max_tracking_distance=100):
        self.next_object_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_disappeared_frames = max_disappeared_frames
        self.max_tracking_distance = max_tracking_distance
        self.tracking_history_size = 10
        
        # Kalman filter parameters for prediction
        self.enable_prediction = True
        self.kalman_filters = {}
        
    def create_kalman_filter(self):
        """Create a Kalman filter for position prediction."""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        return kalman
    
    def predict_position(self, obj_id: int) -> Optional[Tuple[int, int]]:
        """Predict next position using Kalman filter."""
        if not self.enable_prediction or obj_id not in self.kalman_filters:
            return None
            
        kalman = self.kalman_filters[obj_id]
        prediction = kalman.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def update_kalman(self, obj_id: int, position: Tuple[int, int]):
        """Update Kalman filter with new position."""
        if obj_id not in self.kalman_filters:
            self.kalman_filters[obj_id] = self.create_kalman_filter()
            # Initialize with first position
            kalman = self.kalman_filters[obj_id]
            kalman.statePre = np.array([position[0], position[1], 0, 0], dtype=np.float32)
            kalman.statePost = np.array([position[0], position[1], 0, 0], dtype=np.float32)
        
        kalman = self.kalman_filters[obj_id]
        measurement = np.array([[position[0]], [position[1]]], dtype=np.float32)
        kalman.correct(measurement)
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_shape_similarity(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """Calculate shape similarity using Hu moments."""
        try:
            # Calculate Hu moments for both contours
            moments1 = cv2.moments(contour1)
            moments2 = cv2.moments(contour2)
            
            if moments1['m00'] == 0 or moments2['m00'] == 0:
                return 0.0
                
            hu1 = cv2.HuMoments(moments1).flatten()
            hu2 = cv2.HuMoments(moments2).flatten()
            
            # Calculate similarity (lower is more similar)
            similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            return max(0.0, 1.0 - similarity)  # Convert to similarity score (0-1)
        except:
            return 0.0
    
    def match_detections(self, detected_shapes: List[Tuple]) -> Dict[int, Tuple]:
        """Match detected shapes with existing tracked objects."""
        if not detected_shapes:
            return {}
            
        matches = {}
        unmatched_detections = list(range(len(detected_shapes)))
        
        # For each tracked object, find the best matching detection
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.tracking_state == "lost" and tracked_obj.lost_frames > self.max_disappeared_frames:
                continue
                
            best_match_idx = None
            best_score = 0.0
            
            # Predict position if object was lost
            predicted_pos = None
            if tracked_obj.tracking_state == "lost" and self.enable_prediction:
                predicted_pos = self.predict_position(obj_id)
            
            for idx in unmatched_detections:
                contour, shape_type, color = detected_shapes[idx]
                
                # Calculate center of detected shape
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                    
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                area = cv2.contourArea(contour)
                
                # Check color and shape type match
                if tracked_obj.color != color or tracked_obj.shape_type != shape_type:
                    continue
                
                # Calculate position distance
                reference_pos = predicted_pos if predicted_pos else tracked_obj.center
                position_distance = self.calculate_distance(center, reference_pos)
                
                if position_distance > self.max_tracking_distance:
                    continue
                
                # Calculate shape similarity
                shape_similarity = self.calculate_shape_similarity(tracked_obj.contour, contour)
                
                # Calculate area similarity
                area_ratio = min(area, tracked_obj.area) / max(area, tracked_obj.area)
                
                # Combined matching score
                position_score = max(0, 1.0 - position_distance / self.max_tracking_distance)
                combined_score = (position_score * 0.6 + shape_similarity * 0.2 + area_ratio * 0.2)
                
                if combined_score > best_score and combined_score > 0.3:  # Minimum threshold
                    best_score = combined_score
                    best_match_idx = idx
            
            if best_match_idx is not None:
                matches[obj_id] = detected_shapes[best_match_idx]
                unmatched_detections.remove(best_match_idx)
        
        return matches
    
    def update(self, detected_shapes: List[Tuple]) -> List[TrackedObject]:
        """Update tracker with new detections."""
        current_time = time.time()
        
        # Match detections with existing tracked objects
        matches = self.match_detections(detected_shapes)
        
        # Update matched objects
        for obj_id, (contour, shape_type, color) in matches.items():
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                area = cv2.contourArea(contour)
                
                self.tracked_objects[obj_id].update_position(center, contour, area)
                self.update_kalman(obj_id, center)
        
        # Mark unmatched tracked objects as lost
        matched_ids = set(matches.keys())
        for obj_id in self.tracked_objects:
            if obj_id not in matched_ids:
                self.tracked_objects[obj_id].mark_as_lost()
        
        # Create new tracked objects for unmatched detections
        matched_detection_indices = set()
        for contour, shape_type, color in matches.values():
            for i, (det_contour, det_shape, det_color) in enumerate(detected_shapes):
                if (np.array_equal(contour, det_contour) and 
                    shape_type == det_shape and color == det_color):
                    matched_detection_indices.add(i)
                    break
        
        for i, (contour, shape_type, color) in enumerate(detected_shapes):
            if i not in matched_detection_indices:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    area = cv2.contourArea(contour)
                    
                    new_obj = TrackedObject(
                        id=self.next_object_id,
                        shape_type=shape_type,
                        color=color,
                        center=center,
                        contour=contour,
                        area=area,
                        last_seen=current_time,
                        track_history=deque(maxlen=self.tracking_history_size)
                    )
                    new_obj.track_history.append(center)
                    
                    self.tracked_objects[self.next_object_id] = new_obj
                    self.update_kalman(self.next_object_id, center)
                    self.next_object_id += 1
        
        # Remove objects that have been lost for too long
        objects_to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if obj.lost_frames > self.max_disappeared_frames:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.kalman_filters:
                del self.kalman_filters[obj_id]
        
        return list(self.tracked_objects.values())


class AsyncVideoCapture:
    """Asynchronous video capture class to decouple frame grabbing from processing."""
    
    def __init__(self, src=0, queue_size=2):
        # Initialize the camera
        self.cap = cv2.VideoCapture(src)
        
        # Essential camera settings for Raspberry Pi performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        #if camera supports 30fps, set it to 30fps
        self.cap.set(cv2.CAP_PROP_FPS, 30) 
        
        # Create frame queue with minimal size to avoid memory buildup
        self.queue = queue.Queue(maxsize=queue_size)
        
        # Flag to control the thread
        self.running = True
        
        # Start the thread for frame capture
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        """Background thread function to continuously grab frames."""
        while self.running:
            # Grab frame without decoding (fast operation)
            self.cap.grab()
            
            # Only retrieve and queue the frame if there's room
            if not self.queue.full():
                ret, frame = self.cap.retrieve()
                if ret:
                    # If queue is full, remove oldest frame
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            pass
                    # Put the new frame in the queue
                    self.queue.put(frame)
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.001)
    
    def read(self):
        """Read a frame from the queue."""
        try:
            # Get frame with timeout to avoid blocking indefinitely
            frame = self.queue.get(timeout=1.0)
            return True, frame
        except (queue.Empty, AttributeError):
            # Return false if no frame is available
            return False, None
    
    def release(self):
        """Release resources."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class ShapeDetector:
    """Class for detecting specific colored geometric shapes optimized for Raspberry Pi."""
    
    def __init__(self):
        # Pre-compute constants and allocate buffers to avoid repeated memory allocations
        
        # Color boundaries in HSV space (lower, upper) - optimized for Raspberry Pi camera 
        self.red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
        self.red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([160, 100, 100], dtype=np.uint8)  # Red wraps around HSV space
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self.blue_lower = np.array([100, 100, 100], dtype=np.uint8)
        self.blue_upper = np.array([140, 255, 255], dtype=np.uint8)
        
        # Detection thresholds
        self.min_contour_area = 200  # Smaller minimum area for reduced resolution
        self.approx_polygon_epsilon = 0.025
        
        # Fast morphological kernels (pre-computed)
        self.kernel3 = np.ones((3, 3), np.uint8)  # Smaller kernel for better performance
        
        # Reuse memory buffers for intermediate results
        self.hsv_buffer = None
        self.red_mask = None
        self.blue_mask = None
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=10)  # Smaller buffer for faster adaptation
        self.prev_frame_time = 0
        
        # Color intensity thresholds
        self.color_intensity_threshold = 0.4
        
        # Skip frames counter (for periodic processing)
        self.frame_count = 0
        self.process_every_n_frames = 1  # Process every frame by default
        
        # Initialize object tracker
        self.tracker = ObjectTracker(max_disappeared_frames=30, max_tracking_distance=100)

    def preprocess_frame(self, frame):
        """Apply optimized preprocessing for Raspberry Pi."""
        # Resize if needed (improves performance with minimal impact on detection)
        # Already handling 320x240 input from the AsyncVideoCapture
        
        # Allocate memory buffers if not already done
        if self.hsv_buffer is None or self.hsv_buffer.shape[:2] != frame.shape[:2]:
            h, w = frame.shape[:2]
            self.hsv_buffer = np.empty((h, w, 3), dtype=np.uint8)
            self.red_mask = np.empty((h, w), dtype=np.uint8)
            self.blue_mask = np.empty((h, w), dtype=np.uint8)
        
        # Use medianBlur instead of GaussianBlur (faster on Raspberry Pi with NEON)
        blurred = cv2.medianBlur(frame, 3)  # 3x3 median is faster than Gaussian
        
        # Convert to HSV - efficient in-place conversion
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV, dst=self.hsv_buffer)
        
        # Skip CLAHE for performance (only use if lighting varies significantly)
        # CLAHE is computationally expensive on Raspberry Pi
        
        return hsv

    def create_color_masks(self, hsv_frame):
        """Create optimized binary masks for red and blue colors."""
        # Create red masks - using pre-allocated buffers
        red_mask1 = cv2.inRange(hsv_frame, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_frame, self.red_lower2, self.red_upper2)
        
        # Efficiently combine masks
        cv2.bitwise_or(red_mask1, red_mask2, dst=self.red_mask)
        
        # Create blue mask - using pre-allocated buffer
        blue_mask = cv2.inRange(hsv_frame, self.blue_lower, self.blue_upper)
        
        # Use smaller kernel (3x3) and fewer morphological operations
        # Replace morphologyEx with dilate (faster on Raspberry Pi)
        cv2.dilate(self.red_mask, self.kernel3, dst=self.red_mask, iterations=1)
        cv2.dilate(blue_mask, self.kernel3, dst=self.blue_mask, iterations=1)
        
        return self.red_mask, self.blue_mask

    def detect_shapes(self, frame):
        """Detect shapes with frame skipping for better performance."""
        # Increment frame counter
        self.frame_count += 1
        
        # Skip frames if needed (process_every_n_frames can be adjusted dynamically)
        if self.frame_count % self.process_every_n_frames != 0:
            # Return empty list of shapes for skipped frames
            return []
        
        # Process frame normally
        hsv_frame = self.preprocess_frame(frame)
        red_mask, blue_mask = self.create_color_masks(hsv_frame)
        
        # Quick check for "very close" cases using vectorized operations
        frame_area = frame.shape[0] * frame.shape[1]
        threshold_pixels = frame_area * self.color_intensity_threshold
        
        # Fast sum using numpy (avoid OpenCV countNonZero which is slower)
        red_pixels = np.sum(red_mask == 255)
        blue_pixels = np.sum(blue_mask == 255)
        
        red_close = red_pixels > threshold_pixels
        blue_close = blue_pixels > threshold_pixels
        
        if red_close:
            print("Red square/triangle is very close – inside the shape.")
        if blue_close:
            print("Blue square/hexagon is very close – inside the shape.")
        
        # Fast contour finding with simple approximation
        red_shapes = self.process_contours(red_mask, frame, "red")
        blue_shapes = self.process_contours(blue_mask, frame, "blue")
        
        return red_shapes + blue_shapes

    def process_contours(self, mask, frame, color):
        """Process contours with optimized algorithm for Raspberry Pi."""
        detected_shapes = []
        
        # Use RETR_EXTERNAL and CHAIN_APPROX_SIMPLE for better performance
        # Chain approx simple reduces the number of points in contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Early return if no contours (avoid unnecessary processing)
        if not contours:
            return []
        
        # Sort by area and limit to top N contours for performance
        if len(contours) > 5:  # Only sort if necessary
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        # Process each contour
        for contour in contours:
            # Fast area check
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Fast approximation to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_polygon_epsilon * peri, True)
            corners = len(approx)
            
            # Fast shape identification with early returns
            shape_type = None
            
            # Use if/elif cascade for early termination
            if color == "red":
                if corners == 3:
                    shape_type = "triangle"  # Skip complex verification for performance
                elif corners == 4:
                    # Simple aspect ratio test only
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:  # Looser bounds for better detection
                        shape_type = "square"
            else:  # color == "blue"
                if corners == 4:
                    # Simple aspect ratio test only
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        shape_type = "square"
                elif 5 <= corners <= 7:
                    shape_type = "hexagon"  # Skip complex verification for performance
            
            # If shape detected, add to results
            if shape_type:
                detected_shapes.append((contour, shape_type, color))
                
        return detected_shapes

    def draw_results(self, frame, tracked_objects: List[TrackedObject]):
        """Draw tracked objects with enhanced visualization."""
        if not tracked_objects:  # Early return if no objects
            return frame
            
        # Prepare colors in advance (avoid repeated creation)
        red_color = (0, 0, 255)  # BGR format
        blue_color = (255, 0, 0)
        white_color = (255, 255, 255)
        yellow_color = (0, 255, 255)
        green_color = (0, 255, 0)
        orange_color = (0, 165, 255)
        
        for obj in tracked_objects:
            # Choose color based on object color and tracking state
            if obj.color == "red":
                base_color = red_color
            else:
                base_color = blue_color
            
            # Modify color based on tracking state
            if obj.tracking_state == "lost":
                draw_color = orange_color
                thickness = 1
            elif obj.tracking_state == "recovered":
                draw_color = green_color
                thickness = 3
            else:
                draw_color = base_color
                thickness = 2
            
            # Draw contour
            cv2.drawContours(frame, [obj.contour], -1, draw_color, thickness)
            
            # Draw center point
            cv2.circle(frame, obj.center, 5, draw_color, -1)
            
            # Draw tracking history (trajectory)
            if len(obj.track_history) > 1:
                points = list(obj.track_history)
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], yellow_color, 1)
            
            # Draw object ID and info
            label = f"ID:{obj.id} {obj.color} {obj.shape_type}"
            
            # Add tracking state info
            if obj.tracking_state == "lost":
                label += f" (LOST:{obj.lost_frames})"
            elif obj.tracking_state == "recovered":
                label += " (RECOVERED)"
            
            # Add confidence score
            label += f" C:{obj.confidence:.2f}"
            
            # Position label above the object
            label_pos = (obj.center[0] - 40, obj.center[1] - 20)
            
            # Draw label background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, 
                         (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                         (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, label, label_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, white_color, 1)
        
        return frame

    def calculate_fps(self):
        """Calculate FPS with minimal overhead."""
        current_time = time.time()
        delta_time = current_time - self.prev_frame_time
        
        # Avoid division by zero or very small numbers
        if delta_time > 0.001:
            fps = 1.0 / delta_time
            self.fps_buffer.append(fps)
            self.prev_frame_time = current_time
        
        # Fast averaging of FPS buffer using numpy mean
        if self.fps_buffer:
            return np.mean(self.fps_buffer)
        return 0

    def process_frame(self, frame):
        """Process a single frame with object tracking and fail-safe mechanisms."""
        # Measure time to dynamically adjust frame skipping
        start_time = time.time()
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Detect shapes
        detected_shapes = self.detect_shapes(frame)
        
        # Update tracker with new detections
        tracked_objects = self.tracker.update(detected_shapes)
        
        # Draw tracking results (modify frame in-place)
        self.draw_results(frame, tracked_objects)
        
        # Add FPS counter and tracking info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add tracking statistics
        active_count = sum(1 for obj in tracked_objects if obj.tracking_state == "active")
        lost_count = sum(1 for obj in tracked_objects if obj.tracking_state == "lost")
        recovered_count = sum(1 for obj in tracked_objects if obj.tracking_state == "recovered")
        
        tracking_info = f"Active:{active_count} Lost:{lost_count} Recovered:{recovered_count}"
        cv2.putText(frame, tracking_info, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dynamically adjust frame processing rate based on performance
        elapsed_time = time.time() - start_time
        target_time = 1.0 / 20.0  # Target 20 FPS for processing
        
        # If processing is too slow, increase frame skip rate
        if elapsed_time > target_time:
            self.process_every_n_frames = min(3, self.process_every_n_frames + 1)
        # If processing is fast, decrease frame skip rate
        elif elapsed_time < target_time * 0.7 and self.process_every_n_frames > 1:
            self.process_every_n_frames -= 1
            
        return frame

def main():
    """Main function optimized for Raspberry Pi performance."""
    # Set process priority (nice value) - lower means higher priority
    try:
        os.nice(-10)  # Requires appropriate permissions
        print("Process priority increased")
    except:
        print("Could not set process priority (may need sudo)")
    
    # Attempt to bind to specific CPU cores (2-3 if available)
    try:
        # On quad-core Raspberry Pi, use cores 2-3 for this process
        cores = "2-3" if os.cpu_count() >= 4 else "0-1"
        os.system(f"taskset -cp {cores} {os.getpid()} > /dev/null")
        print(f"Process bound to CPU cores {cores}")
    except:
        print("Could not set CPU affinity")
    
    # Initialize camera with async capture
    capture = AsyncVideoCapture(0)
    
    # Allow camera to warm up
    time.sleep(1.0)
    
    # Initialize shape detector
    detector = ShapeDetector()
    
    print("Optimized Shape Detection System Running on Raspberry Pi...")
    print("Press 'q' to quit")
    
    try:
        # Main processing loop
        while True:
            # Read frame
            ret, frame = capture.read()
            if not ret:
                print("Error: Failed to capture frame.")
                time.sleep(0.1)  # Wait before trying again
                continue
            
            # Process the frame
            result_frame = detector.process_frame(frame)
            
            # Display the result
            cv2.imshow('Optimized Shape Detection', result_frame)
            
            # Exit on 'q' key press (with minimal wait time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    finally:
        # Release resources
        capture.release()
        cv2.destroyAllWindows()
        print("Shape Detection System Stopped.")


if __name__ == "__main__":
    # Check for Raspberry Pi-specific configurations
    # Try to disable HDMI to free up memory and processing power
    try:
        tv_service = os.system("tvservice -s 2> /dev/null")
        if tv_service == 0:  # Command exists (we're on Raspberry Pi)
            # Only disable if not using the display
            if not os.environ.get('DISPLAY'):
                os.system("tvservice -o")
                print("HDMI output disabled for performance")
    except:
        pass  # Not on Raspberry Pi or tvservice not available
        
    main()