import cv2
import numpy as np
import time
from collections import deque
import threading
import queue
import os
import atexit

position_history = deque(maxlen=30)
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

class KalmanTracker:
    def __init__(self):
        # Kalman Filter oluştur
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state, 2 measurement
        
        # State: [x, y, vx, vy] - pozisyon ve hız
        self.kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Transition matrix (A) - bir sonraki state nasıl hesaplanır
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x(t+1) = x(t) + vx(t)
            [0, 1, 0, 1],  # y(t+1) = y(t) + vy(t)
            [0, 0, 1, 0],  # vx(t+1) = vx(t)
            [0, 0, 0, 1]   # vy(t+1) = vy(t)
        ], dtype=np.float32)
        
        # Measurement matrix (H) - state'ten measurement'a dönüşüm
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # x measurement
            [0, 1, 0, 0]   # y measurement
        ], dtype=np.float32)
        
        # Process noise (obje ne kadar düzensiz hareket eder)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise (detection ne kadar güvenilir)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10

import numpy as np
import cv2

class KalmanTracker:
    def __init__(self):
        # Kalman Filter oluştur
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state, 2 measurement
        
        # State: [x, y, vx, vy] - pozisyon ve hız
        self.kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Transition matrix (A) - bir sonraki state nasıl hesaplanır
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x(t+1) = x(t) + vx(t)
            [0, 1, 0, 1],  # y(t+1) = y(t) + vy(t)
            [0, 0, 1, 0],  # vx(t+1) = vx(t)
            [0, 0, 0, 1]   # vy(t+1) = vy(t)
        ], dtype=np.float32)
        
        # Measurement matrix (H) - state'ten measurement'a dönüşüm
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # x measurement
            [0, 1, 0, 0]   # y measurement
        ], dtype=np.float32)
        
        # Process noise (obje ne kadar düzensiz hareket eder)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise (detection ne kadar güvenilir)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        
import numpy as np
import cv2

class KalmanTracker:
    def __init__(self):
        # Kalman Filter oluştur
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state, 2 measurement
        
        # State: [x, y, vx, vy] - pozisyon ve hız
        self.kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Transition matrix (A) - bir sonraki state nasıl hesaplanır
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x(t+1) = x(t) + vx(t)
            [0, 1, 0, 1],  # y(t+1) = y(t) + vy(t)
            [0, 0, 1, 0],  # vx(t+1) = vx(t)
            [0, 0, 0, 1]   # vy(t+1) = vy(t)
        ], dtype=np.float32)
        
        # Measurement matrix (H) - state'ten measurement'a dönüşüm
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # x measurement
            [0, 1, 0, 0]   # y measurement
        ], dtype=np.float32)
        
        # Process noise (obje ne kadar düzensiz hareket eder)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise (detection ne kadar güvenilir)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        
    def update(self, detection=None):
        """
        Kalman Filter güncelleme
        detection: contour approx, (x, y) tuple veya None
        """
        if detection is not None:
            # Eğer detection bir tuple ise (contour, shape_type, color)
            if isinstance(detection, tuple) and len(detection) >= 3:
                contour, shape_type, color = detection
                detection = contour  # Sadece contour'u al
            
            # Contour approx formatını kontrol et ve merkez hesapla
            if isinstance(detection, np.ndarray):
                if len(detection.shape) == 3:
                    # OpenCV contour format: [[[x,y]], [[x,y]], ...]
                    try:
                        moments = cv2.moments(detection)
                        if moments['m00'] != 0:
                            center_x = moments['m10'] / moments['m00']
                            center_y = moments['m01'] / moments['m00']
                        else:
                            # Moment hesaplanamadıysa ortalama al
                            points = detection.reshape(-1, 2)
                            center_x = np.mean(points[:, 0])
                            center_y = np.mean(points[:, 1])
                    except:
                        # Hata durumunda basit ortalama
                        points = detection.reshape(-1, 2)
                        center_x = np.mean(points[:, 0])
                        center_y = np.mean(points[:, 1])
                    
                    # Scalar değerlere çevir
                    x = float(center_x)
                    y = float(center_y)
                    
                elif len(detection.shape) == 2:
                    # 2D array format
                    center_x = np.mean(detection[:, 0])
                    center_y = np.mean(detection[:, 1])
                    x = float(center_x)
                    y = float(center_y)
                    
                elif len(detection.shape) == 1 and len(detection) >= 2:
                    # 1D array format
                    x = float(detection[0])
                    y = float(detection[1])
                else:
                    print(f"Unexpected array shape: {detection.shape}")
                    return None
                    
            elif isinstance(detection, (tuple, list)):
                # Tuple/list içindeki değerleri kontrol et
                if len(detection) >= 2:
                    try:
                        x = float(detection[0])
                        y = float(detection[1])
                    except (ValueError, TypeError):
                        # İç içe yapı varsa
                        if hasattr(detection[0], '__len__'):
                            x = float(detection[0][0])
                            y = float(detection[1][0])
                        else:
                            print(f"Cannot convert to float: {detection[0]}, {detection[1]}")
                            return None
                else:
                    print(f"Insufficient elements: {len(detection)}")
                    return None
            else:
                print(f"Unexpected detection format: {type(detection)}")
                return None
            
            # Detection var - correct ve predict
            measurement = np.array([x, y], dtype=np.float32)
            
            if not self.initialized:
                # İlk detection - state'i başlat
                self.kalman.statePre = np.array([x, y, 0.0, 0.0], dtype=np.float32)
                self.initialized = True
                self.lost_frames = 0
                
            # Measurement ile düzelt
            self.kalman.correct(measurement)
            self.lost_frames = 0
            
        else:
            # Detection yok - sadece predict
            self.lost_frames += 1
            
        if self.initialized:
            # Sonraki frame için tahmin
            prediction = self.kalman.predict()
            return prediction
        else:
            return None
        
    def draw_simple_trail(self,frame, position_history, max_length=30, color=(0, 255, 255), thickness=2):
        if len(position_history) < 2:
            return
        
        # Son max_length kadar pozisyonu al
        recent_positions = list(position_history)[-max_length:]
        
        # Çizgileri çiz
        for i in range(1, len(recent_positions)):
            pt1 = (int(recent_positions[i-1][0]), int(recent_positions[i-1][1]))
            pt2 = (int(recent_positions[i][0]), int(recent_positions[i][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
        
    def draw_center_circle(self,frame,data):
        cv2.circle(frame, (data['position'][0], data['position'][1]), 5, (0, 255, 0), 4)
    
    def get_tracking_data(self):
        """Tracking verilerini döndür"""
        if not self.initialized:
            return None
            
        state = self.kalman.statePost
        
        tracking_data = {
            'position': (int(state[0]), int(state[1])),
            'velocity': (state[2], state[3]),
            'predicted_position': (int(state[0] + state[2]), int(state[1] + state[3])),
            'confidence': max(0, 1 - (self.lost_frames / self.max_lost_frames)),
            'tracking_status': 'ACTIVE' if self.lost_frames < 5 else 'PREDICTING' if self.lost_frames < self.max_lost_frames else 'LOST',
            'lost_frames': self.lost_frames,
            'speed': np.sqrt(state[2]**2 + state[3]**2)
        }
        
        return tracking_data
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
    

    def isRegularHexagon(self,approx,x,y,w,h):
        if(len(approx) != 6):
            return False
        #Merkex x ve y noktaları
        center_x = x + w / 2
        center_y = y + h / 2
    
        distances = []

        for point in approx:
            px,py = point[0]
            distance = np.sqrt((px-center_x)**2 + (py-center_y)**2)
            distances.append(distance)
            #noktaların merkeze olan uzaklıklarını al 

        #ortalama merkez uzaklığı
        mean_distance = np.mean(distances)
        angles = []

        for i in range(6):
            #birinci nokta
            p1 = approx[i][0]
            #sonraki nokta
            p2 = approx[(i+1)%6][0]
            #iki sonraki nokta
            p3 = approx[(i+2)%6][0]
            #p2 den p1 e giden vektör
            v1 = np.array([p1[0]-p2[0],p1[1]-p2[1]])
            #p2 den p3 ye giden vektör
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            #np.dot iki vektörün nokta çarpımını alır
            #np.linalg.norm vektörün uzunluğunu alır
            #vektörlerin noktasal çarpımı / iki vektörün uzunluklarının çarpımı
            #cosinus değeri
            #-------- Meraklısına -------
            #v1 = [3, 4]  # Uzunluk = √(9+16) = 5
            # v2 = [1, 0]  # Uzunluk = √(1+0) = 1
            # nokta_carpim = 3*1 + 4*0 = 3
            # cos_angle = 3 / (5 * 1) = 0.6
            # angle = arccos(0.6) = 53.13°
            #-------- Meraklısına -------
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            #arccos ile açı bulunur
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            angles.append(angle)

        #noktaların merkeze olan uzaklığı %10 sapabilir
        distance_tolerance = mean_distance * 0.1  # %10 tolerans
        #açılar ise 10 derece sapabilir.
        angle_tolerance = 10 
        
        #eşit mesafeler = 
        #noktaların uzaklıklarını gez. Eğer ortalama mesafeye tölerans farkı ile geçmiyorsa kabul et.
        uniform_distances = all(abs(d - mean_distance) < distance_tolerance for d in distances)
        #aynı şekilde yukarıdaki işlemi ortalama açılar için yap
        uniform_angles = all(abs(angle - 120) < angle_tolerance for angle in angles)
        # En-boy oranı yaklaşık 1 olmalı (kare benzeri)
        #en boy oranı al büyük olan payda olması için max min kullanıldı.
        aspect_ratio = max(w, h) / min(w, h)
        #eğer en boy oranı 1.3'den küçükse kabul et.
        proper_aspect = aspect_ratio < 1.3

        #koşullar sağlandı ise True döndür.
        return uniform_distances and uniform_angles and proper_aspect


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
                x,y,w,h = cv2.boundingRect(approx)
                if corners == 4:
                    # Simple aspect ratio test only
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        shape_type = "square"
                elif corners == 6:
                    if self.isRegularHexagon(approx,x,y,w,h):
                            shape_type = "hexagon"  # Skip complex verification for performance
            
            # If shape detected, add to results
            if shape_type:
                detected_shapes.append((contour, shape_type, color))
                
        return detected_shapes

    def draw_results(self, frame, shapes):
        """Draw detected shapes efficiently."""
        if not shapes:  # Early return if no shapes
            return frame
            
        # Prepare colors in advance (avoid repeated creation)
        red_color = (0, 0, 255)  # BGR format
        blue_color = (255, 0, 0)
        white_color = (255, 255, 255)
        
        for contour, shape_type, color in shapes:
            # Draw contour (inplace operation)
            color_bgr = red_color if color == "red" else blue_color
            cv2.drawContours(frame, [contour], -1, color_bgr, 2)
            
            # Only calculate moments if needed for label positioning
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw label (smaller font size for performance)
                label = f"{color} {shape_type}"
                cv2.putText(frame, label, (cx - 20, cy), 
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

    def process_frame(self, frame,kalman_tracker:KalmanTracker):
        """Process a single frame with adaptive frame skipping for consistent FPS."""
        # Measure time to dynamically adjust frame skipping
        start_time = time.time()
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Detect shapes
        shapes = self.detect_shapes(frame)
        
        for index,shape in enumerate(shapes):
            print(shapes)
            prediction = kalman_tracker.update(shapes[0])
            data = kalman_tracker.get_tracking_data()
            kalman_tracker.draw_simple_trail(frame, position_history, max_length=30, color=(0, 255, 255), thickness=2)

            if data:
                position_history.append(data['position'])
                kalman_tracker.draw_center_circle(frame, data)
                print(f"Position: {data['position']}")
                velocity_str = list(map('{:.2f}%'.format,data['velocity'][0]))
                print(f"Velocity: ({velocity_str})")
                print(f"Predicted: {data['predicted_position']}")
                print(f"Status: {data['tracking_status']}")
                print(f"Confidence: ({float(data['confidence'])})")
                print(f"Speed: {float(data['confidence'])}")
            else:
                print("Tracker not initialized")
        # Draw results (modify frame in-place)
        self.draw_results(frame, shapes)
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
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
            kalman_tracker = KalmanTracker()
            # Process the frame
            result_frame = detector.process_frame(frame,kalman_tracker)
            
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