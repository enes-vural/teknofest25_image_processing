import socket
import cv2
import numpy as np
import time
from collections import deque
import threading
import queue
import os
import atexit
import subprocess
import signal
from collections import defaultdict
import logging
from datetime import datetime
import RPi.GPIO as GPIO
import time

# GPIO pinini ayarla (servo motorun sinyal kablosu bu pine bağlanacak)
servo_pin = 17


def set_angle(angle):
    """
    Servo motoru belirtilen açıya döndür
    angle: 0-180 derece arası değer
    """
    # Açıyı PWM duty cycle'a çevir
    # 0° = 2.5% duty cycle, 180° = 12.5% duty cycle
    duty_cycle = 2 + (angle / 18)
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Servo motorun hareket etmesi için bekle
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

# GPIO ayarları
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM ayarları (50Hz frekans - servo motorlar için standart)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)  # %0 duty cycle ile başla

# Camera Configuration
WIDTH, HEIGHT = 640, 480
FPS = 30

# Enable OpenCV optimizations (uses NEON SIMD instructions on ARM if available)
cv2.setUseOptimized(True)

# Logging Configuration
def setup_logging():
    """Set up logging configuration for shape detection system."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create log filename with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"logs/shape_detection_{current_date}.txt"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("========== Shape Detection System Started ==========")
    logger.info(f"Log file: {log_filename}")
    
    return logger

# Initialize logger
logger = setup_logging()

# Ensure OpenCV is using optimized code paths
if cv2.useOptimized():
    logger.info("OpenCV optimizations enabled (using hardware acceleration)")
else:
    logger.warning("OpenCV optimizations not available")

# Set Raspberry Pi to performance mode (requires appropriate permissions)
try:
    os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")
    logger.info("CPU set to performance mode")
    
    # Reset to ondemand governor on exit
    def reset_governor():
        os.system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")
        logger.info("CPU governor reset to ondemand")
    
    atexit.register(reset_governor)
except:
    logger.warning("Could not set CPU governor (may need sudo)")


class GstreamerCamera:
    """Camera class using libcamera and GStreamer for optimal Raspberry Pi performance."""
    
    def __init__(self, queue_size=2):
        logger.info("Initializing GStreamer camera...")
        
        # Kill any existing libcamera-vid or gst-launch processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        time.sleep(1)
        logger.info("Existing camera processes terminated")
        
        # Start streaming process
        self.stream_process = self.start_stream()
        logger.info("Streaming process started")
        
        # Create frame queue with minimal size to avoid memory buildup
        self.queue = queue.Queue(maxsize=queue_size)
        
        # Flag to control the thread
        self.running = True
        
        # Open GStreamer pipeline for receiving the stream
        self.cap = self.receive_stream()
        
        # Check if pipeline opened successfully
        if not self.cap.isOpened():
            logger.error("Could not open GStreamer pipeline!")
            raise RuntimeError("Failed to open GStreamer pipeline")
        else:
            logger.info("GStreamer pipeline successfully opened")
            
        # Warm up the camera
        ret, _ = self.cap.read()
        if not ret:
            logger.warning("Could not read initial frame from camera")
        else:
            logger.info("Camera warmed up successfully")
        
        # Start the thread for frame capture
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("Frame capture thread started")
    
    def start_stream(self):
        """Start libcamera-vid streaming process with optimized parameters."""
        cmd = """
        libcamera-vid -t 0 \
        --nopreview \
        --width 640 --height 480 \
        --mode 1280:720 \
        --framerate 30 \
        --shutter 20000 \
        --gain 3 \
        --denoise cdn_off \
        --brightness 0.1 \
        --contrast 1.2 \
        --sharpness 1.5 \
        --ev 0.1 \
        --awb auto \
        --autofocus-mode continuous \
        --autofocus-speed fast \
        --autofocus-range normal \
        --autofocus-window 0.25,0.25,0.5,0.5 \
        --lens-position 1.0 \
        --codec h264 \
        --inline \
        -o - | \
        gst-launch-1.0 fdsrc ! queue max-size-buffers=0 ! h264parse ! rtph264pay config-interval=1 ! udpsink host=0.0.0.0 port=5000 sync=false
        """
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def receive_stream(self):
        """Create GStreamer pipeline for receiving the video stream."""
        pipeline = (
            "udpsrc port=5000 buffer-size=65536 timeout=0 ! "
            "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=100 drop-on-latency=false ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false emit-signals=true drop=true"
        )
        logger.debug(f"GStreamer pipeline: {pipeline}")
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    def _update(self):
        """Background thread function to continuously grab frames."""
        frame_count = 0
        while self.running:
            # Read frame from GStreamer pipeline
            ret, frame = self.cap.read()
            
            if ret:
                frame_count += 1
                # If queue is full, remove oldest frame
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                # Put the new frame in the queue
                self.queue.put(frame)
                
                # Log frame capture every 100 frames to avoid spam
                if frame_count % 100 == 0:
                    logger.debug(f"Captured {frame_count} frames")
            else:
                logger.warning("Failed to read frame from GStreamer pipeline")
                time.sleep(0.1)  # Wait before trying again
            
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
        logger.info("Releasing camera resources...")
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # Release OpenCV resources
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        # Terminate the streaming process
        if hasattr(self, 'stream_process') and self.stream_process is not None:
            try:
                self.stream_process.terminate()
                self.stream_process.wait(timeout=2)
            except:
                # Force kill if termination fails
                try:
                    self.stream_process.kill()
                except:
                    pass
        
        # Make sure to kill any remaining processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        logger.info("Camera resources released")


class ShapeDetector:
    """Class for detecting specific colored geometric shapes optimized for Raspberry Pi."""
    
    def __init__(self):
        logger.info("Initializing Shape Detector...")
        
        # Pre-compute constants and allocate buffers to avoid repeated memory allocations
        
        # Color boundaries in HSV space (lower, upper) - optimized for Raspberry Pi camera 
        self.red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
        self.red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([160, 100, 100], dtype=np.uint8)  # Red wraps around HSV space
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self.blue_lower = np.array([100, 100, 100], dtype=np.uint8)
        self.blue_upper = np.array([140, 255, 255], dtype=np.uint8)
        
        # Detection thresholds
        self.min_contour_area = 500  # Adjusted for 640x480 resolution
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
        
        # Statistics for logging
        self.total_frames_processed = 0
        self.shapes_detected_count = {'red_triangle': 0, 'red_square': 0, 'blue_square': 0, 'blue_hexagon': 0}
        self.last_log_time = time.time()
        
        logger.info("Shape Detector initialized successfully")
        logger.info(f"Color detection thresholds - Red HSV: {self.red_lower1}-{self.red_upper1} & {self.red_lower2}-{self.red_upper2}")
        logger.info(f"Color detection thresholds - Blue HSV: {self.blue_lower}-{self.blue_upper}")

    def preprocess_frame(self, frame):
        """Apply optimized preprocessing for Raspberry Pi."""
        # Allocate memory buffers if not already done
        if self.hsv_buffer is None or self.hsv_buffer.shape[:2] != frame.shape[:2]:
            h, w = frame.shape[:2]
            self.hsv_buffer = np.empty((h, w, 3), dtype=np.uint8)
            self.red_mask = np.empty((h, w), dtype=np.uint8)
            self.blue_mask = np.empty((h, w), dtype=np.uint8)
            logger.debug(f"Allocated buffers for frame size: {w}x{h}")
        
        # Use medianBlur instead of GaussianBlur (faster on Raspberry Pi with NEON)
        blurred = cv2.medianBlur(frame, 3)  # 3x3 median is faster than Gaussian
        
        # Convert to HSV - efficient in-place conversion
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV, dst=self.hsv_buffer)
        
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
    
    def isRegularHexagon(self, approx, x, y, w, h):
        if(len(approx) != 6):
            return False
        #Merkez x ve y noktaları
        center_x = x + w / 2
        center_y = y + h / 2
    
        distances = []

        for point in approx:
            px, py = point[0]
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
            v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
            #p2 den p3 ye giden vektör
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            #arccos ile açı bulunur
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            angles.append(angle)

        #noktaların merkeze olan uzaklığı %10 sapabilir
        distance_tolerance = mean_distance * 0.1  # %10 tolerans
        #açılar ise 10 derece sapabilir.
        angle_tolerance = 10 
        
        uniform_distances = all(abs(d - mean_distance) < distance_tolerance for d in distances)
        uniform_angles = all(abs(angle - 120) < angle_tolerance for angle in angles)
        aspect_ratio = max(w, h) / min(w, h)
        proper_aspect = aspect_ratio < 1.3

        return uniform_distances and uniform_angles and proper_aspect

    def detect_shapes(self, frame):
        """Detect shapes with frame skipping for better performance."""
        # Increment frame counter
        self.frame_count += 1
        self.total_frames_processed += 1
        
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
            logger.info("PROXIMITY ALERT: Red shape detected very close (inside detection zone)")
        if blue_close:
            logger.info("PROXIMITY ALERT: Blue shape detected very close (inside detection zone)")
        
        # Fast contour finding with simple approximation
        red_shapes = self.process_contours(red_mask, frame, "red")
        blue_shapes = self.process_contours(blue_mask, frame, "blue")
        
        all_shapes = red_shapes + blue_shapes
        
        # Log detected shapes
        if all_shapes:
            shape_info = []
            for contour, shape_type, color in all_shapes:
                shape_key = f"{color}_{shape_type}"
                self.shapes_detected_count[shape_key] += 1
                
                # Get shape position and size
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                shape_info.append(f"{color} {shape_type} (pos:{x},{y} size:{w}x{h} area:{area:.0f})")
            
            logger.info(f"SHAPES DETECTED: {'; '.join(shape_info)}")
        
        # Log statistics every 30 seconds
        current_time = time.time()
        if current_time - self.last_log_time > 30:
            self.log_statistics()
            self.last_log_time = current_time
        
        return all_shapes

    def process_contours(self, mask, frame, color):
        """Process contours with optimized algorithm for Raspberry Pi."""
        detected_shapes = []
        
        # Use RETR_EXTERNAL and CHAIN_APPROX_SIMPLE for better performance
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Early return if no contours (avoid unnecessary processing)
        if not contours:
            return []
        
        logger.debug(f"Found {len(contours)} {color} contours to analyze")
        
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
            x, y, w, h = cv2.boundingRect(approx)
            
            # Use if/elif cascade for early termination
            if color == "red":
                if corners == 3:
                    shape_type = "triangle"
                    logger.debug(f"Red triangle detected: corners={corners}, area={area:.0f}")
                    set_angle(75)
                    time.sleep(3)
                    set_angle(35)
                    #TODO triangle algortiması yazılacak
                # elif corners == 4:
                #     # Simple aspect ratio test only
                #     aspect_ratio = float(w) / h
                #     if 0.7 <= aspect_ratio <= 1.3:  # Looser bounds for better detection
                #         shape_type = "square"
                #         logger.debug(f"Red square detected: corners={corners}, aspect_ratio={aspect_ratio:.2f}, area={area:.0f}")
            else:  # color == "blue"
                # if corners == 4:
                #     # Simple aspect ratio test only
                #     aspect_ratio = float(w) / h
                #     if 0.7 <= aspect_ratio <= 1.3:
                #         shape_type = "square"
                #         logger.debug(f"Blue square detected: corners={corners}, aspect_ratio={aspect_ratio:.2f}, area={area:.0f}")
                if corners == 6:
                    if self.isRegularHexagon(approx, x, y, w, h):
                        shape_type = "hexagon"
                        logger.debug(f"Blue hexagon detected: corners={corners}, area={area:.0f}")
                        set_angle(0)
                        time.sleep(3)
                        set_angle(35)
                        #TODO hexagon algortiması yazılacak

            
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

    def log_statistics(self):
        """Log detection statistics."""
        total_shapes = sum(self.shapes_detected_count.values())
        logger.info("=== DETECTION STATISTICS (Last 30 seconds) ===")
        logger.info(f"Total frames processed: {self.total_frames_processed}")
        logger.info(f"Total shapes detected: {total_shapes}")
        
        for shape_type, count in self.shapes_detected_count.items():
            if count > 0:
                logger.info(f"  {shape_type.replace('_', ' ').title()}: {count} detections")
        
        if self.fps_buffer:
            avg_fps = np.mean(self.fps_buffer)
            logger.info(f"Average FPS: {avg_fps:.2f}")
        
        logger.info("=" * 45)
        
        # Reset counters
        self.shapes_detected_count = {'red_triangle': 0, 'red_square': 0, 'blue_square': 0, 'blue_hexagon': 0}

    def process_frame(self, frame):
        """Process a single frame with adaptive frame skipping for consistent FPS."""
        # Measure time to dynamically adjust frame skipping
        start_time = time.time()
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Detect shapes
        shapes = self.detect_shapes(frame)
        
        # Draw detected shapes
        self.draw_results(frame, shapes)
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add frame counter
        cv2.putText(frame, f"Frames: {self.total_frames_processed}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dynamically adjust frame processing rate based on performance
        elapsed_time = time.time() - start_time
        target_time = 1.0 / 20.0  # Target 20 FPS for processing
        
        # If processing is too slow, increase frame skip rate
        if elapsed_time > target_time:
            if self.process_every_n_frames < 3:
                self.process_every_n_frames += 1
                logger.debug(f"Increased frame skip to every {self.process_every_n_frames} frames (processing too slow)")
        # If processing is fast, decrease frame skip rate
        elif elapsed_time < target_time * 0.7 and self.process_every_n_frames > 1:
            self.process_every_n_frames -= 1
            logger.debug(f"Decreased frame skip to every {self.process_every_n_frames} frames (processing fast enough)")
            
        return frame


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig} to terminate. Cleaning up...")
        # Kill libcamera and GStreamer processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        logger.info("Shape Detection System terminated gracefully")
        # Exit program
        import sys
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function optimized for Raspberry Pi performance with logging."""
    # Set up signal handlers for graceful termination
    setup_signal_handlers()
    
    # Set process priority (nice value) - lower means higher priority
    try:
        os.nice(-10)  # Requires appropriate permissions
        logger.info("Process priority increased (nice value: -10)")
    except:
        logger.warning("Could not set process priority (may need sudo)")
    
    # Attempt to bind to specific CPU cores (2-3 if available)
    try:
        # On quad-core Raspberry Pi, use cores 2-3 for this process
        cores = "2-3" if os.cpu_count() >= 4 else "0-1"
        os.system(f"taskset -cp {cores} {os.getpid()} > /dev/null")
        logger.info(f"Process bound to CPU cores {cores}")
    except:
        logger.warning("Could not set CPU affinity")
    
    # Kill any existing camera processes before starting
    os.system("sudo pkill -9 libcamera-vid")
    os.system("sudo pkill -9 gst-launch-1.0")
    time.sleep(1)
    logger.info("Existing camera processes terminated before startup")
    
    # Initialize camera with GStreamer pipeline
    try:
        capture = GstreamerCamera()
        #TODO DEFAULT SERVO DEGREE
        set_angle(35)
        
        # Allow camera to warm up
        time.sleep(2.0)
        logger.info("Camera warm-up completed")
        
        # Test if camera is working
        ret, test_frame = capture.read()
        if not ret or test_frame is None:
            logger.error("Camera not working properly. Exiting.")
            return
        else:
            logger.info(f"Camera test successful - frame size: {test_frame.shape}")
            
        # Initialize shape detector
        detector = ShapeDetector()
        
        logger.info("========== Shape Detection System Active ==========")
        logger.info("System running - Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        # Main processing loop
        while True:
            # Read frame
            ret, frame = capture.read()
            if not ret:
                logger.warning("Failed to capture frame - retrying...")
                time.sleep(0.1)  # Wait before trying again
                continue

            # Process the frame (shape detection)
            result_frame = detector.process_frame(frame)
            
            frame_count += 1
            
            # Log system status every 500 frames
            if frame_count % 500 == 0:
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time
                logger.info(f"SYSTEM STATUS: Processed {frame_count} frames in {elapsed_time:.1f}s (avg: {avg_fps:.2f} FPS)")
            
            # Display the result locally (optional - can be commented out for headless operation)
            cv2.imshow('Shape Detection with Logging', result_frame)
            
            # Exit on 'q' key press (with minimal wait time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested shutdown (pressed 'q')")
                break
                
    except KeyboardInterrupt:
        logger.info("Program interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Release resources
        if 'capture' in locals():
            capture.release()
        cv2.destroyAllWindows()
        
        # Kill any remaining camera processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        
        # Log final statistics
        if 'detector' in locals():
            detector.log_statistics()
        
        total_runtime = time.time() - start_time if 'start_time' in locals() else 0
        logger.info(f"Total runtime: {total_runtime:.2f} seconds")
        logger.info("========== Shape Detection System Stopped ==========")


if __name__ == "__main__":
    # Check for required modules
    import sys
    required_modules = ['cv2', 'numpy', 'subprocess']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"ERROR: Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install " + ' '.join(missing_modules))
        sys.exit(1)
    
    # Check for GStreamer support in OpenCV
    if not cv2.getBuildInformation().find('GStreamer') != -1:
        print("WARNING: OpenCV was not built with GStreamer support.")
        print("This program requires OpenCV with GStreamer support to function correctly.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    # Start the main program
    main()
