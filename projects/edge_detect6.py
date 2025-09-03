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

# UDP Configuration
DEST_IP = '127.0.0.1'
PRO_DEST_PORT = 5051
RAW_DEST_PORT = 5052
WIDTH, HEIGHT = 640, 480
FPS = 30
MAX_PACKET_SIZE = 1400  # UDP için güvenli paket boyutu

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


class GstreamerCamera:
    """Camera class using libcamera and GStreamer for optimal Raspberry Pi performance."""
    
    def __init__(self, queue_size=2):
        # Kill any existing libcamera-vid or gst-launch processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        time.sleep(1)
        
        # Start streaming process
        self.stream_process = self.start_stream()
        
        # Create frame queue with minimal size to avoid memory buildup
        self.queue = queue.Queue(maxsize=queue_size)
        
        # Flag to control the thread
        self.running = True
        
        # Open GStreamer pipeline for receiving the stream
        self.cap = self.receive_stream()
        
        # Check if pipeline opened successfully
        if not self.cap.isOpened():
            print("ERROR: Could not open GStreamer pipeline!")
            raise RuntimeError("Failed to open GStreamer pipeline")
        else:
            print("GStreamer pipeline successfully opened")
            
        # Warm up the camera
        ret, _ = self.cap.read()
        if not ret:
            print("WARNING: Could not read initial frame from camera")
        
        # Start the thread for frame capture
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
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
        print("GStreamer pipeline:", pipeline)
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    def _update(self):
        """Background thread function to continuously grab frames."""
        while self.running:
            # Read frame from GStreamer pipeline
            ret, frame = self.cap.read()
            
            if ret:
                # If queue is full, remove oldest frame
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                # Put the new frame in the queue
                self.queue.put(frame)
            else:
                print("WARNING: Failed to read frame from GStreamer pipeline")
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

    def preprocess_frame(self, frame):
        """Apply optimized preprocessing for Raspberry Pi."""
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
            x, y, w, h = cv2.boundingRect(approx)
            
            # Use if/elif cascade for early termination
            if color == "red":
                if corners == 3:
                    shape_type = "triangle"
                elif corners == 4:
                    # Simple aspect ratio test only
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:  # Looser bounds for better detection
                        shape_type = "square"
            else:  # color == "blue"
                if corners == 4:
                    # Simple aspect ratio test only
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        shape_type = "square"
                elif corners == 6:
                    if self.isRegularHexagon(approx, x, y, w, h):
                        shape_type = "hexagon"
            
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


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        # Kill libcamera and GStreamer processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        # Exit program
        import sys
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function optimized for Raspberry Pi performance with UDP streaming."""
    # Set up signal handlers for graceful termination
    setup_signal_handlers()
    
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
    
    # Kill any existing camera processes before starting
    os.system("sudo pkill -9 libcamera-vid")
    os.system("sudo pkill -9 gst-launch-1.0")
    time.sleep(1)
    
    # Initialize UDP sockets for streaming
    sock_pro = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Processed video stream: {DEST_IP}:{PRO_DEST_PORT}")
    print(f"Raw video stream: {DEST_IP}:{RAW_DEST_PORT}")
    
    # Initialize camera with GStreamer pipeline
    try:
        capture = GstreamerCamera()
        
        # Allow camera to warm up
        time.sleep(2.0)
        
        # Test if camera is working
        ret, test_frame = capture.read()
        if not ret or test_frame is None:
            print("ERROR: Camera not working properly. Exiting.")
            return
            
        # Initialize shape detector
        detector = ShapeDetector()
        
        print("Optimized Shape Detection System Running on Raspberry Pi...")
        print("UDP Streaming Active")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        # Main processing loop
        while True:
            # Read frame
            ret, frame = capture.read()
            if not ret:
                print("Error: Failed to capture frame.")
                time.sleep(0.1)  # Wait before trying again
                continue
            
            # First, encode and send raw frame via UDP
            raw_success, raw_encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            if raw_success:
                # Convert encoded frame to bytes
                raw_frame_bytes = raw_encoded_frame.tobytes()
                
                # Split frame into packets and send
                total_size = len(raw_frame_bytes)
                num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                
                for i in range(num_packets):
                    start_idx = i * MAX_PACKET_SIZE
                    end_idx = min((i + 1) * MAX_PACKET_SIZE, total_size)
                    
                    # Packet header: frame_id, packet_id, total_packets, data_size
                    header = f"{frame_count:06d},{i:03d},{num_packets:03d},{end_idx-start_idx:04d},".encode()
                    packet_data = raw_frame_bytes[start_idx:end_idx]
                    
                    packet = header + packet_data
                    sock_raw.sendto(packet, (DEST_IP, RAW_DEST_PORT))
            
            # Process the frame (shape detection only - no tracking)
            result_frame = detector.process_frame(frame)
            
            # Encode and send processed frame via UDP
            pro_success, pro_encoded_frame = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            if pro_success:
                # Convert encoded frame to bytes
                pro_frame_bytes = pro_encoded_frame.tobytes()
                
                # Split frame into packets and send
                total_size = len(pro_frame_bytes)
                num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                
                for i in range(num_packets):
                    start_idx = i * MAX_PACKET_SIZE
                    end_idx = min((i + 1) * MAX_PACKET_SIZE, total_size)
                    
                    # Packet header: frame_id, packet_id, total_packets, data_size
                    header = f"{frame_count:06d},{i:03d},{num_packets:03d},{end_idx-start_idx:04d},".encode()
                    packet_data = pro_frame_bytes[start_idx:end_idx]
                    
                    packet = header + packet_data
                    sock_pro.sendto(packet, (DEST_IP, PRO_DEST_PORT))
            
            frame_count += 1
            
            # Display the result locally (optional - can be commented out for headless operation)
            cv2.imshow('Optimized Shape Detection', result_frame)
            
            # Exit on 'q' key press (with minimal wait time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Release resources
        if 'capture' in locals():
            capture.release()
        if 'sock_pro' in locals():
            sock_pro.close()
        if 'sock_raw' in locals():
            sock_raw.close()
        cv2.destroyAllWindows()
        
        # Kill any remaining camera processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        
        print("Shape Detection System Stopped.")


if __name__ == "__main__":
    # Check for required modules
    import sys
    required_modules = ['cv2', 'numpy', 'subprocess', 'socket']
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