import cv2
import numpy as np
import time
import json
import threading
import queue
import os
import sys
import socket
import struct
import atexit
import subprocess
import signal
import logging
from collections import deque
from datetime import datetime
import asyncio
import websockets
import base64
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/vision_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Optimize for Raspberry Pi 4 cores

class SystemOptimizer:
    """System optimization utilities for Raspberry Pi 4"""
    
    @staticmethod
    def optimize_system():
        """Apply system-level optimizations"""
        try:
            # Set CPU governor to performance
            os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")
            
            # Increase GPU memory split
            os.system("sudo raspi-config nonint do_memory_split 128")
            
            # Set process priority
            os.nice(-10)
            
            # Bind to specific CPU cores
            cores = "2-3" if os.cpu_count() >= 4 else "0-1"
            os.system(f"taskset -cp {cores} {os.getpid()} > /dev/null 2>&1")
            
            logger.info("System optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
    
    @staticmethod
    def cleanup():
        """Reset system settings on exit"""
        try:
            os.system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")
            os.system("sudo pkill -f libcamera")
            os.system("sudo pkill -f gst-launch")
            os.system("sudo pkill -f ffmpeg")
        except:
            pass

class H265RTCStreamer:
    """High-performance H.265 RTC streaming for ground station communication"""
    
    def __init__(self, ground_station_ip="192.168.1.100", rtc_port=8000, websocket_port=8001):
        self.ground_station_ip = ground_station_ip
        self.rtc_port = rtc_port
        self.websocket_port = websocket_port
        self.streaming = False
        self.encoder_process = None
        self.websocket_server = None
        self.connected_clients = set()
        
        # H.265 encoding parameters optimized for Raspberry Pi 4
        self.encoding_params = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'crf': '23',
            'maxrate': '2000k',
            'bufsize': '4000k',
            'keyint': '30',
            'fps': '25'
        }
        
        # Setup directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        Path("/tmp/vision_streams").mkdir(exist_ok=True)
        Path("/tmp/vision_logs").mkdir(exist_ok=True)
        
    def start_h265_encoder(self, input_source="udp://127.0.0.1:5000"):
        """Start H.265 encoder with hardware acceleration if available"""
        
        # Check for hardware encoder support
        hw_encoder = "h264_v4l2m2m"  # Raspberry Pi hardware encoder
        
        # FFmpeg command with optimized H.265 encoding
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
            '-f', 'h264',
            '-i', input_source,
            '-c:v', 'libx265',
            '-preset', self.encoding_params['preset'],
            '-tune', self.encoding_params['tune'],
            '-crf', self.encoding_params['crf'],
            '-maxrate', self.encoding_params['maxrate'],
            '-bufsize', self.encoding_params['bufsize'],
            '-g', self.encoding_params['keyint'],
            '-r', self.encoding_params['fps'],
            '-pix_fmt', 'yuv420p',
            '-f', 'mpegts',
            f'udp://{self.ground_station_ip}:{self.rtc_port}'
        ]
        
        try:
            self.encoder_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info(f"H.265 encoder started, streaming to {self.ground_station_ip}:{self.rtc_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start H.265 encoder: {e}")
            return False
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for telemetry and control"""
        self.connected_clients.add(websocket)
        logger.info(f"Ground station connected via WebSocket: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_ground_station_command(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Ground station disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_ground_station_command(self, websocket, data):
        """Handle commands from ground station"""
        command = data.get('command')
        
        if command == 'get_status':
            status = {
                'timestamp': datetime.now().isoformat(),
                'streaming': self.streaming,
                'encoder_running': self.encoder_process is not None and self.encoder_process.poll() is None,
                'system_stats': self.get_system_stats()
            }
            await websocket.send(json.dumps(status))
            
        elif command == 'start_stream':
            if not self.streaming:
                self.start_streaming()
            await websocket.send(json.dumps({'status': 'streaming_started'}))
            
        elif command == 'stop_stream':
            if self.streaming:
                self.stop_streaming()
            await websocket.send(json.dumps({'status': 'streaming_stopped'}))
            
        elif command == 'adjust_quality':
            quality = data.get('quality', 'medium')
            self.adjust_stream_quality(quality)
            await websocket.send(json.dumps({'status': f'quality_set_to_{quality}'}))
    
    def get_system_stats(self):
        """Get system performance statistics"""
        try:
            # CPU temperature
            temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
            
            # Memory usage
            with open('/proc/meminfo') as f:
                meminfo = f.read()
            
            # CPU usage (simplified)
            load_avg = os.getloadavg()[0]
            
            return {
                'cpu_temp': round(temp, 1),
                'cpu_load': round(load_avg, 2),
                'timestamp': time.time()
            }
        except:
            return {'error': 'stats_unavailable'}
    
    def adjust_stream_quality(self, quality):
        """Dynamically adjust streaming quality"""
        quality_presets = {
            'low': {'crf': '28', 'maxrate': '1000k', 'bufsize': '2000k'},
            'medium': {'crf': '23', 'maxrate': '2000k', 'bufsize': '4000k'},
            'high': {'crf': '20', 'maxrate': '4000k', 'bufsize': '8000k'}
        }
        
        if quality in quality_presets:
            self.encoding_params.update(quality_presets[quality])
            logger.info(f"Stream quality adjusted to {quality}")
    
    def start_websocket_server(self):
        """Start WebSocket server for ground station communication"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.websocket_server = websockets.serve(
                self.websocket_handler,
                "0.0.0.0",
                self.websocket_port
            )
            
            loop.run_until_complete(self.websocket_server)
            logger.info(f"WebSocket server started on port {self.websocket_port}")
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    def start_streaming(self):
        """Start the complete streaming pipeline"""
        if not self.streaming:
            self.streaming = True
            
            # Start WebSocket server in separate thread
            websocket_thread = threading.Thread(
                target=self.start_websocket_server,
                daemon=True
            )
            websocket_thread.start()
            
            # Start H.265 encoder
            self.start_h265_encoder()
    
    def stop_streaming(self):
        """Stop streaming and cleanup"""
        self.streaming = False
        
        if self.encoder_process:
            try:
                self.encoder_process.terminate()
                self.encoder_process.wait(timeout=5)
            except:
                self.encoder_process.kill()
        
        logger.info("Streaming stopped")
    
    async def send_telemetry(self, data):
        """Send telemetry data to connected ground stations"""
        if self.connected_clients:
            message = json.dumps({
                'type': 'telemetry',
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all connected clients
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected

class OptimizedCamera:
    """Optimized camera system using libcamera and hardware acceleration"""
    
    def __init__(self, width=1280, height=720, fps=25):
        self.width = width
        self.height = height
        self.fps = fps
        self.stream_process = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = False
        self.capture_thread = None
        
        # Kill existing processes
        self.cleanup_processes()
        time.sleep(1)
        
        # Start camera pipeline
        self.start_camera_pipeline()
        
    def cleanup_processes(self):
        """Clean up existing camera processes"""
        os.system("sudo pkill -f libcamera-vid")
        os.system("sudo pkill -f gst-launch-1.0")
        
    def start_camera_pipeline(self):
        """Start optimized libcamera pipeline"""
        
        # Libcamera command with hardware optimization
        camera_cmd = f"""
        libcamera-vid -t 0 --nopreview \
        --width {self.width} --height {self.height} \
        --framerate {self.fps} \
        --codec h264 --profile baseline --level 4.0 \
        --bitrate 8000000 \
        --intra 30 \
        --inline \
        --flush \
        --shutter 8000 \
        --gain 2.0 \
        --awb auto \
        --denoise cdn_hq \
        --contrast 1.1 \
        --brightness 0.05 \
        --sharpness 1.2 \
        -o - | \
        gst-launch-1.0 fdsrc ! queue ! h264parse ! \
        rtph264pay config-interval=1 ! \
        udpsink host=127.0.0.1 port=5000 sync=false
        """
        
        try:
            self.stream_process = subprocess.Popen(
                camera_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give camera time to initialize
            time.sleep(2)
            
            # Create GStreamer receiver
            pipeline = (
                "udpsrc port=5000 buffer-size=131072 ! "
                "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
                "rtpjitterbuffer latency=50 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink sync=false emit-signals=true drop=true max-buffers=1"
            )
            
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera pipeline")
            
            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"Camera pipeline started: {self.width}x{self.height}@{self.fps}fps")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise
    
    def _capture_frames(self):
        """Background thread for frame capture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Drop old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)
    
    def read(self):
        """Read frame from queue"""
        try:
            frame = self.frame_queue.get(timeout=0.5)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        """Release camera resources"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        if self.stream_process:
            try:
                self.stream_process.terminate()
                self.stream_process.wait(timeout=3)
            except:
                self.stream_process.kill()
        
        self.cleanup_processes()

class AdvancedShapeDetector:
    """Advanced shape detection with professional computer vision algorithms"""
    
    def __init__(self):
        # Initialize detection parameters
        self.setup_detection_parameters()
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Detection results history
        self.detection_history = deque(maxlen=50)
        
        # Adaptive processing
        self.adaptive_threshold = True
        self.auto_exposure = True
        
    def setup_detection_parameters(self):
        """Setup optimized detection parameters"""
        
        # Color ranges in HSV (optimized for various lighting conditions)
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))
            ],
            'blue': [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
            'green': [(np.array([40, 120, 70]), np.array([80, 255, 255]))],
            'yellow': [(np.array([20, 120, 70]), np.array([30, 255, 255]))]
        }
        
        # Detection thresholds
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        self.approx_epsilon = 0.02
        
        # Morphological kernels
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        
        # Kalman filters for shape tracking
        self.shape_trackers = {}
        
    def create_kalman_filter(self):
        """Create Kalman filter for shape tracking"""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                      [0, 1, 0, 1],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        return kf
    
    def preprocess_frame(self, frame):
        """Advanced preprocessing with adaptive enhancement"""
        
        # Apply CLAHE for adaptive histogram equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        denoised = cv2.bilateralFilter(enhanced, 5, 75, 75)
        
        # Convert to HSV
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        
        return hsv, enhanced
    
    def detect_shapes_advanced(self, frame):
        """Advanced shape detection with multi-stage filtering"""
        start_time = time.time()
        
        hsv, enhanced = self.preprocess_frame(frame)
        detected_objects = []
        
        # Process each color
        for color_name, ranges in self.color_ranges.items():
            # Create combined mask for color
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Morphological operations
            mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_small)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, self.kernel_medium)
            
            # Find contours
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_contour_area < area < self.max_contour_area:
                    # Shape analysis
                    shape_info = self.analyze_shape(contour, color_name)
                    if shape_info:
                        detected_objects.append(shape_info)
        
        # Update processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return detected_objects
    
    def analyze_shape(self, contour, color):
        """Detailed shape analysis and classification"""
        
        # Basic shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return None
        
        # Approximate polygon
        epsilon = self.approx_epsilon * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Moments for centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Shape classification
        shape_type = self.classify_shape(vertices, aspect_ratio, area, perimeter)
        
        # Calculate confidence
        confidence = self.calculate_confidence(contour, shape_type)
        
        return {
            'contour': contour,
            'shape': shape_type,
            'color': color,
            'center': (cx, cy),
            'area': area,
            'vertices': vertices,
            'aspect_ratio': aspect_ratio,
            'confidence': confidence,
            'timestamp': time.time()
        }
    
    def classify_shape(self, vertices, aspect_ratio, area, perimeter):
        """Advanced shape classification"""
        
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            if 0.8 <= aspect_ratio <= 1.2:
                return "square"
            else:
                return "rectangle"
        elif vertices == 5:
            return "pentagon"
        elif vertices == 6:
            return "hexagon"
        elif vertices > 6:
            # Check circularity for circles
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                return "circle"
            else:
                return "polygon"
        else:
            return "unknown"
    
    def calculate_confidence(self, contour, shape_type):
        """Calculate detection confidence"""
        
        # Basic confidence based on contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity measure
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Solidity measure
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Combine measures for confidence
        confidence = (circularity + solidity) / 2.0
        return min(confidence, 1.0)
    
    def draw_professional_overlay(self, frame, detections):
        """Draw professional overlay with telemetry information"""
        
        overlay = frame.copy()
        
        # Draw detection results
        for detection in detections:
            contour = detection['contour']
            shape = detection['shape']
            color = detection['color']
            center = detection['center']
            confidence = detection['confidence']
            
            # Color mapping
            color_map = {
                'red': (0, 0, 255),
                'blue': (255, 0, 0),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255)
            }
            
            draw_color = color_map.get(color, (255, 255, 255))
            
            # Draw contour
            cv2.drawContours(overlay, [contour], -1, draw_color, 2)
            
            # Draw center point
            cv2.circle(overlay, center, 5, draw_color, -1)
            
            # Draw label with confidence
            label = f"{color} {shape} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(overlay, 
                         (center[0] - label_size[0]//2 - 5, center[1] - 25),
                         (center[0] + label_size[0]//2 + 5, center[1] - 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(overlay, label, 
                       (center[0] - label_size[0]//2, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
        
        # Add telemetry information
        self.draw_telemetry_overlay(overlay, detections)
        
        return overlay
    
    def draw_telemetry_overlay(self, frame, detections):
        """Draw telemetry and system information overlay"""
        
        h, w = frame.shape[:2]
        
        # Calculate FPS
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        else:
            fps = 0
        
        # System information
        info_text = [
            f"FPS: {fps:.1f}",
            f"Detections: {len(detections)}",
            f"Frame: {self.frame_count}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Draw information panel
        panel_height = len(info_text) * 25 + 20
        cv2.rectangle(frame, (10, 10), (200, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, panel_height), (0, 255, 0), 2)
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Detection statistics
        if detections:
            stats_text = [f"{d['color']} {d['shape']}" for d in detections]
            
            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (w - 200, 30 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

class VisionSystem:
    """Main vision system integrating all components"""
    
    def __init__(self, ground_station_ip="192.168.1.100"):
        self.ground_station_ip = ground_station_ip
        
        # Initialize components
        self.camera = None
        self.detector = AdvancedShapeDetector()
        self.streamer = H265RTCStreamer(ground_station_ip)
        
        # System state
        self.running = False
        self.recording = False
        
        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'detections_made': 0,
            'start_time': time.time()
        }
        
    def initialize_system(self):
        """Initialize the complete vision system"""
        
        logger.info("Initializing Professional Vision System...")
        
        # Apply system optimizations
        SystemOptimizer.optimize_system()
        
        # Initialize camera
        try:
            self.camera = OptimizedCamera(width=1280, height=720, fps=25)
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
        
        # Start streaming
        self.streamer.start_streaming()
        logger.info(f"RTC streaming started to {self.ground_station_ip}")
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        logger.info("Vision system initialization complete")
    
    def process_frame(self, frame):
        """Process single frame through the vision pipeline"""
        
        # Detect shapes
        detections = self.detector.detect_shapes_advanced(frame)
        
        # Draw professional overlay
        result_frame = self.detector.draw_professional_overlay(frame, detections)
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['detections_made'] += len(detections)
        self.detector.frame_count += 1
        
        # Send telemetry to ground station
        if detections:
            telemetry = {
                'detections': len(detections),
                'objects': [
                    {
                        'shape': d['shape'],
                        'color': d['color'],
                        'position': d['center'],
                        'confidence': d['confidence']
                    }
                    for d in detections
                ],
                'frame_id': self.detector.frame_count,
                'timestamp': time.time()
            }
            
            # Send via WebSocket (async)
            asyncio.create_task(self.streamer.send_telemetry(telemetry))
        
        return result_frame, detections
    
    def run(self):
        """Main system loop"""
        
        try:
            self.initialize_system()
            self.running = True
            
            logger.info("Professional Vision System Running...")
            logger.info("Press 'q' to quit, 'r' to toggle recording")
            
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                result_frame, detections = self.process_frame(frame)
                
                # Display result
                cv2.imshow('Professional Vision System', result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Shutdown requested by user")
                    break
                elif key == ord('r'):
                    self.toggle_recording()
                elif key == ord('s'):
                    self.save_detection_snapshot(result_frame, detections)
                elif key == ord('c'):
                    self.calibrate_colors(frame)
                
                # Adaptive frame rate control
                if self.detector.processing_times:
                    avg_time = np.mean(self.detector.processing_times)
                    if avg_time > 0.05:  # If processing takes > 50ms
                        time.sleep(0.01)  # Add small delay
        
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.cleanup()
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_filename = f"/tmp/vision_streams/recording_{timestamp}.mp4"
            
            # Start recording with H.265
            self.recording_process = subprocess.Popen([
                'ffmpeg', '-y',
                '-f', 'h264',
                '-i', 'udp://127.0.0.1:5000',
                '-c:v', 'libx265',
                '-preset', 'medium',
                '-crf', '23',
                self.recording_filename
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.recording = True
            logger.info(f"Recording started: {self.recording_filename}")
        else:
            if hasattr(self, 'recording_process'):
                self.recording_process.terminate()
                self.recording_process.wait()
            
            self.recording = False
            logger.info("Recording stopped")
    
    def save_detection_snapshot(self, frame, detections):
        """Save snapshot with detection data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_path = f"/tmp/vision_streams/snapshot_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Save detection data
        data_path = f"/tmp/vision_streams/snapshot_{timestamp}.json"
        detection_data = {
            'timestamp': timestamp,
            'detections': [
                {
                    'shape': d['shape'],
                    'color': d['color'],
                    'center': d['center'],
                    'area': d['area'],
                    'confidence': d['confidence']
                }
                for d in detections
            ],
            'frame_stats': {
                'processed_frames': self.stats['frames_processed'],
                'total_detections': self.stats['detections_made']
            }
        }
        
        with open(data_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"Snapshot saved: {image_path}")
    
    def calibrate_colors(self, frame):
        """Interactive color calibration"""
        logger.info("Color calibration mode - click on objects to sample colors")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get HSV value at clicked point
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = hsv[y, x]
                logger.info(f"Color at ({x},{y}): H={h}, S={s}, V={v}")
                
                # Suggest color range
                h_range = 10
                s_range = 50
                v_range = 50
                
                lower = np.array([max(0, h-h_range), max(0, s-s_range), max(0, v-v_range)])
                upper = np.array([min(179, h+h_range), min(255, s+s_range), min(255, v+v_range)])
                
                logger.info(f"Suggested range - Lower: {lower}, Upper: {upper}")
        
        cv2.setMouseCallback('Professional Vision System', mouse_callback)
        logger.info("Click on the display window to sample colors. Press any key to exit calibration.")
    
    def get_system_performance(self):
        """Get comprehensive system performance metrics"""
        runtime = time.time() - self.stats['start_time']
        
        return {
            'runtime_seconds': runtime,
            'frames_processed': self.stats['frames_processed'],
            'detections_made': self.stats['detections_made'],
            'fps_average': self.stats['frames_processed'] / runtime if runtime > 0 else 0,
            'detections_per_minute': (self.stats['detections_made'] / runtime) * 60 if runtime > 0 else 0,
            'memory_usage': self.get_memory_usage(),
            'cpu_temperature': self.get_cpu_temperature()
        }
    
    def get_memory_usage(self):
        """Get current memory usage"""
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
            
            mem_total = int(lines[0].split()[1])
            mem_available = int(lines[2].split()[1])
            mem_used = mem_total - mem_available
            
            return {
                'total_kb': mem_total,
                'used_kb': mem_used,
                'available_kb': mem_available,
                'usage_percent': (mem_used / mem_total) * 100
            }
        except:
            return {'error': 'unable_to_read_memory'}
    
    def get_cpu_temperature(self):
        """Get CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return None
    
    def cleanup(self):
        """Comprehensive system cleanup"""
        logger.info("Cleaning up system resources...")
        
        self.running = False
        
        # Stop recording if active
        if self.recording and hasattr(self, 'recording_process'):
            try:
                self.recording_process.terminate()
                self.recording_process.wait(timeout=5)
            except:
                self.recording_process.kill()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Stop streaming
        if self.streamer:
            self.streamer.stop_streaming()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # System cleanup
        SystemOptimizer.cleanup()
        
        # Print final performance statistics
        performance = self.get_system_performance()
        logger.info("Final Performance Statistics:")
        logger.info(f"  Runtime: {performance['runtime_seconds']:.1f} seconds")
        logger.info(f"  Frames Processed: {performance['frames_processed']}")
        logger.info(f"  Total Detections: {performance['detections_made']}")
        logger.info(f"  Average FPS: {performance['fps_average']:.1f}")
        logger.info(f"  Detections/min: {performance['detections_per_minute']:.1f}")
        
        logger.info("Professional Vision System shutdown complete")


def setup_signal_handlers(vision_system):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        if vision_system:
            vision_system.running = False
            vision_system.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_system_requirements():
    """Check system requirements and dependencies"""
    
    logger.info("Checking system requirements...")
    
    # Check required modules
    required_modules = [
        'cv2', 'numpy', 'websockets', 'asyncio'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        logger.error("Install with: pip install opencv-python numpy websockets")
        return False
    
    # Check OpenCV build
    build_info = cv2.getBuildInformation()
    
    if 'GStreamer' not in build_info:
        logger.warning("OpenCV not built with GStreamer support")
        logger.warning("Some features may not work correctly")
    
    if 'OpenCL' in build_info:
        logger.info("OpenCL support detected - hardware acceleration available")
    
    # Check hardware
    if os.path.exists('/opt/vc/bin/vcgencmd'):
        logger.info("Raspberry Pi hardware detected")
    else:
        logger.warning("Not running on Raspberry Pi - some optimizations may not work")
    
    # Check available commands
    required_commands = ['ffmpeg', 'libcamera-vid', 'gst-launch-1.0']
    missing_commands = []
    
    for cmd in required_commands:
        try:
            subprocess.run(['which', cmd], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing_commands.append(cmd)
    
    if missing_commands:
        logger.error(f"Missing required commands: {', '.join(missing_commands)}")
        logger.error("Install with: sudo apt install ffmpeg gstreamer1.0-tools")
        return False
    
    logger.info("All system requirements satisfied")
    return True


def main():
    """Main function"""
    
    print("=" * 60)
    print("  PROFESSIONAL RASPBERRY PI 4 VISION SYSTEM")
    print("  H.265 RTC Streaming & Advanced Computer Vision")
    print("  Ground Station Communication Platform")
    print("=" * 60)
    print()
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements not met. Exiting.")
        return 1
    
    # Get ground station IP from command line or use default
    ground_station_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.100"
    
    logger.info(f"Ground station IP: {ground_station_ip}")
    
    # Initialize vision system
    vision_system = VisionSystem(ground_station_ip)
    
    # Setup signal handlers
    setup_signal_handlers(vision_system)
    
    try:
        # Run the system
        vision_system.run()
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
