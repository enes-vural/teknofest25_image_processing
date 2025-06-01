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
import socket
import struct
import json
import logging
from datetime import datetime
import hashlib
import asyncio
import websockets
from typing import Optional, Tuple, Dict, Any

# LoRa communication (requires pyLoRa library)
try:
    import pyLoRa
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("WARNING: LoRa library not available. LoRa features disabled.")

# Enable OpenCV optimizations (uses NEON SIMD instructions on ARM if available)
cv2.setUseOptimized(True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/drone_vision.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure OpenCV is using optimized code paths
if cv2.useOptimized():
    logger.info("OpenCV optimizations enabled (using hardware acceleration)")
else:
    logger.warning("OpenCV optimizations not available")

# Set Raspberry Pi to performance mode (requires appropriate permissions)
try:
    os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")
    logger.info("Set CPU to performance mode")
    
    # Reset to ondemand governor on exit
    def reset_governor():
        os.system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")
    
    atexit.register(reset_governor)
except:
    logger.warning("Could not set CPU governor (may need sudo)")


class StreamingConfig:
    """Configuration class for streaming parameters"""
    
    # Main video stream (processed video)
    MAIN_STREAM_PORT = 5000
    RAW_STREAM_PORT = 5001
    
    # Control and telemetry
    CONTROL_PORT = 8080
    TELEMETRY_PORT = 8081
    
    # Stream quality settings
    HIGH_QUALITY = {
        'width': 1280, 'height': 720, 'fps': 30, 'bitrate': 2000000
    }
    MEDIUM_QUALITY = {
        'width': 854, 'height': 480, 'fps': 25, 'bitrate': 1000000
    }
    LOW_QUALITY = {
        'width': 640, 'height': 360, 'fps': 15, 'bitrate': 500000
    }
    
    # Network settings
    MAX_PACKET_SIZE = 1400  # MTU safe size
    RETRY_ATTEMPTS = 3
    HEARTBEAT_INTERVAL = 1.0
    CONNECTION_TIMEOUT = 5.0


class NetworkStreamManager:
    """Professional network streaming manager with fail-safe mechanisms"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.active_clients = {}
        self.stream_sockets = {}
        self.running = False
        
        # Streaming statistics
        self.stats = {
            'bytes_sent': 0,
            'packets_sent': 0,
            'failed_attempts': 0,
            'active_connections': 0,
            'last_heartbeat': time.time()
        }
        
        # Quality adaptation
        self.current_quality = config.MEDIUM_QUALITY
        self.quality_history = deque(maxlen=10)
        
        # Threading
        self.heartbeat_thread = None
        self.stats_thread = None
        
    def initialize_sockets(self):
        """Initialize UDP sockets for streaming"""
        try:
            # Main processed stream
            self.stream_sockets['main'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.stream_sockets['main'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.stream_sockets['main'].setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB buffer
            
            # Raw stream
            self.stream_sockets['raw'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.stream_sockets['raw'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.stream_sockets['raw'].setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512*1024)   # 512KB buffer
            
            # Control socket (TCP for reliability)
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_socket.bind(('0.0.0.0', self.config.CONTROL_PORT))
            self.control_socket.listen(5)
            
            logger.info("Network sockets initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sockets: {e}")
            return False
    
    def start_streaming(self):
        """Start the streaming service"""
        if not self.initialize_sockets():
            return False
            
        self.running = True
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        
        # Start statistics thread
        self.stats_thread = threading.Thread(target=self._stats_worker, daemon=True)
        self.stats_thread.start()
        
        # Start control server
        self.control_thread = threading.Thread(target=self._control_server, daemon=True)
        self.control_thread.start()
        
        logger.info("Streaming service started")
        return True
    
    def stop_streaming(self):
        """Stop streaming service gracefully"""
        self.running = False
        
        # Close all sockets
        for sock in self.stream_sockets.values():
            try:
                sock.close()
            except:
                pass
                
        try:
            self.control_socket.close()
        except:
            pass
            
        logger.info("Streaming service stopped")
    
    def send_frame(self, frame_data: bytes, stream_type: str, clients: list = None):
        """Send frame data to clients with error handling"""
        if not self.running or stream_type not in self.stream_sockets:
            return False
            
        if clients is None:
            clients = list(self.active_clients.keys())
            
        if not clients:
            return False
            
        sock = self.stream_sockets[stream_type]
        success_count = 0
        
        # Fragment large frames
        fragments = self._fragment_data(frame_data)
        
        for client_addr in clients:
            try:
                for fragment in fragments:
                    sock.sendto(fragment, client_addr)
                    
                success_count += 1
                self.stats['packets_sent'] += len(fragments)
                
            except Exception as e:
                logger.warning(f"Failed to send to client {client_addr}: {e}")
                self.stats['failed_attempts'] += 1
                self._handle_client_disconnect(client_addr)
        
        self.stats['bytes_sent'] += len(frame_data) * success_count
        return success_count > 0
    
    def _fragment_data(self, data: bytes) -> list:
        """Fragment large data into network-safe packets"""
        if len(data) <= self.config.MAX_PACKET_SIZE:
            return [data]
            
        fragments = []
        fragment_id = int(time.time() * 1000) % 65536  # 16-bit fragment ID
        total_fragments = (len(data) + self.config.MAX_PACKET_SIZE - 1) // self.config.MAX_PACKET_SIZE
        
        for i in range(total_fragments):
            start = i * self.config.MAX_PACKET_SIZE
            end = min(start + self.config.MAX_PACKET_SIZE, len(data))
            
            # Header: fragment_id (2 bytes) + fragment_num (2 bytes) + total_fragments (2 bytes)
            header = struct.pack('!HHH', fragment_id, i, total_fragments)
            fragment = header + data[start:end]
            fragments.append(fragment)
            
        return fragments
    
    def _heartbeat_worker(self):
        """Send periodic heartbeat to maintain connections"""
        while self.running:
            current_time = time.time()
            heartbeat_data = json.dumps({
                'type': 'heartbeat',
                'timestamp': current_time,
                'stats': self.stats
            }).encode()
            
            # Send to all active clients
            disconnected_clients = []
            for client_addr in list(self.active_clients.keys()):
                try:
                    self.stream_sockets['main'].sendto(heartbeat_data, client_addr)
                    
                    # Check if client is responsive
                    if current_time - self.active_clients[client_addr]['last_seen'] > self.config.CONNECTION_TIMEOUT:
                        disconnected_clients.append(client_addr)
                        
                except:
                    disconnected_clients.append(client_addr)
            
            # Remove disconnected clients
            for client_addr in disconnected_clients:
                self._handle_client_disconnect(client_addr)
            
            self.stats['last_heartbeat'] = current_time
            time.sleep(self.config.HEARTBEAT_INTERVAL)
    
    def _stats_worker(self):
        """Monitor and log streaming statistics"""
        while self.running:
            time.sleep(10)  # Log every 10 seconds
            
            logger.info(f"Streaming Stats - Active: {len(self.active_clients)}, "
                       f"Sent: {self.stats['bytes_sent']/1024/1024:.1f}MB, "
                       f"Packets: {self.stats['packets_sent']}, "
                       f"Failed: {self.stats['failed_attempts']}")
    
    def _control_server(self):
        """Handle control connections from clients"""
        while self.running:
            try:
                client_sock, client_addr = self.control_socket.accept()
                threading.Thread(
                    target=self._handle_control_client, 
                    args=(client_sock, client_addr),
                    daemon=True
                ).start()
                
            except:
                if self.running:
                    logger.error("Control server error")
                    time.sleep(1)
    
    def _handle_control_client(self, client_sock, client_addr):
        """Handle individual control client"""
        logger.info(f"Control client connected: {client_addr}")
        
        try:
            while self.running:
                data = client_sock.recv(1024)
                if not data:
                    break
                    
                try:
                    command = json.loads(data.decode())
                    response = self._process_control_command(command, client_addr)
                    client_sock.send(json.dumps(response).encode())
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_addr}")
                    
        except Exception as e:
            logger.warning(f"Control client {client_addr} error: {e}")
        finally:
            client_sock.close()
            logger.info(f"Control client disconnected: {client_addr}")
    
    def _process_control_command(self, command: dict, client_addr) -> dict:
        """Process control commands from clients"""
        cmd_type = command.get('type', '')
        
        if cmd_type == 'register':
            # Register client for streaming
            stream_addr = (client_addr[0], command.get('stream_port', self.config.MAIN_STREAM_PORT))
            self.active_clients[stream_addr] = {
                'registered_at': time.time(),
                'last_seen': time.time(),
                'stream_type': command.get('stream_type', 'main'),
                'quality': command.get('quality', 'medium')
            }
            logger.info(f"Client registered: {stream_addr}")
            return {'status': 'success', 'message': 'Client registered'}
            
        elif cmd_type == 'unregister':
            stream_addr = (client_addr[0], command.get('stream_port', self.config.MAIN_STREAM_PORT))
            self._handle_client_disconnect(stream_addr)
            return {'status': 'success', 'message': 'Client unregistered'}
            
        elif cmd_type == 'quality_change':
            quality = command.get('quality', 'medium')
            self._adapt_quality(quality)
            return {'status': 'success', 'message': f'Quality changed to {quality}'}
            
        elif cmd_type == 'get_stats':
            return {'status': 'success', 'stats': self.stats}
            
        else:
            return {'status': 'error', 'message': 'Unknown command'}
    
    def _handle_client_disconnect(self, client_addr):
        """Handle client disconnection"""
        if client_addr in self.active_clients:
            del self.active_clients[client_addr]
            logger.info(f"Client disconnected: {client_addr}")
            self.stats['active_connections'] = len(self.active_clients)
    
    def _adapt_quality(self, quality_level: str):
        """Adapt streaming quality based on network conditions"""
        quality_map = {
            'high': self.config.HIGH_QUALITY,
            'medium': self.config.MEDIUM_QUALITY,
            'low': self.config.LOW_QUALITY
        }
        
        if quality_level in quality_map:
            self.current_quality = quality_map[quality_level]
            logger.info(f"Quality adapted to: {quality_level}")


class LoRaTelemetry:
    """LoRa communication for telemetry and low-bandwidth data"""
    
    def __init__(self, cs_pin=8, reset_pin=22, irq_pin=18):
        self.lora = None
        self.running = False
        self.message_queue = queue.Queue(maxsize=100)
        
        if LORA_AVAILABLE:
            try:
                self.lora = pyLoRa.LoRa(cs_pin, reset_pin, irq_pin)
                self.lora.set_frequency(868.1)  # EU frequency
                self.lora.set_bandwidth(125000)
                self.lora.set_spreading_factor(7)
                self.lora.set_coding_rate(5)
                self.lora.set_sync_word(0x34)
                logger.info("LoRa initialized successfully")
            except Exception as e:
                logger.error(f"LoRa initialization failed: {e}")
                self.lora = None
    
    def start_telemetry(self):
        """Start LoRa telemetry service"""
        if not self.lora:
            logger.warning("LoRa not available, telemetry disabled")
            return False
            
        self.running = True
        self.telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self.telemetry_thread.start()
        logger.info("LoRa telemetry started")
        return True
    
    def stop_telemetry(self):
        """Stop LoRa telemetry service"""
        self.running = False
        if hasattr(self, 'telemetry_thread'):
            self.telemetry_thread.join(timeout=1.0)
        logger.info("LoRa telemetry stopped")
    
    def send_telemetry(self, data: dict):
        """Queue telemetry data for LoRa transmission"""
        if not self.running:
            return False
            
        try:
            # Compress telemetry data
            compressed_data = json.dumps(data, separators=(',', ':'))
            if len(compressed_data) > 200:  # LoRa packet size limit
                # Send only critical data
                critical_data = {
                    'lat': data.get('latitude', 0),
                    'lon': data.get('longitude', 0),
                    'alt': data.get('altitude', 0),
                    'bat': data.get('battery', 0),
                    'sig': data.get('signal_strength', 0)
                }
                compressed_data = json.dumps(critical_data, separators=(',', ':'))
            
            self.message_queue.put(compressed_data, block=False)
            return True
            
        except queue.Full:
            logger.warning("LoRa telemetry queue full")
            return False
    
    def _telemetry_worker(self):
        """LoRa telemetry transmission worker"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.message_queue.get(timeout=1.0)
                
                # Send via LoRa
                if self.lora:
                    self.lora.send(message.encode())
                    logger.debug(f"LoRa telemetry sent: {len(message)} bytes")
                    
                time.sleep(0.1)  # Rate limiting
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"LoRa transmission error: {e}")
                time.sleep(1)


class EnhancedGstreamerCamera:
    """Enhanced camera class with dual streaming support"""
    
    def __init__(self, queue_size=2, enable_raw_stream=True):
        # Kill any existing processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        time.sleep(1)
        
        self.enable_raw_stream = enable_raw_stream
        
        # Start streaming processes
        self.main_stream_process = self.start_main_stream()
        if enable_raw_stream:
            self.raw_stream_process = self.start_raw_stream()
        
        # Create frame queues
        self.main_queue = queue.Queue(maxsize=queue_size)
        self.raw_queue = queue.Queue(maxsize=queue_size) if enable_raw_stream else None
        
        # Control flags
        self.running = True
        
        # Open GStreamer pipelines
        self.main_cap = self.receive_main_stream()
        self.raw_cap = self.receive_raw_stream() if enable_raw_stream else None
        
        # Verify pipelines
        if not self.main_cap.isOpened():
            raise RuntimeError("Failed to open main GStreamer pipeline")
        if enable_raw_stream and not self.raw_cap.isOpened():
            logger.warning("Failed to open raw stream pipeline")
            
        # Start capture threads
        self.main_thread = threading.Thread(target=self._update_main, daemon=True)
        self.main_thread.start()
        
        if enable_raw_stream and self.raw_cap:
            self.raw_thread = threading.Thread(target=self._update_raw, daemon=True)
            self.raw_thread.start()
    
    def start_main_stream(self):
        """Start main processed video stream"""
        cmd = """
        libcamera-vid -t 0 \
        --nopreview \
        --width 854 --height 480 \
        --framerate 25 \
        --shutter 20000 \
        --gain 3 \
        --denoise cdn_off \
        --brightness 0.1 \
        --contrast 1.2 \
        --sharpness 1.5 \
        --codec h264 \
        --bitrate 1000000 \
        --inline \
        -o - | \
        gst-launch-1.0 fdsrc ! queue ! h264parse ! rtph264pay config-interval=1 ! udpsink host=127.0.0.1 port=5100 sync=false
        """
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def start_raw_stream(self):
        """Start raw video stream"""
        cmd = """
        libcamera-vid -t 0 \
        --nopreview \
        --width 640 --height 480 \
        --framerate 15 \
        --codec h264 \
        --bitrate 500000 \
        --inline \
        -o - | \
        gst-launch-1.0 fdsrc ! queue ! h264parse ! rtph264pay config-interval=1 ! udpsink host=127.0.0.1 port=5101 sync=false
        """
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def receive_main_stream(self):
        """Create GStreamer pipeline for main stream"""
        pipeline = (
            "udpsrc port=5100 ! "
            "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=50 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink sync=false emit-signals=true drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    def receive_raw_stream(self):
        """Create GStreamer pipeline for raw stream"""
        if not self.enable_raw_stream:
            return None
            
        pipeline = (
            "udpsrc port=5101 ! "
            "application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=50 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink sync=false emit-signals=true drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    def _update_main(self):
        """Main stream capture thread"""
        while self.running:
            ret, frame = self.main_cap.read()
            if ret:
                if self.main_queue.full():
                    try:
                        self.main_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.main_queue.put(frame)
            else:
                time.sleep(0.01)
    
    def _update_raw(self):
        """Raw stream capture thread"""
        if not self.raw_cap:
            return
            
        while self.running:
            ret, frame = self.raw_cap.read()
            if ret:
                if self.raw_queue.full():
                    try:
                        self.raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.raw_queue.put(frame)
            else:
                time.sleep(0.01)
    
    def read_main(self):
        """Read main processed frame"""
        try:
            frame = self.main_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def read_raw(self):
        """Read raw frame"""
        if not self.raw_queue:
            return False, None
            
        try:
            frame = self.raw_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        """Release all resources"""
        self.running = False
        
        # Wait for threads
        if hasattr(self, 'main_thread'):
            self.main_thread.join(timeout=1.0)
        if hasattr(self, 'raw_thread'):
            self.raw_thread.join(timeout=1.0)
        
        # Release captures
        if self.main_cap:
            self.main_cap.release()
        if self.raw_cap:
            self.raw_cap.release()
        
        # Terminate processes
        for process in [getattr(self, 'main_stream_process', None), 
                       getattr(self, 'raw_stream_process', None)]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                    except:
                        pass
        
        # Clean up
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")


class ShapeDetector:
    """Enhanced shape detector with streaming integration"""
    
    def __init__(self):
        # Original shape detection code (keeping all the optimizations)
        self.red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
        self.red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([160, 100, 100], dtype=np.uint8)
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self.blue_lower = np.array([100, 100, 100], dtype=np.uint8)
        self.blue_upper = np.array([140, 255, 255], dtype=np.uint8)
        
        self.min_contour_area = 7500
        self.approx_polygon_epsilon = 0.015
        self.kernel3 = np.ones((3, 3), np.uint8)
        
        self.hsv_buffer = None
        self.red_mask = None
        self.blue_mask = None
        
        self.fps_buffer = deque(maxlen=10)
        self.prev_frame_time = 0
        self.color_intensity_threshold = 0.4
        self.frame_count = 0
        self.process_every_n_frames = 1
        
        # Detection results for telemetry
        self.latest_detections = []
        
    # Keep all original methods but add telemetry data collection
    def preprocess_frame(self, frame):
        if self.hsv_buffer is None or self.hsv_buffer.shape[:2] != frame.shape[:2]:
            h, w = frame.shape[:2]
            self.hsv_buffer = np.empty((h, w, 3), dtype=np.uint8)
            self.red_mask = np.empty((h, w), dtype=np.uint8)
            self.blue_mask = np.empty((h, w), dtype=np.uint8)
        
        blurred = cv2.medianBlur(frame, 3)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV, dst=self.hsv_buffer)
        return hsv

    def create_color_masks(self, hsv_frame):
        red_mask1 = cv2.inRange(hsv_frame, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_frame, self.red_lower2, self.red_upper2)
        cv2.bitwise_or(red_mask1, red_mask2, dst=self.red_mask)
        
        blue_mask = cv2.inRange(hsv_frame, self.blue_lower, self.blue_upper)
        
        cv2.dilate(self.red_mask, self.kernel3, dst=self.red_mask, iterations=1)
        cv2.dilate(blue_mask, self.kernel3, dst=self.blue_mask, iterations=1)
        
        return self.red_mask, self.blue_mask

    def detect_shapes(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.process_every_n_frames != 0:
            return self.latest_detections  # Return cached results
        
        hsv_frame = self.preprocess_frame(frame)
        red_mask, blue_mask = self.create_color_masks(hsv_frame)
        
        frame_area = frame.shape[0] * frame.shape[1]
        threshold_pixels = frame_area * self.color_intensity_threshold
        
        red_pixels = np.sum(red_mask == 255)
        blue_pixels = np.sum(blue_mask == 255)
        
        red_close = red_pixels > threshold_pixels
        blue_close = blue_pixels > threshold_pixels
        
        if red_close:
            logger.info("Red shape detected - very close")
        if blue_close:
            logger.info("Blue shape detected - very close")
        
        red_shapes = self.process_contours(red_mask, frame, "red")
        blue_shapes = self.process_contours(blue_mask, frame, "blue")
        
        self.latest_detections = red_shapes + blue_shapes
        return self.latest_detections

    def process_contours(self, mask, frame, color):
        detected_shapes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        if len(contours) > 5:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_polygon_epsilon * peri, True)
            corners = len(approx)
            
            shape_type = None
            
            if color == "red":
                if corners == 3:
                    shape_type = "triangle"
                elif corners == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        shape_type = "square"
            else:  # color == "blue"
                if corners == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        shape_type = "square"
                elif 5 <= corners <= 7:
                    shape_type = "hexagon"
            
            if shape_type:
                # Calculate centroid for telemetry
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_shapes.append((contour, shape_type, color, (cx, cy), area))
                
        return detected_shapes

    def draw_results(self, frame, shapes):
        if not shapes:
            return frame
            
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)
        white_color = (255, 255, 255)
        
        for shape_data in shapes:
            contour, shape_type, color = shape_data[:3]
            color_bgr = red_color if color == "red" else blue_color
            cv2.drawContours(frame, [contour], -1, color_bgr, 2)
            
            if len(shape_data) >= 4:
                cx, cy = shape_data[3]
                label = f"{color} {shape_type}"
                cv2.putText(frame, label, (cx - 20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, white_color, 1)
        
        return frame

    def calculate_fps(self):
        current_time = time.time()
        delta_time = current_time - self.prev_frame_time
        
        if delta_time > 0.001:
            fps = 1.0 / delta_time
            self.fps_buffer.append(fps)
            self.prev_frame_time = current_time
        
        if self.fps_buffer:
            return np.mean(self.fps_buffer)
        return 0

    def process_frame(self, frame):
        start_time = time.time()
        fps = self.calculate_fps()
        
        shapes = self.detect_shapes(frame)
        self.draw_results(frame, shapes)
        
        # Add system info to frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Shapes: {len(shapes)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dynamic frame processing adjustment
        elapsed_time = time.time() - start_time
        target_time = 1.0 / 20.0
        
        if elapsed_time > target_time:
            self.process_every_n_frames = min(3, self.process_every_n_frames + 1)
        elif elapsed_time < target_time * 0.7 and self.process_every_n_frames > 1:
            self.process_every_n_frames -= 1
            
        return frame, shapes, fps

    def get_telemetry_data(self, shapes, fps):
        """Generate telemetry data from detection results"""
        telemetry = {
            'timestamp': time.time(),
            'fps': fps,
            'shape_count': len(shapes),
            'shapes': []
        }
        
        for shape_data in shapes:
            if len(shape_data) >= 5:
                _, shape_type, color, (cx, cy), area = shape_data
                telemetry['shapes'].append({
                    'type': shape_type,
                    'color': color,
                    'position': [cx, cy],
                    'area': int(area)
                })
        
        return telemetry


class DroneVisionSystem:
    """Main drone vision system with professional streaming capabilities"""
    
    def __init__(self):
        self.config = StreamingConfig()
        self.stream_manager = NetworkStreamManager(self.config)
        self.lora_telemetry = LoRaTelemetry()
        self.camera = None
        self.detector = ShapeDetector()
        
        # System state
        self.running = False
        self.system_stats = {
            'start_time': time.time(),
            'frames_processed': 0,
            'detection_count': 0,
            'stream_errors': 0
        }
        
        # H.264 encoders for streaming
        self.main_encoder = None
        self.raw_encoder = None
        
        # Frame buffers for streaming
        self.encoded_main_buffer = queue.Queue(maxsize=5)
        self.encoded_raw_buffer = queue.Queue(maxsize=5)
        
    def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing Drone Vision System...")
        
        try:
            # Initialize camera
            self.camera = EnhancedGstreamerCamera(enable_raw_stream=True)
            logger.info("Camera initialized")
            
            # Initialize streaming
            if not self.stream_manager.start_streaming():
                raise RuntimeError("Failed to start streaming service")
            logger.info("Streaming service started")
            
            # Initialize LoRa telemetry
            self.lora_telemetry.start_telemetry()
            logger.info("LoRa telemetry initialized")
            
            # Initialize H.264 encoders
            self._initialize_encoders()
            logger.info("Video encoders initialized")
            
            logger.info("Drone Vision System fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _initialize_encoders(self):
        """Initialize H.264 encoders for streaming"""
        # Main stream encoder (processed video)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        # We'll use GStreamer pipelines for encoding instead of OpenCV VideoWriter
        # This provides better performance and streaming capabilities
        
        # Main encoder pipeline
        self.main_encoder_pipeline = (
            "appsrc ! videoconvert ! "
            "x264enc bitrate=1000 speed-preset=ultrafast tune=zerolatency ! "
            "h264parse ! rtph264pay config-interval=1 pt=96 ! "
            "udpsink host=0.0.0.0 port=5002 sync=false"
        )
        
        # Raw encoder pipeline  
        self.raw_encoder_pipeline = (
            "appsrc ! videoconvert ! "
            "x264enc bitrate=500 speed-preset=ultrafast tune=zerolatency ! "
            "h264parse ! rtph264pay config-interval=1 pt=96 ! "
            "udpsink host=0.0.0.0 port=5003 sync=false"
        )
        
        # Create GStreamer writers
        try:
            self.main_encoder = cv2.VideoWriter(
                self.main_encoder_pipeline, 
                cv2.CAP_GSTREAMER, 
                0, 25.0, (854, 480), True
            )
            
            self.raw_encoder = cv2.VideoWriter(
                self.raw_encoder_pipeline,
                cv2.CAP_GSTREAMER,
                0, 15.0, (640, 480), True
            )
            
            if not self.main_encoder.isOpened():
                logger.warning("Main encoder failed to open")
            if not self.raw_encoder.isOpened():
                logger.warning("Raw encoder failed to open")
                
        except Exception as e:
            logger.error(f"Encoder initialization failed: {e}")
    
    def start_system(self):
        """Start the main system loop"""
        if not self.initialize_system():
            return False
            
        self.running = True
        logger.info("Starting main system loop...")
        
        # Start encoding threads
        self.main_encoding_thread = threading.Thread(target=self._main_encoding_worker, daemon=True)
        self.raw_encoding_thread = threading.Thread(target=self._raw_encoding_worker, daemon=True)
        
        self.main_encoding_thread.start()
        self.raw_encoding_thread.start()
        
        # Main processing loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.stop_system()
            
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        last_telemetry_time = 0
        telemetry_interval = 2.0  # Send telemetry every 2 seconds
        
        while self.running:
            try:
                # Read frames from camera
                ret_main, main_frame = self.camera.read_main()
                ret_raw, raw_frame = self.camera.read_raw()
                
                if not ret_main:
                    logger.warning("Failed to read main frame")
                    time.sleep(0.01)
                    continue
                
                # Process main frame (shape detection)
                processed_frame, shapes, fps = self.detector.process_frame(main_frame.copy())
                
                # Update statistics
                self.system_stats['frames_processed'] += 1
                if shapes:
                    self.system_stats['detection_count'] += len(shapes)
                
                # Queue frames for encoding
                try:
                    if self.encoded_main_buffer.full():
                        self.encoded_main_buffer.get_nowait()  # Remove oldest
                    self.encoded_main_buffer.put(processed_frame, block=False)
                except queue.Full:
                    pass
                
                if ret_raw and raw_frame is not None:
                    try:
                        if self.encoded_raw_buffer.full():
                            self.encoded_raw_buffer.get_nowait()  # Remove oldest
                        self.encoded_raw_buffer.put(raw_frame, block=False)
                    except queue.Full:
                        pass
                
                # Send telemetry via LoRa
                current_time = time.time()
                if current_time - last_telemetry_time > telemetry_interval:
                    telemetry_data = self._generate_telemetry(shapes, fps)
                    self.lora_telemetry.send_telemetry(telemetry_data)
                    last_telemetry_time = current_time
                
                # Display local preview (optional)
                if hasattr(self, 'show_preview') and self.show_preview:
                    cv2.imshow('Drone Vision - Main', processed_frame)
                    if ret_raw:
                        cv2.imshow('Drone Vision - Raw', raw_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.system_stats['stream_errors'] += 1
                time.sleep(0.1)
    
    def _main_encoding_worker(self):
        """Worker thread for encoding and streaming main video"""
        while self.running:
            try:
                frame = self.encoded_main_buffer.get(timeout=1.0)
                
                # Encode with GStreamer
                if self.main_encoder and self.main_encoder.isOpened():
                    self.main_encoder.write(frame)
                
                # Also create raw encoded data for UDP streaming
                ret, encoded_data = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    # Send to active clients
                    clients = [addr for addr, info in self.stream_manager.active_clients.items() 
                              if info.get('stream_type') == 'main']
                    
                    if clients:
                        self.stream_manager.send_frame(encoded_data.tobytes(), 'main', clients)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Main encoding error: {e}")
                time.sleep(0.1)
    
    def _raw_encoding_worker(self):
        """Worker thread for encoding and streaming raw video"""
        while self.running:
            try:
                frame = self.encoded_raw_buffer.get(timeout=1.0)
                
                # Encode with GStreamer
                if self.raw_encoder and self.raw_encoder.isOpened():
                    self.raw_encoder.write(frame)
                
                # Also create raw encoded data for UDP streaming
                ret, encoded_data = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 60])  # Lower quality for raw
                
                if ret:
                    # Send to active clients
                    clients = [addr for addr, info in self.stream_manager.active_clients.items() 
                              if info.get('stream_type') == 'raw']
                    
                    if clients:
                        self.stream_manager.send_frame(encoded_data.tobytes(), 'raw', clients)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Raw encoding error: {e}")
                time.sleep(0.1)
    
    def _generate_telemetry(self, shapes, fps):
        """Generate comprehensive telemetry data"""
        current_time = time.time()
        uptime = current_time - self.system_stats['start_time']
        
        telemetry = {
            'timestamp': current_time,
            'uptime': uptime,
            'system': {
                'fps': fps,
                'frames_processed': self.system_stats['frames_processed'],
                'detection_count': self.system_stats['detection_count'],
                'stream_errors': self.system_stats['stream_errors'],
                'active_clients': len(self.stream_manager.active_clients),
                'cpu_temp': self._get_cpu_temperature(),
                'memory_usage': self._get_memory_usage()
            },
            'detections': self.detector.get_telemetry_data(shapes, fps),
            'network': {
                'bytes_sent': self.stream_manager.stats['bytes_sent'],
                'packets_sent': self.stream_manager.stats['packets_sent'],
                'failed_attempts': self.stream_manager.stats['failed_attempts']
            }
        }
        
        return telemetry
    
    def _get_cpu_temperature(self):
        """Get Raspberry Pi CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
                return temp
        except:
            return 0.0
    
    def _get_memory_usage(self):
        """Get system memory usage percentage"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                
            mem_total = 0
            mem_available = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1])
            
            if mem_total > 0:
                usage = ((mem_total - mem_available) / mem_total) * 100
                return round(usage, 1)
        except:
            pass
        return 0.0
    
    def stop_system(self):
        """Stop all system components gracefully"""
        logger.info("Stopping Drone Vision System...")
        
        self.running = False
        
        # Stop streaming
        self.stream_manager.stop_streaming()
        
        # Stop telemetry
        self.lora_telemetry.stop_telemetry()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close encoders
        if self.main_encoder:
            self.main_encoder.release()
        if self.raw_encoder:
            self.raw_encoder.release()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Clean up processes
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        
        logger.info("Drone Vision System stopped")


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info("Received signal to terminate. Cleaning up...")
        os.system("sudo pkill -9 libcamera-vid")
        os.system("sudo pkill -9 gst-launch-1.0")
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function for drone vision system"""
    setup_signal_handlers()
    
    # Set process priority
    try:
        os.nice(-10)
        logger.info("Process priority increased")
    except:
        logger.warning("Could not set process priority")
    
    # Set CPU affinity
    try:
        cores = "2-3" if os.cpu_count() >= 4 else "0-1"
        os.system(f"taskset -cp {cores} {os.getpid()} > /dev/null")
        logger.info(f"Process bound to CPU cores {cores}")
    except:
        logger.warning("Could not set CPU affinity")
    
    # Clean up any existing processes
    os.system("sudo pkill -9 libcamera-vid")
    os.system("sudo pkill -9 gst-launch-1.0")
    time.sleep(1)
    
    # Create and start the drone vision system
    try:
        drone_system = DroneVisionSystem()
        
        # Enable preview for debugging (set to False for headless operation)
        drone_system.show_preview = True
        
        logger.info("=== Professional Drone Vision System Starting ===")
        logger.info(f"Main Stream Port: {StreamingConfig.MAIN_STREAM_PORT}")
        logger.info(f"Raw Stream Port: {StreamingConfig.RAW_STREAM_PORT}")
        logger.info(f"Control Port: {StreamingConfig.CONTROL_PORT}")
        logger.info(f"H.264 Main Stream: udp://0.0.0.0:5002")
        logger.info(f"H.264 Raw Stream: udp://0.0.0.0:5003")
        
        if not drone_system.start_system():
            logger.error("Failed to start drone vision system")
            return 1
            
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Check dependencies
    import sys
    required_modules = ['cv2', 'numpy', 'subprocess', 'websockets']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            if module != 'websockets':  # websockets is optional
                missing_modules.append(module)
    
    if missing_modules:
        print(f"ERROR: Missing required modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + ' '.join(missing_modules))
        sys.exit(1)
    
    # Check GStreamer support
    if not cv2.getBuildInformation().find('GStreamer') != -1:
        print("WARNING: OpenCV was not built with GStreamer support.")
    
    # Start system
    sys.exit(main())
