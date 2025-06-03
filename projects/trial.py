#!/usr/bin/env python3
"""
Raspberry Pi Camera UDP Streamer - H.265
Basit UDP streaming - hiçbir ek kurulum gerektirmez
"""

import cv2
import socket
import struct
import time
import threading
import subprocess
import os

class RaspberryStreamer:
    def __init__(self, port=9999, quality=23, fps=25):
        self.port = port
        self.quality = quality
        self.fps = fps
        self.streaming = False
        self.broadcast_socket = None
        self.clients = set()
        
        # Kendi IP adresini bul
        self.my_ip = self.get_local_ip()
        print(f"Raspberry Pi IP: {self.my_ip}")
        print(f"Stream Port: {self.port}")
        
    def get_local_ip(self):
        """Kendi IP adresini bul"""
        try:
            # En basit yöntem
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def broadcast_presence(self):
        """Varlığını ağa duyur"""
        try:
            broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Broadcast mesajı
            message = f"RASPBERRY_STREAM:{self.my_ip}:{self.port}".encode()
            broadcast_addr = self.get_broadcast_ip()
            
            while self.streaming:
                try:
                    broadcast_sock.sendto(message, (broadcast_addr, 8888))
                    time.sleep(5)  # 5 saniyede bir duyur
                except:
                    pass
            
            broadcast_sock.close()
        except Exception as e:
            print(f"Broadcast hatası: {e}")
    
    def get_broadcast_ip(self):
        """Broadcast IP'yi hesapla"""
        ip_parts = self.my_ip.split('.')
        return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.255"
    
    def start_streaming(self):
        """Streaming başlat"""
        self.streaming = True
        
        # Broadcast thread başlat
        broadcast_thread = threading.Thread(target=self.broadcast_presence)
        broadcast_thread.daemon = True
        broadcast_thread.start()
        
        # FFmpeg ile H.265 streaming
        self.start_ffmpeg_stream()
    
    def start_ffmpeg_stream(self):
        """FFmpeg ile H.265 UDP stream"""
        try:
            # FFmpeg komutu - H.265 hardware encoding
            cmd = [
                'ffmpeg', '-y',
                '-f', 'v4l2',
                '-framerate', str(self.fps),
                '-video_size', '1280x720',
                '-i', '/dev/video0',
                '-c:v', 'libx265',  # H.265 codec
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-crf', str(self.quality),
                '-maxrate', '2M',
                '-bufsize', '1M',
                '-g', str(self.fps),
                '-keyint_min', str(self.fps),
                '-pix_fmt', 'yuv420p',
                '-f', 'mpegts',
                '-muxdelay', '0.1',
                f'udp://{self.get_broadcast_ip()}:{self.port}?broadcast=1'
            ]
            
            print("FFmpeg başlatılıyor...")
            print(f"Komut: {' '.join(cmd)}")
            
            # FFmpeg sürecini başlat
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            
            print("H.265 UDP streaming başladı!")
            print(f"Broadcast IP: {self.get_broadcast_ip()}:{self.port}")
            
            # Sürecin çalışmasını bekle
            try:
                while self.streaming:
                    if process.poll() is not None:
                        print("FFmpeg süreci durdu!")
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Kullanıcı tarafından durduruldu...")
            finally:
                process.terminate()
                process.wait()
                
        except Exception as e:
            print(f"FFmpeg hatası: {e}")
            # Fallback: OpenCV ile basic streaming
            self.opencv_fallback_stream()
    
    def opencv_fallback_stream(self):
        """OpenCV ile fallback streaming (MJPEG)"""
        print("OpenCV fallback streaming başlatılıyor...")
        
        # UDP socket oluştur
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Kamera başlat
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print(f"OpenCV streaming başladı - {self.get_broadcast_ip()}:{self.port}")
        
        frame_count = 0
        try:
            while self.streaming:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Frame'i sıkıştır
                _, buffer = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 50])
                
                # Paketlere böl (UDP 65507 byte limit)
                data = buffer.tobytes()
                packet_size = 60000
                
                for i in range(0, len(data), packet_size):
                    packet = data[i:i + packet_size]
                    
                    # Header ekle: frame_id + packet_id + total_packets
                    total_packets = (len(data) + packet_size - 1) // packet_size
                    packet_id = i // packet_size
                    
                    header = struct.pack('!III', frame_count, packet_id, total_packets)
                    full_packet = header + packet
                    
                    try:
                        sock.sendto(full_packet, (self.get_broadcast_ip(), self.port))
                    except:
                        pass
                
                frame_count += 1
                time.sleep(1.0 / self.fps)
                
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            sock.close()
    
    def stop_streaming(self):
        """Streaming durdur"""
        self.streaming = False
        print("Streaming durduruldu!")

def main():
    print("=== Raspberry Pi Camera Streamer ===")
    print("H.265 UDP Streaming")
    
    # Parametreler
    port = 9999
    quality = 23  # CRF değeri (düşük = yüksek kalite)
    fps = 25
    
    if len(os.sys.argv) > 1:
        port = int(os.sys.argv[1])
    if len(os.sys.argv) > 2:
        quality = int(os.sys.argv[2])
    if len(os.sys.argv) > 3:
        fps = int(os.sys.argv[3])
    
    streamer = RaspberryStreamer(port=port, quality=quality, fps=fps)
    
    try:
        streamer.start_streaming()
    except KeyboardInterrupt:
        print("\nÇıkış yapılıyor...")
    finally:
        streamer.stop_streaming()

if __name__ == "__main__":
    main()
