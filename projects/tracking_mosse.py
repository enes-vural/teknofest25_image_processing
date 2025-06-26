#!/usr/bin/env python3
"""
Drone MOSSE Tracker - Şekil Algılama Sistemi
Raspberry Pi 4 için optimize edilmiş
Kırmızı üçgen/kare ve mavi altıgen/kare takibi
OpenCV sürüm uyumlu
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
import math

# OpenCV tracker kontrolü
def get_available_tracker():
    """Mevcut tracker'ı kontrol et"""
    trackers = []
    
    # MOSSE tracker kontrolü
    try:
        if hasattr(cv2, 'TrackerMOSSE_create'):
            trackers.append(('MOSSE', cv2.TrackerMOSSE_create))
        elif hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            trackers.append(('MOSSE', cv2.legacy.TrackerMOSSE_create))
    except:
        pass
    
    # Alternatif tracker'lar
    try:
        if hasattr(cv2, 'TrackerCSRT_create'):
            trackers.append(('CSRT', cv2.TrackerCSRT_create))
    except:
        pass
    
    try:
        if hasattr(cv2, 'TrackerKCF_create'):
            trackers.append(('KCF', cv2.TrackerKCF_create))
    except:
        pass
    
    return trackers[0] if trackers else None

class DroneShapeTracker:
    def __init__(self, camera_id=0, frame_width=640, frame_height=480):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Mevcut tracker'ı kontrol et
        self.tracker_info = get_available_tracker()
        if self.tracker_info:
            print(f"Kullanılan tracker: {self.tracker_info[0]}")
        else:
            raise Exception("Uygun tracker bulunamadı!")
        
        # Kamera başlatma
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Tracker'lar
        self.red_tracker = None
        self.blue_tracker = None
        self.tracking_red = False
        self.tracking_blue = False
        
        # Renk aralıkları (HSV)
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([100, 150, 0])
        self.blue_upper = np.array([130, 255, 255])
        
        # Performans metrikleri
        self.fps_counter = deque(maxlen=30)
        self.detection_history = deque(maxlen=10)
        
        # Kontrol parametreleri
        self.min_area = 500
        self.max_area = 50000
        self.running = False
        
    def detect_shapes(self, mask, color_name):
        """Şekil algılama fonksiyonu"""
        # Morfolojik işlemler
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Kontur bulma
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Kontur yaklaşımı
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Şekil sınıflandırma
                shape_type = self.classify_shape(approx, area, w, h)
                
                if shape_type:
                    detected_objects.append({
                        'shape': shape_type,
                        'color': color_name,
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'area': area,
                        'center': (x + w//2, y + h//2)
                    })
        
        return detected_objects
    
    def classify_shape(self, approx, area, width, height):
        """Şekil sınıflandırma"""
        vertices = len(approx)
        aspect_ratio = width / height if height > 0 else 0
        
        # Üçgen algılama
        if vertices == 3:
            return "triangle"
        
        # Kare/Dikdörtgen algılama
        elif vertices == 4:
            if 0.8 <= aspect_ratio <= 1.2:
                return "square"
            else:
                return "rectangle"
        
        # Altıgen algılama
        elif vertices == 6:
            return "hexagon"
        
        # Çember/Elips algılama (yedek)
        elif vertices > 8:
            circularity = 4 * math.pi * area / (cv2.arcLength(approx, True) ** 2)
            if circularity > 0.7:
                return "circle"
        
        return None
    
    def init_tracker(self, frame, bbox, tracker_type):
        """Tracker başlatma"""
        try:
            # Mevcut tracker'ı kullan
            tracker = self.tracker_info[1]()
            success = tracker.init(frame, bbox)
            return tracker if success else None
        except Exception as e:
            print(f"Tracker başlatma hatası: {e}")
            return None
    
    def update_tracking(self, frame):
        """Takip güncelleme"""
        results = {}
        
        # Kırmızı nesne takibi
        if self.tracking_red and self.red_tracker:
            success, bbox = self.red_tracker.update(frame)
            if success:
                results['red'] = {
                    'bbox': bbox,
                    'center': (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                }
            else:
                self.tracking_red = False
                self.red_tracker = None
        
        # Mavi nesne takibi
        if self.tracking_blue and self.blue_tracker:
            success, bbox = self.blue_tracker.update(frame)
            if success:
                results['blue'] = {
                    'bbox': bbox,
                    'center': (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                }
            else:
                self.tracking_blue = False
                self.blue_tracker = None
        
        return results
    
    def draw_results(self, frame, detections, tracking_results):
        """Sonuçları çizme"""
        result_frame = frame.copy()
        
        # Algılanan şekilleri çiz
        for detection in detections:
            x, y, w, h = detection['bbox']
            color = detection['color']
            shape = detection['shape']
            
            # Renk seçimi
            if color == 'red':
                draw_color = (0, 0, 255)
            else:  # blue
                draw_color = (255, 0, 0)
            
            # Bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), draw_color, 2)
            
            # Merkez nokta
            center = detection['center']
            cv2.circle(result_frame, center, 5, draw_color, -1)
            
            # Etiket
            label = f"{color} {shape}"
            cv2.putText(result_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
        
        # Takip sonuçlarını çiz
        for color, result in tracking_results.items():
            bbox = result['bbox']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Takip rengi
            track_color = (0, 255, 0) if color == 'red' else (255, 255, 0)
            
            # Takip kutusu
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), track_color, 3)
            
            # Takip etiketi
            cv2.putText(result_frame, f"Tracking {color}", (x, y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
        
        return result_frame
    
    def process_frame(self, frame):
        """Ana işleme fonksiyonu"""
        start_time = time.time()
        
        # HSV dönüşümü
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Renk maskeleri
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        # Şekil algılama
        all_detections = []
        red_detections = self.detect_shapes(red_mask, 'red')
        blue_detections = self.detect_shapes(blue_mask, 'blue')
        
        all_detections.extend(red_detections)
        all_detections.extend(blue_detections)
        
        # Yeni hedefleri takip için başlat
        for detection in all_detections:
            color = detection['color']
            shape = detection['shape']
            bbox = detection['bbox']
            
            # Kırmızı şekiller için (üçgen/kare)
            if color == 'red' and shape in ['triangle', 'square'] and not self.tracking_red:
                self.red_tracker = self.init_tracker(frame, bbox, 'red')
                if self.red_tracker:
                    self.tracking_red = True
                    print(f"Kırmızı {shape} takip başlatıldı")
            
            # Mavi şekiller için (altıgen/kare)
            elif color == 'blue' and shape in ['hexagon', 'square'] and not self.tracking_blue:
                self.blue_tracker = self.init_tracker(frame, bbox, 'blue')
                if self.blue_tracker:
                    self.tracking_blue = True
                    print(f"Mavi {shape} takip başlatıldı")
        
        # Takip güncelleme
        tracking_results = self.update_tracking(frame)
        
        # Sonuçları çiz
        result_frame = self.draw_results(frame, all_detections, tracking_results)
        
        # FPS hesaplama
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        self.fps_counter.append(fps)
        avg_fps = sum(self.fps_counter) / len(self.fps_counter)
        
        # Bilgi metinleri
        cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = f"Takip: Kırmızı={self.tracking_red}, Mavi={self.tracking_blue}"
        cv2.putText(result_frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame, all_detections, tracking_results
    
    def run(self):
        """Ana çalışma döngüsü"""
        print("Drone Shape Tracker başlatılıyor...")
        print("Kırmızı üçgen/kare ve mavi altıgen/kare aranıyor...")
        print("Çıkış için 'q' tuşuna basın")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Kamera okuma hatası!")
                    break
                
                # Frame işleme
                result_frame, detections, tracking_results = self.process_frame(frame)
                
                # Sonucu göster
                cv2.imshow('Drone MOSSE Tracker', result_frame)
                
                # Klavye kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Takip sıfırlama
                    self.tracking_red = False
                    self.tracking_blue = False
                    self.red_tracker = None
                    self.blue_tracker = None
                    print("Takip sıfırlandı")
                elif key == ord('s'):
                    # Ekran görüntüsü kaydet
                    filename = f"drone_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"Görüntü kaydedildi: {filename}")
                
        except KeyboardInterrupt:
            print("\nProgram durduruldu")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Kaynakları temizle"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Temizlik tamamlandı")

def main():
    """Ana fonksiyon"""
    # Raspberry Pi kamera ayarları
    tracker = DroneShapeTracker(
        camera_id=0,  # USB kamera için 0, Pi kamera için değiştirin
        frame_width=640,
        frame_height=480
    )
    
    try:
        tracker.run()
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    main()