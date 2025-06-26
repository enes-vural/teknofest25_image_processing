import cv2
import numpy as np
from collections import deque
import time

class ShapeTracker:
    def __init__(self, max_disappeared=10):
        # Tracking state
        self.tracked_objects = {}  # ID -> object info
        self.next_id = 0
        self.max_disappeared = max_disappeared
        
        # Renk aralıkları HSV'de
        self.color_ranges = {
            'blue': [(100, 50, 50), (130, 255, 255)],
            'red': [(0, 50, 50), (10, 255, 255)],  # Kırmızının alt aralığı
            'red2': [(170, 50, 50), (180, 255, 255)]  # Kırmızının üst aralığı
        }
        
        # Hedef şekiller
        self.target_shapes = {
            'blue_hexagon': {'color': 'blue', 'shape': 'hexagon', 'name': 'Mavi Altıgen'},
            'red_square': {'color': 'red', 'shape': 'square', 'name': 'Kırmızı Kare'},
            'blue_square': {'color': 'blue', 'shape': 'square', 'name': 'Mavi Kare'},
            'red_triangle': {'color': 'red', 'shape': 'triangle', 'name': 'Kırmızı Üçgen'}
        }
        
        # Kalman filter'lar her obje için
        self.kalman_filters = {}
        
    def create_kalman_filter(self, initial_pos):
        """Yeni obje için Kalman filter oluştur"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 1.0 * np.eye(2, dtype=np.float32)
        
        # Initial state
        kalman.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        kalman.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        
        return kalman
    
    def extract_color_mask(self, frame, color_name):
        """Belirli renk için mask oluştur"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if color_name == 'red':
            # Kırmızı için iki aralık birleştir
            lower1, upper1 = self.color_ranges['red']
            lower2, upper2 = self.color_ranges['red2']
            
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = self.color_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def classify_shape(self, contour):
        """Contour'un şeklini sınıflandır"""
        # Contour'u basitleştir
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        
        # Alan ve perimeter kontrolü
        area = cv2.contourArea(contour)
        if area < 1000:  # Çok küçük objeler
            return None
        
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (daire benzeri) hesapla
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Şekil sınıflandırması
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            # Kare mi dikdörtgen mi kontrol et
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.85 <= aspect_ratio <= 1.15:  # Kareye yakın
                return 'square'
            else:
                return 'rectangle'
        elif vertices >= 5 and vertices <= 8:
            # Altıgen kontrolü - circularity ile
            if circularity > 0.6:  # Yeterince dairesel
                if vertices == 6:
                    return 'hexagon'
                elif vertices >= 7:  # Çok köşeli -> muhtemelen daire
                    return 'circle'
        elif vertices > 8 and circularity > 0.7:
            return 'circle'
        
        return 'unknown'
    
    def detect_shapes(self, frame):
        """Frame'deki hedef şekilleri tespit et"""
        detected_objects = []
        
        for shape_key, shape_info in self.target_shapes.items():
            color = shape_info['color']
            target_shape = shape_info['shape']
            name = shape_info['name']
            
            # Renk maskesi
            mask = self.extract_color_mask(frame, color)
            
            # Contour bulma
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Şekil sınıflandırması
                detected_shape = self.classify_shape(contour)
                
                if detected_shape == target_shape:
                    # Merkez nokta hesapla
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        detected_objects.append({
                            'type': shape_key,
                            'name': name,
                            'center': (cx, cy),
                            'contour': contour,
                            'bbox': cv2.boundingRect(contour),
                            'area': cv2.contourArea(contour),
                            'color': color,
                            'shape': target_shape
                        })
        
        return detected_objects
    
    def match_objects(self, detected_objects):
        """Yeni tespit edilen objeler ile mevcut tracker'ları eşleştir"""
        if not self.tracked_objects:
            # İlk frame - tüm objeleri yeni olarak ekle
            for obj in detected_objects:
                obj_id = self.next_id
                self.next_id += 1
                
                obj['id'] = obj_id
                obj['disappeared'] = 0
                obj['trajectory'] = deque(maxlen=30)
                obj['trajectory'].append(obj['center'])
                
                # Kalman filter oluştur
                self.kalman_filters[obj_id] = self.create_kalman_filter(obj['center'])
                
                self.tracked_objects[obj_id] = obj
            
            return
        
        # Mevcut objeler için prediction yap
        predictions = {}
        for obj_id, kalman in self.kalman_filters.items():
            if obj_id in self.tracked_objects:
                pred = kalman.predict()
                predictions[obj_id] = (int(pred[0]), int(pred[1]))
        
        # Distance matrix hesapla
        tracked_ids = list(self.tracked_objects.keys())
        
        if not detected_objects:
            # Hiç obje bulunamadı - tüm objeler disappeared
            for obj_id in tracked_ids:
                self.tracked_objects[obj_id]['disappeared'] += 1
            return
        
        # Hungarian algorithm yerine basit greedy matching
        used_detections = set()
        
        for obj_id in tracked_ids:
            if obj_id not in predictions:
                continue
                
            pred_pos = predictions[obj_id]
            tracked_obj = self.tracked_objects[obj_id]
            
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, detected_obj in enumerate(detected_objects):
                if i in used_detections:
                    continue
                
                # Aynı tip obje mi?
                if detected_obj['type'] != tracked_obj['type']:
                    continue
                
                # Mesafe hesapla
                det_pos = detected_obj['center']
                distance = np.sqrt((pred_pos[0] - det_pos[0])**2 + 
                                 (pred_pos[1] - det_pos[1])**2)
                
                if distance < best_distance and distance < 100:  # Max distance threshold
                    best_distance = distance
                    best_match = detected_obj
                    best_idx = i
            
            if best_match:
                # Match bulundu - güncelle
                used_detections.add(best_idx)
                
                # Kalman güncelle
                measurement = np.array([[np.float32(best_match['center'][0])], 
                                      [np.float32(best_match['center'][1])]])
                self.kalman_filters[obj_id].correct(measurement)
                
                # Obje bilgilerini güncelle
                self.tracked_objects[obj_id].update(best_match)
                self.tracked_objects[obj_id]['disappeared'] = 0
                self.tracked_objects[obj_id]['trajectory'].append(best_match['center'])
                
            else:
                # Match bulunamadı
                self.tracked_objects[obj_id]['disappeared'] += 1
        
        # Yeni objeler (match edilmemiş detections)
        for i, detected_obj in enumerate(detected_objects):
            if i not in used_detections:
                obj_id = self.next_id
                self.next_id += 1
                
                detected_obj['id'] = obj_id
                detected_obj['disappeared'] = 0
                detected_obj['trajectory'] = deque(maxlen=30)
                detected_obj['trajectory'].append(detected_obj['center'])
                
                # Kalman filter oluştur
                self.kalman_filters[obj_id] = self.create_kalman_filter(detected_obj['center'])
                
                self.tracked_objects[obj_id] = detected_obj
        
        # Kayıp objeler temizle
        to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if obj['disappeared'] > self.max_disappeared:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.kalman_filters:
                del self.kalman_filters[obj_id]
    
    def update(self, frame):
        """Ana güncelleme fonksiyonu"""
        # Şekilleri tespit et
        detected_objects = self.detect_shapes(frame)
        
        # Objeleri eşleştir ve tracker'ları güncelle
        self.match_objects(detected_objects)
        
        return self.tracked_objects
    
    def draw_results(self, frame, tracked_objects):
        """Sonuçları frame üzerine çiz"""
        colors = {
            'blue_hexagon': (255, 0, 0),    # Mavi
            'red_square': (0, 0, 255),      # Kırmızı
            'blue_square': (255, 100, 0),   # Açık Mavi
            'red_triangle': (0, 100, 255)   # Turuncu-Kırmızı
        }
        
        for obj_id, obj in tracked_objects.items():
            if obj['disappeared'] > 0:
                continue  # Kayıp objeler çizilmez
            
            center = obj['center']
            obj_type = obj['type']
            name = obj['name']
            color = colors.get(obj_type, (255, 255, 255))
            
            # Contour çiz
            if 'contour' in obj and obj['contour'] is not None:
                cv2.drawContours(frame, [obj['contour']], -1, color, 3)
            
            # Merkez nokta
            cv2.circle(frame, center, 8, color, -1)
            cv2.circle(frame, center, 15, color, 2)
            
            # Bounding box
            if 'bbox' in obj:
                x, y, w, h = obj['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Label
            label = f"{name} (ID:{obj_id})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (center[0]-label_size[0]//2-5, center[1]-30), 
                         (center[0]+label_size[0]//2+5, center[1]-10), color, -1)
            
            # Label text
            cv2.putText(frame, label, (center[0]-label_size[0]//2, center[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Trajectory
            trajectory = list(obj['trajectory'])
            for i in range(1, len(trajectory)):
                alpha = i / len(trajectory)
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, trajectory[i-1], trajectory[i], line_color, 2)
        
        return frame

# Ana uygulama
def main():
    cap = cv2.VideoCapture(0)
    
    # Kamera ayarları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracker = ShapeTracker(max_disappeared=50)
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    
    print("Hedef Şekiller:")
    print("- Mavi Altıgen")
    print("- Kırmızı Kare") 
    print("- Mavi Kare")
    print("- Kırmızı Üçgen")
    print("\nKontroller:")
    print("Q: Çıkış")
    print("R: Reset")
    print("S: Screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fps_counter += 1
        
        # Gaussian blur ile noise azalt
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Tracking güncelle
        tracked_objects = tracker.update(frame_blurred)
        
        # Sonuçları çiz
        result_frame = tracker.draw_results(frame.copy(), tracked_objects)
        
        # Status bilgileri
        active_objects = len([obj for obj in tracked_objects.values() if obj['disappeared'] == 0])
        cv2.putText(result_frame, f"Aktif Objeler: {active_objects}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tespit edilen şekiller listesi
        y_offset = 60
        shape_counts = {}
        for obj in tracked_objects.values():
            if obj['disappeared'] == 0:
                shape_name = obj['name']
                shape_counts[shape_name] = shape_counts.get(shape_name, 0) + 1
        
        for shape_name, count in shape_counts.items():
            cv2.putText(result_frame, f"{shape_name}: {count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # FPS
        if fps_counter % 30 == 0:
            elapsed = time.time() - start_time
            fps = fps_counter / elapsed
            print(f"FPS: {fps:.1f}, Toplam Obje: {len(tracked_objects)}")
        
        cv2.imshow('Drone Shape Tracker', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker = ShapeTracker(max_disappeared=15)
            print("Tracker sıfırlandı!")
        elif key == ord('s'):
            filename = f'shape_detection_{int(time.time())}.jpg'
            cv2.imwrite(filename, result_frame)
            print(f"Screenshot kaydedildi: {filename}")
        elif key == ord('c'):
            # Renk kalibrasyonu için debug
            print("Renk aralıkları debug modunda...")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Mouse callback için debug
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    hsv_value = hsv[y, x]
                    bgr_value = frame[y, x]
                    print(f"Pos: ({x},{y}), HSV: {hsv_value}, BGR: {bgr_value}")
            
            cv2.setMouseCallback('Drone Shape Tracker', mouse_callback)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()