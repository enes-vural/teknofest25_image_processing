from ultralytics import YOLO
import cv2
import numpy as np

# Modeli yükle
model = YOLO("runs/detect/train3/weights/best.pt")

# Kamera ayarları
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Arka planı blurla (false positive'leri azaltmak için)
    blurred_bg = cv2.GaussianBlur(frame, (25, 25), 0)
    
    # Model tahmini yap
    results = model.predict(blurred_bg, iou=0.5, conf=0.7, verbose=False)
    
    # Algılanan nesneleri işle
    for r in results:
        for box in r.boxes:
            # Koordinatlar ve sınıf bilgisi
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]
            
            # En-boy oranı hesapla
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height != 0 else 0
            
            # Şekil belirleme
            shape_type = "undefined"
            if 0.85 < aspect_ratio < 1.15:  # Kare için geniş tolerans
                shape_type = "SQUARE"
            elif 1.15 < aspect_ratio < 1.5:  # Altıgen için aralık
                shape_type = "HEXAGON"
            
            # Sadece belirli bir güven eşiğinin üstündekileri göster
            if conf > 0.7:
                # Çerçeve çiz
                color = (0, 255, 0) if shape_type == "SQUARE" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Bilgi yazısı
                info_text = f"{label} {shape_type} {conf:.2f}"
                cv2.putText(frame, info_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Debug bilgisi
                print(f"Detected: {label} | Shape: {shape_type} | AR: {aspect_ratio:.2f} | Conf: {conf:.2f}")
    
    # Görüntüyü göster
    cv2.imshow('Shape Detection', frame)
    
    # Çıkış için 'q' tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()