import cv2
import numpy as np
import pyautogui
import time
import keyboard

# İzlenecek ekran bölgesi
region_left = 738
region_top = 207
region_width = 792
region_height = 774

# Gelişmiş renk aralıkları (HSV formatında)
lower_blue = np.array([90, 150, 50])    # Mavi tonları
upper_blue = np.array([130, 255, 255])
lower_red1 = np.array([0, 150, 50])     # Kırmızı tonları (0-10)
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 50])   # Kırmızı tonları (170-180)
upper_red2 = np.array([180, 255, 255])

# Algılama parametreleri
min_area = 500
max_area = (region_width * region_height) * 0.4
solidity_threshold = 0.85  # Katılık oranı (kareler için)

print("Başladı. Çıkmak için 'q' tuşuna basın.")

def detect_squares(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mavi ve kırmızı maskeleri
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(red_mask1, red_mask2))
    
    # Gelişmiş temizleme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Kontur analizi
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Kare şekil kontrolü
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        
        if len(approx) == 4:
            # Katılık kontrolü
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area)/hull_area
                if solidity > solidity_threshold:
                    # En/boy oranı kontrolü
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w)/h
                    if 0.8 <= aspect_ratio <= 1.2:
                        squares.append((x, y, w, h))
    
    return squares

while True:
    if keyboard.is_pressed("q"):
        print("Çıkılıyor...")
        break

    # Ekran görüntüsü al
    screenshot = pyautogui.screenshot(region=(region_left, region_top, region_width, region_height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Kareleri tespit et
    squares = detect_squares(frame)
    
    if squares:
        for x, y, w, h in squares:
            # Koordinatları ayarla
            screen_x1 = region_left + x
            screen_y1 = region_top + y
            screen_x2 = screen_x1 + w
            screen_y2 = screen_y1 + h
            
            print(f"Kare bulundu: ({screen_x1}, {screen_y1}) - ({screen_x2}, {screen_y2})")
            
            # İşaretleme yap
            pyautogui.moveTo(screen_x1, screen_y1, duration=0.2)
            pyautogui.mouseDown()
            pyautogui.moveTo(screen_x2, screen_y2, duration=0.3)
            pyautogui.mouseUp()
            
        print("Kare bulunamadı, sonraki resme geçiliyor...")
        pyautogui.press('right')
        time.sleep(0.5)