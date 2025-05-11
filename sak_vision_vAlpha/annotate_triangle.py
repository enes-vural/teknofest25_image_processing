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

# Kırmızı renk aralıkları (HSV formatında)
lower_red1 = np.array([0, 150, 50])     # Açık kırmızı tonları
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 150, 50])   # Koyu kırmızı tonları
upper_red2 = np.array([180, 255, 255])

# Algılama parametreleri
min_area = 300
max_area = (region_width * region_height) * 0.3
triangle_threshold = 0.85  # Üçgen benzerlik eşiği

print("Başladı. Çıkmak için 'q' tuşuna basın.")

def detect_triangles(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Kırmızı maskeleri
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    combined_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Gürültü azaltma
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Kontur analizi
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Üçgen şekil kontrolü
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        
        if len(approx) == 3:
            # Eşkenar üçgen kontrolü
            sides = [
                np.linalg.norm(approx[1][0] - approx[0][0]),
                np.linalg.norm(approx[2][0] - approx[1][0]),
                np.linalg.norm(approx[0][0] - approx[2][0])
            ]
            
            # Kenar uzunluklarının benzerliği
            max_side = max(sides)
            similarity = min(sides)/max_side
            
            if similarity > triangle_threshold:
                x, y, w, h = cv2.boundingRect(approx)
                triangles.append((x, y, w, h))
    
    return triangles

while True:
    if keyboard.is_pressed("q"):
        print("Çıkılıyor...")
        break

    # Ekran görüntüsü al
    screenshot = pyautogui.screenshot(region=(region_left, region_top, region_width, region_height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Üçgenleri tespit et
    triangles = detect_triangles(frame)
    
    if triangles:
        for x, y, w, h in triangles:
            # Koordinatları ayarla
            screen_x1 = region_left + x
            screen_y1 = region_top + y
            screen_x2 = screen_x1 + w
            screen_y2 = screen_y1 + h
            
            print(f"Üçgen bulundu: ({screen_x1}, {screen_y1}) - ({screen_x2}, {screen_y2})")
            
            # İşaretleme yap
            pyautogui.moveTo(screen_x1, screen_y1, duration=0.2)
            pyautogui.mouseDown()
            pyautogui.moveTo(screen_x2, screen_y2, duration=0.3)
            pyautogui.mouseUp()
            
        # Onayla ve sonraki resme geç
        pyautogui.press('enter')
        time.sleep(0.3)
        pyautogui.press('right')
        time.sleep(0.5)
    else:
        print("Üçgen bulunamadı, sonraki resme geçiliyor...")
        pyautogui.press('right')
        time.sleep(0.5)