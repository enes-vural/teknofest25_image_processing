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

# Genişletilmiş renk aralıkları (mavi, mor ve turkuaz)
lower_blue1 = np.array([90, 50, 50])   # Koyu mavi
upper_blue1 = np.array([120, 255, 255]) # Açık mavi
lower_blue2 = np.array([120, 50, 50])  # Mor
upper_blue2 = np.array([150, 255, 255]) # Açık mor
lower_turquoise = np.array([80, 50, 50]) # Turkuaz
upper_turquoise = np.array([100, 255, 255]) # Açık turkuaz

# Maksimum kabul edilebilir altıgen alanı (ekran alanının %40'ı)
max_area = (region_width * region_height) * 0.4
# Minimum kabul edilebilir altıgen alanı
min_area = 500

print("Başladı. Çıkmak için 'q' tuşuna basın.")
while True:
    if keyboard.is_pressed("q"):
        print("Çıkılıyor...")
        break

    # Ekran görüntüsü al
    screenshot = pyautogui.screenshot(region=(region_left, region_top, region_width, region_height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tüm renk aralıkları için maske oluştur
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask3 = cv2.inRange(hsv, lower_turquoise, upper_turquoise)
    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    # Gürültü azaltma
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Kontur bulma
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Alan kontrolü
        if area < min_area or area > max_area:
            continue
            
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        sides = len(approx)
        
        # 5-7 kenarlı şekilleri kabul et (esnek altıgen)
        if 5 <= sides <= 7:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Koordinatları tam ekran için ayarla
            screen_x1 = region_left + x
            screen_y1 = region_top + y
            screen_x2 = screen_x1 + w
            screen_y2 = screen_y1 + h

            print(f"Altıgen bulundu: ({screen_x1}, {screen_y1}) - ({screen_x2}, {screen_y2}) | Alan: {area}")

            # Mouse ile dikdörtgen çiz
            pyautogui.moveTo(screen_x1, screen_y1, duration=0.2)
            pyautogui.mouseDown()
            pyautogui.moveTo(screen_x2, screen_y2, duration=0.3)
            pyautogui.mouseUp()

            # Onayla ve sonraki resme geç
            pyautogui.press('enter')
            time.sleep(0.3)
            pyautogui.press('right')
            time.sleep(2) # Yeni resim için bekle
            
            found = True
            break

    if not found:
        print("Uygun altıgen bulunamadı, sonraki resme geçiliyor...")
        pyautogui.press('right')
        time.sleep(2)