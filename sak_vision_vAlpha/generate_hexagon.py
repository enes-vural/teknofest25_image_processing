import cv2
import numpy as np
import os
import random

def random_blue_color():
    # Mavi tonları üret (BGR)
    return (random.randint(100, 255), random.randint(0, 100), random.randint(0, 100))

def adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def draw_random_hexagon(img_size=512):
    # Arka plan beyaz
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Rastgele merkez ve boyut
    center = (
        random.randint(150, img_size - 150),
        random.randint(150, img_size - 150)
    )
    radius = random.randint(40, 100)
    angle_offset = random.uniform(0, 2 * np.pi)

    # Altıgenin köşe noktaları
    points = np.array([
        (
            int(center[0] + radius * np.cos(2 * np.pi * i / 6 + angle_offset)),
            int(center[1] + radius * np.sin(2 * np.pi * i / 6 + angle_offset))
        )
        for i in range(6)
    ], np.int32)

    # Rastgele mavi ton
    color = random_blue_color()
    
    # Altıgeni doldur
    cv2.fillPoly(img, [points], color)
    
    return img

# Kayıt dizini
output_dir = "output/hexagon"
os.makedirs(output_dir, exist_ok=True)

total_images = 2500

for i in range(total_images):
    img = draw_random_hexagon()
    
    # Rastgele ışıklandırma (parlaklık): 0.7 ile 1.3 arası
    brightness_factor = random.uniform(0.7, 1.3)
    img = adjust_brightness(img, brightness_factor)

    # İlk 1000 net, kalanlar bulanık
    if i >= 1000:
        img = cv2.GaussianBlur(img, (7, 7), 0)

    # Görseli kaydet
    filename = f"{output_dir}/hexagon_{i:04d}.png"
    cv2.imwrite(filename, img)

print("Altıgen görselleri başarıyla üretildi.")
