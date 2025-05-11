import cv2
import numpy as np
import os
import random
import math

# Kayıt klasörü
output_folder = "output/red_triangles"
os.makedirs(output_folder, exist_ok=True)

def generate_random_red():
    r = random.randint(150, 255)
    g = random.randint(0, 70)
    b = random.randint(0, 70)
    return (b, g, r)

def draw_equilateral_triangle(image_size=256):
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    color = generate_random_red()

    # Rastgele merkez noktası ve boyut
    center_x = random.randint(60, image_size - 60)
    center_y = random.randint(60, image_size - 60)
    size = random.randint(40, 80)

    # Üçgenin 3 köşesini hesapla (eşkenar üçgen)
    angle_offset = random.uniform(0, 2 * math.pi)  # dönüş açısı
    points = []
    for i in range(3):
        angle = angle_offset + i * 2 * math.pi / 3
        x = int(center_x + size * math.cos(angle))
        y = int(center_y + size * math.sin(angle))
        points.append([x, y])
    pts = np.array([points], dtype=np.int32)

    cv2.drawContours(img, pts, 0, color, -1)
    return img

for i in range(2500):
    img = draw_equilateral_triangle()

    # Rastgele ışıklandırma
    brightness = random.uniform(0.6, 1.4)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Blur (1000'den sonra)
    if i >= 1000:
        ksize = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    filename = f"{output_folder}/triangle_{i:04d}.png"
    cv2.imwrite(filename, img)

print("✅ Düzgün 2500 kırmızı üçgen başarıyla üretildi.")
