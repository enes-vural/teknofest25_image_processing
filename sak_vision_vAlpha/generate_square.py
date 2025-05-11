import cv2
import numpy as np
import os
import random

# Kayıt klasörü
output_folder = "output/squares"
os.makedirs(output_folder, exist_ok=True)

def generate_random_blue():
    b = random.randint(150, 255)
    g = random.randint(0, 70)
    r = random.randint(0, 70)
    return (b, g, r)

def generate_random_red():
    r = random.randint(150, 255)
    g = random.randint(0, 70)
    b = random.randint(0, 70)
    return (b, g, r)

def draw_random_square(image_size=256):
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    # Kare boyutu ve konum
    size = random.randint(40, 80)
    top_left_x = random.randint(0, image_size - size)
    top_left_y = random.randint(0, image_size - size)
    
    return img, (top_left_x, top_left_y, size)

# Üretim işlemi
for i in range(2000):
    if i < 500:  # İlk 500 mavi kare
        color = generate_random_blue()
        blur = False  # Net
    elif i < 1000:  # Sonraki 500 kırmızı kare
        color = generate_random_red()
        blur = False  # Net
    elif i < 1500:  # Sonraki 500 mavi blur kare
        color = generate_random_blue()
        blur = True   # Blur
    else:  # Sonraki 500 kırmızı blur kare
        color = generate_random_red()
        blur = True   # Blur

    # Kareyi çiz
    img, (x, y, size) = draw_random_square()
    cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)

    # Rastgele ışıklandırma
    brightness = random.uniform(0.6, 1.4)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Blur işlemi (500'den sonrası)
    if blur:
        ksize = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Dosyayı kaydet
    filename = f"{output_folder}/square_{i:04d}.png"
    cv2.imwrite(filename, img)

print("✅ 2.000 mavi ve kırmızı kare başarıyla üretildi.")
