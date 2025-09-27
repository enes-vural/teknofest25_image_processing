# ENGLISH

# Fırtına UAV - TEKNOFEST International Unmanned Aerial Vehicles Competition Rotorcraft Category Computer Vision Project
## Technical Documentation and User Manual

### 📋 Project Overview

This project is the "Fırtına UAV" software repository developed for **Teknofest 2025 competition**. It is an advanced Python/OpenCV application optimized for **real-time colored shape detection** on relatively low-power hardware such as Raspberry Pi.

### 🎯 Key Features
- **Real-time shape detection:** Detection of red and blue triangles, squares, hexagons
- **Hardware optimization:** Special performance settings for Raspberry Pi
- **Asynchronous video processing:** Thread-based camera management
- **Dynamic performance adjustment:** FPS tracking and automatic frame skipping
- **Reliable system:** Error-tolerant design

---

## 🛠 System Requirements

### Hardware Requirements
- **Main Platform:** Raspberry Pi 3B+ or higher (recommended: Raspberry Pi 4)
- **RAM:** Minimum 2GB (4GB recommended)
- **Camera:** USB camera or Raspberry Pi camera module
- **Storage:** Minimum 8GB SD card (Class 10)
- **Power:** 5V/3A power supply

### Software Requirements
- **Operating System:** Raspberry Pi OS (Bullseye or later)
- **Python Version:** Python 3.7+
- **OpenCV:** 4.5.0+

### Core Dependencies
```bash
# Main libraries
opencv-python>=4.5.0
numpy>=1.19.0
imutils>=0.5.4

# Hardware optimization libraries
psutil>=5.8.0          # CPU and memory management
threading              # Asynchronous operations (Python built-in)
queue                  # Buffer management (Python built-in)
```

## 🚀 Installation and Running

### 1. System Preparation (Raspberry Pi)
```bash
# System update
sudo apt update && sudo apt upgrade -y

# Required system packages
sudo apt install python3-pip python3-venv git -y
sudo apt install libatlas-base-dev libhdf5-dev libhdf5-serial-dev -y
sudo apt install libharfbuzz0b libwebp6 libtiff5 libjasper1 libilmbase25 -y
sudo apt install libopenexr25 libgstreamer1.0-0 libavcodec58 libavformat58 libswscale5 -y
```

### 2. Project Download and Installation
```bash
# Clone the project
git clone https://github.com/enes-vural/teknofest25_image_processing.git
cd teknofest25_image_processing

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Camera Settings
```bash
# For Raspberry Pi camera module
sudo raspi-config
# Interface Options > Camera > Enable

# USB camera test
v4l2-ctl --list-devices
```

### 4. Running the Application
```bash
# Basic execution
python3 src/main.py

# Performance mode execution
sudo python3 src/main.py --performance-mode

# Debug mode
python3 src/main.py --debug --show-fps
```

---

## ⚙️ Technical Details

### 1. Hardware and Performance Optimization

#### CPU Optimizations
- **NEON SIMD** activation (fast vector operations on ARM processors)
- **CPU performance mode** temporary activation
- **Process priority (nice)** adjustment
- **CPU core assignment** (CPU affinity)

```python
# Example optimization code
import os
import psutil

def optimize_cpu():
    # CPU performance mode
    os.system('echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
    
    # Process priority
    proc = psutil.Process()
    proc.nice(-10)  # High priority
    
    # CPU core assignment
    proc.cpu_affinity([2, 3])  # Use cores 2 and 3
```

#### Memory Optimizations
- **HDMI output deactivation** (if not in use)
- **GPU memory sharing** optimization
- **Buffer size** dynamic adjustment

### 2. Asynchronous Video Capture (AsyncVideoCapture)

#### Features
- **Thread-based video capture:** Camera operations independent of main program
- **Buffer system:** Frame holding with small queue (typical: 2-3 frames)
- **Automatic frame dropping:** Drop old frames when buffer is full
- **Low resolution:** 320x240 pixels (for performance)
- **Fixed FPS:** 30 FPS target

```python
class AsyncVideoCapture:
    def __init__(self, src=0, buffer_size=2):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.q = queue.Queue(maxsize=buffer_size)
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Drop old frame
                except queue.Empty:
                    pass
                    
            self.q.put(frame)
```

### 3. Shape Detection System (ShapeDetector)

#### Color-Based Masking
- **HSV color space** usage (more stable color detection)
- **Separate masks** for red and blue
- **Color ranges** read from configuration

```python
# HSV color ranges example
color_ranges = {
    "red": {
        "lower1": [0, 120, 120],    # Lower red range
        "upper1": [10, 255, 255],
        "lower2": [160, 120, 120],  # Upper red range
        "upper2": [179, 255, 255]
    },
    "blue": {
        "lower": [100, 120, 120],
        "upper": [130, 255, 255]
    }
}
```

#### Morphological Operations
- **Dilate operation:** Closing gaps in masks
- **Median blur:** Fast noise reduction
- **Small contour filtering:** Area-based filtering

#### Shape Recognition Algorithm
- **Contour approximation:** Douglas-Peucker algorithm
- **Vertex count:** 3=triangle, 4=square, 6=hexagon
- **Area and ratio checks:** Minimum size and geometric ratios
- **Real-time labeling:** Shape type and color display

```python
def detect_shape(self, contour):
    # Contour approximation
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Shape determination by vertex count
    vertices = len(approx)
    
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        # Square vs rectangle check
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif vertices == 6:
        return "hexagon"
    else:
        return "circle"
```

### 4. Real-Time FPS Calculation and Dynamic Adjustment

#### FPS Tracking
- **Rolling average:** Average of last 10 frames
- **Frame timing:** Processing time measurement per frame
- **Performance metrics:** CPU usage, memory status

#### Dynamic Frame Skipping
```python
class PerformanceManager:
    def __init__(self, target_fps=15):
        self.target_fps = target_fps
        self.frame_times = []
        self.skip_counter = 0
        
    def should_process_frame(self):
        avg_fps = self.get_average_fps()
        
        if avg_fps < self.target_fps * 0.8:  # If below 80%
            self.skip_counter += 1
            return self.skip_counter % 2 == 0  # Process every other frame
        else:
            self.skip_counter = 0
            return True  # Process every frame
```

---

## 🔧 Configuration

### Camera Settings (camera_config.json)
```json
{
    "resolution": {
        "width": 320,
        "height": 240
    },
    "fps": 30,
    "buffer_size": 2,
    "auto_exposure": true,
    "brightness": 50,
    "contrast": 50
}
```

### Color Detection Ranges (color_ranges.json)
```json
{
    "colors": {
        "red": {
            "hsv_lower1": [0, 120, 120],
            "hsv_upper1": [10, 255, 255],
            "hsv_lower2": [160, 120, 120],
            "hsv_upper2": [179, 255, 255]
        },
        "blue": {
            "hsv_lower": [100, 120, 120],
            "hsv_upper": [130, 255, 255]
        }
    }
}
```

### Performance Settings (performance_config.json)
```json
{
    "cpu_optimization": {
        "enable_performance_mode": true,
        "nice_value": -10,
        "cpu_affinity": [2, 3]
    },
    "memory_optimization": {
        "disable_hdmi": true,
        "gpu_memory_split": 64
    },
    "processing": {
        "target_fps": 15,
        "min_contour_area": 500,
        "max_contour_area": 50000,
        "frame_skip_threshold": 0.8
    }
}
```

---

## 📊 Performance Metrics

### Typical Performance Values (Raspberry Pi 4)
- **Processing Speed:** 12-18 FPS (at 320x240 resolution)
- **CPU Usage:** 40-60%
- **RAM Usage:** ~150-200MB
- **Detection Latency:** 50-80ms
- **Power Consumption:** ~8-12W (with optimization)

### Performance Improvement Tips
1. **Resolution reduction:** 320x240 → 240x180
2. **FPS limiting:** 30 FPS → 20 FPS
3. **Buffer size:** 2 → 1 (less latency)
4. **Color masking:** Single color detection
5. **GPU overclock:** Raspberry Pi configuration

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Camera Not Recognized
```bash
# Problem: Camera not found
# Solution:
sudo modprobe bcm2835-v4l2  # For Pi camera
v4l2-ctl --list-devices      # List available cameras
```

#### 2. Low FPS Performance
```python
# Problem: Very low FPS
# Solution: Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
```

#### 3. High CPU Usage
```bash
# Problem: 100% CPU usage
# Solution: Enable frame skipping
python3 src/main.py --frame-skip 2
```

#### 4. Memory Shortage
```bash
# Problem: Memory exhaustion
# Solution: GPU memory allocation
sudo raspi-config
# Advanced Options > Memory Split > 64MB
```

### Debug Commands
```bash
# System status check
htop
iostat 1
vcgencmd measure_temp

# Camera status
vcgencmd get_camera
v4l2-ctl --all

# OpenCV optimization check
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

---

## 📞 Support and Contact

### Project Owner
- **Developer:** Enes Vural
- **GitHub:** [@enes-vural](https://github.com/enes-vural)
- **Project Repository:** [teknofest25_image_processing](https://github.com/enes-vural/teknofest25_image_processing)

**Last Update:** September 2025  
**Version:** 1.0.0





# TURKISH


# Fırtına İHA - Teknofest 2025 Görüntü İşleme Projesi
## Teknik Dokümantasyon ve Kullanıcı El Kitabı

### 📋 Proje Genel Bakış

Bu proje, **Teknofest 2025 yarışması** kapsamında geliştirilen "Fırtına İHA" yazılım reposudur. Raspberry Pi gibi görece düşük güçlü donanımlarda **gerçek zamanlı renkli şekil tespiti** yapmak için optimize edilmiş, ileri seviye bir Python/OpenCV uygulamasıdır.

### 🎯 Temel Özellikler
- **Gerçek zamanlı şekil tespiti:** Kırmızı ve mavi renkli üçgen, kare, altıgen tespit
- **Donanım optimizasyonu:** Raspberry Pi için özel performans ayarları
- **Asenkron video işleme:** Thread tabanlı kamera yönetimi
- **Dinamik performans ayarı:** FPS takibi ve otomatik frame atlama
- **Güvenilir sistem:** Hata toleranslı tasarım

---

## 🛠 Sistem Gereksinimleri

### Donanım Gereksinimleri
- **Ana Platform:** Raspberry Pi 3B+ veya üzeri (önerilir: Raspberry Pi 4)
- **RAM:** Minimum 2GB (4GB önerilir)
- **Kamera:** USB kamera veya Raspberry Pi kamera modülü
- **Depolama:** Minimum 8GB SD kart (Class 10)
- **Güç:** 5V/3A güç kaynağı

### Yazılım Gereksinimleri
- **İşletim Sistemi:** Raspberry Pi OS (Bullseye veya üzeri)
- **Python Sürümü:** Python 3.7+
- **OpenCV:** 4.5.0+

### Temel Bağımlılıklar
```bash
# Ana kütüphaneler
opencv-python>=4.5.0
numpy>=1.19.0
imutils>=0.5.4

# Donanım optimizasyon kütüphaneleri
psutil>=5.8.0          # CPU ve bellek yönetimi
threading              # Asenkron işlemler (Python built-in)
queue                  # Buffer yönetimi (Python built-in)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Sistem Hazırlığı (Raspberry Pi)
```bash
# Sistem güncellemesi
sudo apt update && sudo apt upgrade -y

# Gerekli sistem paketleri
sudo apt install python3-pip python3-venv git -y
sudo apt install libatlas-base-dev libhdf5-dev libhdf5-serial-dev -y
sudo apt install libharfbuzz0b libwebp6 libtiff5 libjasper1 libilmbase25 -y
sudo apt install libopenexr25 libgstreamer1.0-0 libavcodec58 libavformat58 libswscale5 -y
```

### 2. Proje İndirme ve Kurulum
```bash
# Projeyi klonla
git clone https://github.com/enes-vural/teknofest25_image_processing.git
cd teknofest25_image_processing

# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 3. Kamera Ayarları
```bash
# Raspberry Pi kamera modülü için
sudo raspi-config
# Interface Options > Camera > Enable seçeneğini aktifleştir

# USB kamera test
v4l2-ctl --list-devices
```

### 4. Uygulamayı Çalıştırma
```bash
# Temel çalıştırma
python3 src/main.py

# Performans modu ile çalıştırma
sudo python3 src/main.py --performance-mode

# Debug modu
python3 src/main.py --debug --show-fps
```

---

## ⚙️ Teknik Detaylar

### 1. Donanım ve Performans Optimizasyonu

#### CPU Optimizasyonları
- **NEON SIMD** aktifleştirmesi (ARM işlemcilerde hızlı vektör işlemleri)
- **CPU performans modu** geçici aktivasyonu
- **İşlem önceliği (nice)** ayarlaması
- **CPU çekirdeği ataması** (CPU affinity)

```python
# Örnek optimizasyon kodu
import os
import psutil

def optimize_cpu():
    # CPU performans modu
    os.system('echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
    
    # İşlem önceliği
    proc = psutil.Process()
    proc.nice(-10)  # Yüksek öncelik
    
    # CPU çekirdeği ataması
    proc.cpu_affinity([2, 3])  # 2. ve 3. çekirdekleri kullan
```

#### Bellek Optimizasyonları
- **HDMI çıkışı deaktivasyonu** (kullanılmıyorsa)
- **GPU bellek paylaşımı** optimizasyonu
- **Buffer boyutu** dinamik ayarlama

### 2. Asenkron Video Yakalama (AsyncVideoCapture)

#### Özellikler
- **Thread-based video capture:** Ana programdan bağımsız kamera işlemi
- **Buffer sistemi:** Küçük kuyruk ile kare tutma (tipik: 2-3 kare)
- **Otomatik kare atma:** Buffer dolduğunda eski kareleri at
- **Düşük çözünürlük:** 320x240 piksel (performans için)
- **Sabit FPS:** 30 FPS hedefi

```python
class AsyncVideoCapture:
    def __init__(self, src=0, buffer_size=2):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.q = queue.Queue(maxsize=buffer_size)
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Eski kareyi at
                except queue.Empty:
                    pass
                    
            self.q.put(frame)
```

### 3. Şekil Tespit Sistemi (ShapeDetector)

#### Renk Tabanlı Maskeleme
- **HSV renk uzayı** kullanımı (daha kararlı renk tespiti)
- **Kırmızı ve mavi** için ayrı maskeler
- **Renk aralıkları** konfigürasyondan okunur

```python
# HSV renk aralıkları örneği
color_ranges = {
    "red": {
        "lower1": [0, 120, 120],    # Alt kırmızı aralığı
        "upper1": [10, 255, 255],
        "lower2": [160, 120, 120],  # Üst kırmızı aralığı
        "upper2": [179, 255, 255]
    },
    "blue": {
        "lower": [100, 120, 120],
        "upper": [130, 255, 255]
    }
}
```

#### Morfolojik İşlemler
- **Dilate işlemi:** Maskelerdeki boşlukları kapatma
- **Median blur:** Hızlı gürültü temizleme
- **Küçük kontur eleme:** Alan tabanlı filtreleme

#### Şekil Tanıma Algoritması
- **Kontur aproximasyonu:** Douglas-Peucker algoritması
- **Köşe sayısı:** 3=üçgen, 4=kare, 6=altıgen
- **Alan ve oran kontrolleri:** Minimum boyut ve geometrik oranlar
- **Gerçek zamanlı etiketleme:** Şekil tipı ve renk gösterimi

```python
def detect_shape(self, contour):
    # Kontur aproximasyonu
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Köşe sayısına göre şekil belirleme
    vertices = len(approx)
    
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        # Kare vs dikdörtgen kontrolü
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif vertices == 6:
        return "hexagon"
    else:
        return "circle"
```

### 4. Gerçek Zamanlı FPS Hesabı ve Dinamik Ayar

#### FPS Tracking
- **Rolling average:** Son 10 karenin ortalaması
- **Frame timing:** Her karede işleme süresi ölçümü
- **Performans metrikler:** CPU kullanımı, bellek durumu

#### Dinamik Frame Skipping
```python
class PerformanceManager:
    def __init__(self, target_fps=15):
        self.target_fps = target_fps
        self.frame_times = []
        self.skip_counter = 0
        
    def should_process_frame(self):
        avg_fps = self.get_average_fps()
        
        if avg_fps < self.target_fps * 0.8:  # %80'in altındaysa
            self.skip_counter += 1
            return self.skip_counter % 2 == 0  # Her ikinci kareyi işle
        else:
            self.skip_counter = 0
            return True  # Her kareyi işle
```

---

## 🔧 Konfigürasyon

### Kamera Ayarları (camera_config.json)
```json
{
    "resolution": {
        "width": 320,
        "height": 240
    },
    "fps": 30,
    "buffer_size": 2,
    "auto_exposure": true,
    "brightness": 50,
    "contrast": 50
}
```

### Renk Tespit Aralıkları (color_ranges.json)
```json
{
    "colors": {
        "red": {
            "hsv_lower1": [0, 120, 120],
            "hsv_upper1": [10, 255, 255],
            "hsv_lower2": [160, 120, 120],
            "hsv_upper2": [179, 255, 255]
        },
        "blue": {
            "hsv_lower": [100, 120, 120],
            "hsv_upper": [130, 255, 255]
        }
    }
}
```

### Performans Ayarları (performance_config.json)
```json
{
    "cpu_optimization": {
        "enable_performance_mode": true,
        "nice_value": -10,
        "cpu_affinity": [2, 3]
    },
    "memory_optimization": {
        "disable_hdmi": true,
        "gpu_memory_split": 64
    },
    "processing": {
        "target_fps": 15,
        "min_contour_area": 500,
        "max_contour_area": 50000,
        "frame_skip_threshold": 0.8
    }
}
```

---

## 📊 Performans Metrikleri

### Tipik Performans Değerleri (Raspberry Pi 4)
- **İşleme Hızı:** 12-18 FPS (320x240 çözünürlükte)
- **CPU Kullanımı:** %40-60
- **RAM Kullanımı:** ~150-200MB
- **Tespit Gecikmesi:** 50-80ms
- **Güç Tüketimi:** ~8-12W (optimizasyon ile)

### Performans İyileştirme İpuçları
1. **Çözünürlük düşürme:** 320x240 → 240x180
2. **FPS sınırlama:** 30 FPS → 20 FPS
3. **Buffer boyutu:** 2 → 1 (daha az gecikme)
4. **Renk maskeleme:** Tek renk tespiti yapma
5. **GPU overclock:** Raspberry Pi konfigürasyonunda

---

## 🚨 Sorun Giderme

### Yaygın Sorunlar ve Çözümler

#### 1. Kamera Tanınmıyor
```bash
# Problem: Kamera bulunamadı
# Çözüm:
sudo modprobe bcm2835-v4l2  # Pi kamerası için
v4l2-ctl --list-devices      # Mevcut kameraları listele
```

#### 2. Düşük FPS Performansı
```python
# Problem: FPS çok düşük
# Çözüm: Çözünürlüğü düşür
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
```

#### 3. Yüksek CPU Kullanımı
```bash
# Problem: CPU %100 kullanım
# Çözüm: Frame skipping etkinleştir
python3 src/main.py --frame-skip 2
```

#### 4. Bellek Yetersizliği
```bash
# Problem: Bellek tükeniyor
# Çözüm: GPU bellek ayırımı
sudo raspi-config
# Advanced Options > Memory Split > 64MB
```

### Debug Komutları
```bash
# Sistem durumu kontrolü
htop
iostat 1
vcgencmd measure_temp

# Kamera durumu
vcgencmd get_camera
v4l2-ctl --all

# OpenCV optimizasyon kontrolü
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

---

## 📈 Gelecek Geliştirmeler

### Planlanan Özellikler
- [ ] **YOLOv8 entegrasyonu:** Daha gelişmiş nesne tespiti
- [ ] **Multi-threading:** Paralel renk işleme
- [ ] **Kalman filtering:** Şekil takibi ve tahminleme
- [ ] **Web arayüzü:** Uzaktan monitoring
- [ ] **MQTT iletişimi:** IoT entegrasyonu
- [ ] **Edge AI optimizasyonu:** TensorFlow Lite desteği

### Performans İyileştirmeleri
- [ ] **GPU acceleration:** OpenCV CUDA desteği
- [ ] **Assembly optimizasyonu:** Kritik döngüler için
- [ ] **Memory pooling:** Bellek tahsisi optimizasyonu
- [ ] **Adaptive filtering:** Dinamik parametre ayarlama

---

## 📞 Destek ve İletişim

### Proje Sahibi
- **Geliştirici:** Enes Vural
- **GitHub:** [@enes-vural](https://github.com/enes-vural)
- **Proje Reposu:** [teknofest25_image_processing](https://github.com/enes-vural/teknofest25_image_processing)

### Lisans
Bu proje Teknofest 2025 yarışması kapsamında geliştirilmiştir. Kullanım koşulları için projenin LICENSE dosyasını inceleyiniz.

---

**Son Güncelleme:** Eylül 2025  
**Versiyon:** 1.0.0  
