from enum import Enum
import numpy as np

SHAPE_TIMEOUT = 1
default_epsilon = 0.028

isDeviceMac:bool = True
video_path = "firtina-iha/assets/object_video5.mp4"


color_names = {'r': "Red", 'g': "Green", 'b': "Blue"}
shape_colors:list = ('b','g','r')

kernelErode =np.ones((5, 5), np.uint8)
kernelOpen = np.ones((3, 3), np.uint8)

# Mavi renk aralığı
blue_lower = np.array([95, 100, 50])   # Mavi renk için alt sınır
blue_upper = np.array([140, 255, 255])  # Mavi renk için üst sınır

# Kırmızı renk aralığı
red_lower_1 = np.array([-2, 120, 50])     # Kırmızı renk için alt sınır 1
red_upper_1 = np.array([12, 255, 255])   # Kırmızı renk için üst sınır 1

red_lower_2 = np.array([165, 120, 50])   # Kırmızı renk için alt sınır 2
red_upper_2 = np.array([185, 255, 255])  # Kırmızı renk için üst sınır 2


class Color(Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"

class Shape(Enum):
    SQUARE = "Square"
    HEXAGON = "Hexagon"
    TRIANGLE = "Triangle"