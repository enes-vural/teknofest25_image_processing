import cv2
import matplotlib.pyplot as plt         #imaginziation | pyplot folder inside mayplotlib folder 
import time

basePath:str = "/Users/sakastudio/development-py/course/"
video_path:str = basePath+"assets/MOT17-04-DPM.mp4"



capture = cv2.VideoCapture(video_path)  #video assigned to capture

if capture.isOpened():
    print("VID_OPEN: SUCCESS")
else:
    exit(TypeError('No Image Found, Check your folder path'))



capture_width,capture_height = capture.get(3), capture.get(4)  # 3 assigns => width || 4 assigns => height


# f"{variable}
print("VID_SIZE: "+ f"{capture_width}"+" x "+ f"{capture_height}")



def initalizeKeyBinds()->bool:          # if you dont call this func you can not show your images with opencv
    keybind = cv2.waitKey(1) &0xFF      # 0xFF converts & marks the keybinds & range of 0 - 255 / q means = 113
                                        # if you set waitKey to Zero, your keyboard actions gonna listen until you press a keybind, so we set it to 1
    if keybind == ord('q'):             # returns 113 for q value
        cv2.destroyAllWindows()
        return True
    return False
   

while True:
    time.sleep(0.006)                   # frame each 1 milisecond
    load_state, frame = capture.read()

    if not load_state:                  # if frame not readable
        print("Failed to read frame.")
        break

    cv2.imshow("Video", frame)          # start to show frames

    if(initalizeKeyBinds()):
        break

capture.release()                       # Release the capture.
cv2.destroyAllWindows()                 # Kill al windows.
                                        # exit(0) does not need, it automatically kills.

'''
SINGLETON PATTERN

class Enes:
    _instance = None  # Sınıf düzeyinde tek örnek için değişken

    def __new__(cls,*args):
        if not cls._instance:  # Checks is there a class has created from previous seassion
            cls._instance = super(Enes, cls).__new__(cls)  # create a new sample
        return cls._instance  # return existing instance

    def __init__(self, name: str, age: int):
        if not hasattr(self, 'initialized'):  # runs in first attempt
            self.name = name
            self.age = age
            self.initialized = True  # Başlatıldığını belirt

    def printHello(self):
        print("Hello, World!"),

a = Enes('Alper',26)
b = Enes._instance
b.printHello()

'''