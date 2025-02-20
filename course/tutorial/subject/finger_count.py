import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import time


def initalizeKeyBinds()->bool:          # if you dont call this func you can not show your images with opencv
    keybind = cv2.waitKey(1) &0xFF      # 0xFF converts & marks the keybinds & range of 0 - 255 / q means = 113
                                        # if you set waitKey to Zero, your keyboard actions gonna listen until you press a keybind, so we set it to 1
    if keybind == ord('q'):             # returns 113 for q value
        cv2.destroyAllWindows()
        return True
    return False
   

cap = cv2.VideoCapture(0)
#480x640
cap.set(3, 640)
cap.set(4, 480)


mpHand = mp.solutions.hands


hands = mpHand.Hands(
    static_image_mode=True,
    max_num_hands = 1,
    model_complexity =1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
)

mpDraw = mp.solutions.drawing_utils

def isClose(firstFinger:int,secondFinger:int)->bool:
    if landmark_list[firstFinger][2] < landmark_list[secondFinger][2]:
        print("Close")
        return False
    else:
        return True
    
def getThumbNearDistance():
    dist = landmark_list[5][1] - landmark_list[4][1]
    print("Distance: : : ") 
    print(dist)
    #-90 when open
    if(dist >25):
        print("Open")
        return False
    else:
        print("Close")
        return True


while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    landmark_list = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)

            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                landmark_list.append([id,cx,cy])
                print(id,cx,cy)
        # 1 2 3 4  arasında 4 2 'nin y axisine göre aşşağısındaysa bu parmak kapalıdır
        # buna göre sayıları algılayıp işleyeceksin kurstan bağımsız kendin yapmaya çalış
        print(landmark_list)

        if(len(landmark_list) > 0):
            if (getThumbNearDistance() and isClose(8,6) and isClose(12,11) and isClose(16,15) and isClose(20,19)):
                print("Count: 0")
                cv2.putText(img, "Count: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif not getThumbNearDistance() and isClose(8,6) and isClose(12,11) and isClose(16,14) and isClose(20,19):
                print("Count: 1")
                cv2.putText(img, "Count: 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif not getThumbNearDistance() and not isClose(8,6) and isClose(12,11):
                print("Count 2")
                cv2.putText(img, "Count: 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif not getThumbNearDistance() and not isClose(8,6) and not isClose(12,11) and isClose(16,15):
                print("Count 3")
                cv2.putText(img, "Count: 3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif not getThumbNearDistance() and not isClose(8,6) and not isClose(12,11) and not isClose(16,15) and isClose(20,19):
                print("Count 4")
                cv2.putText(img, "Count: 4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif not getThumbNearDistance() and not isClose(8,6) and not isClose(12,11) and not isClose(16,15) and not isClose(20,19):
                print("Count 5")
                cv2.putText(img, "Count: 5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                print("Non defined")
                cv2.putText(img, "Non defined", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        print("Thumb (4, 3):", isClose(4, 3))
        print("Index (8, 6):", isClose(8, 6))
        print("Middle(12, 11):", isClose(12, 11))
        print("Ring(16, 14):", isClose(16, 15))
        print("Pinky(20, 19):", isClose(20, 19))

 

    cv2.imshow("Video",img)


    if initalizeKeyBinds():
        break

