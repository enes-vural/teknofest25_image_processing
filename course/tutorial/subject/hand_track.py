import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import time


def initalizeKeyBinds()->bool:          # if you dont call this func you can not show your images with opencv
    keybind = cv2.waitKey(1) &0xFF      # 0xFF converts & marks the keybinds & range of 0 - 255 / q means = 113
                                        # if you set waitKey to Zero, your keyboard actions gonna listen until you press a keybind, so we set it to 1
    if keybind == ord('q'):             # returns 113 for q value
        cv2.destroyAllWindows()
        return True
    return False
   


#kamera aç (0.kamera default cam)
capture = cv2.VideoCapture(0)

#hand detection module
mpHand = mp.solutions.hands
#hands method has been called to Hands
hands = mpHand.Hands(
    static_image_mode=True,
    max_num_hands = 1,
    model_complexity =1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
)


mpDraw = mp.solutions.drawing_utils

while True:
    cam_state,captureImage = capture.read()

    if not cam_state:
        print("Web Cam State has an Error.")
        break

    #bu sefer RGB geldi BGR a çevirmeye gerek yok
    #captureImage = cv2.cvtColor(captureImage,cv2.COLOR_BGR2RGB)

    results = hands.process(captureImage)
    #print(results.multi_hand_landmarks)
    gesture_data = results.multi_hand_landmarks
    if (gesture_data == None):
        None

    else:
        startPoint = None
        endPoint = None
        for hand_landmarks in gesture_data:
            mpDraw.draw_landmarks(captureImage,hand_landmarks,mpHand.HAND_CONNECTIONS)

            for id, ankle in enumerate(hand_landmarks.landmark):
                thumb = [0,1,2,3,4]
                if(thumb.__contains__(id)):
                    height, width, color = captureImage.shape
                    
                    if(id == 0):
                        Icordx,Icordy = ankle.x,ankle.y

                        IthumbPoint = (int((width * Icordx)-50), int(height * Icordy))
                        startPoint= IthumbPoint
                    if(id == 4):
                        Ecordx,Ecordy = ankle.x,ankle.y

                        EthumbPoint = (int((width * Ecordx)-20), int((height * Ecordy-30)))
                        endPoint = EthumbPoint
                    
                    if startPoint is not None and endPoint is not None:
                        print("Start Point")
                        print(startPoint)

                        print("EndPoint")
                        print(endPoint)
                        print("OK")
                        cv2.rectangle(captureImage,pt1=startPoint,pt2=endPoint,color=(0,255,0))
                        cv2.putText(captureImage,"Enes's thumb",endPoint,fontFace=2,fontScale=2.0,color=(0,255,0),thickness=1,)
                    #print(thumbPoint)


    cv2.imshow("Cam",captureImage)
    
    if initalizeKeyBinds():
        break


