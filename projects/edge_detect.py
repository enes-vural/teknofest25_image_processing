import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


video_path = "assets/object_video5.mp4"
#video read
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

def initalizeKeyBinds()->bool:          # if you dont call this func you can not show your images with opencv
    keybind = cv2.waitKey(1) &0xFF      # 0xFF converts & marks the keybinds & range of 0 - 255 / q means = 113
                                        # if you set waitKey to Zero, your keyboard actions gonna listen until you press a keybind, so we set it to 1
    if keybind == ord('q'):             # returns 113 for q value
        cv2.destroyAllWindows()
        return True
    return False
   

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))


print("Width: ",frame_width)
print("Height: ",frame_height)
print("Frame Count: ",frame_count)
print("FPS: ",fps)

def nothing():
    pass





#cv2.createTrackbar()
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",0,255,nothing)
cv2.createTrackbar("L-S","Trackbars",42,255,nothing)
cv2.createTrackbar("L-V","Trackbars",181,255,nothing)
cv2.createTrackbar("U-H","Trackbars",200,255,nothing)
cv2.createTrackbar("U-S","Trackbars",255,255,nothing)
cv2.createTrackbar("U-V","Trackbars",255,255,nothing)


while True:
    state,frame = video.read()
    if not state:
        print("Video Ended")
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Videoyu başa al
        continue

    clean_frame = frame.copy()
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    hsv = cv2.medianBlur(hsv,ksize=5)
    hsv = cv2.erode(hsv,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #hsv = cv2.morphologyEx(hsv.astype(np.float32),cv2.MORPH_CLOSE,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)
    hsv = cv2.morphologyEx(hsv,cv2.MORPH_OPEN,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)

    lh = cv2.getTrackbarPos("L-H","Trackbars")
    ls = cv2.getTrackbarPos("L-S","Trackbars")
    lv = cv2.getTrackbarPos("L-V","Trackbars")
    uh = cv2.getTrackbarPos("U-H","Trackbars")
    us = cv2.getTrackbarPos("U-S","Trackbars")
    uv = cv2.getTrackbarPos("U-V","Trackbars")


    lower = np.array([lh,ls,lv])
    upper = np.array([uh,us,uv])

    mask = cv2.inRange(hsv,lower,upper)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)

    # # Yuvarlaklar tespit edilirse, bunları çizin
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     for (x, y, r) in circles:
    #         # Yuvarlağı çizin
    #         cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    #         # Yuvarlağın merkezini işaretle
    #         cv2.circle(frame, (x, y), 5, (0, 0, 255), 3)


    #Contours
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    equalized_frame = cv2.equalizeHist(blurred_frame)
    _, otsu_thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh_val, _ = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred_frame, otsu_thresh_val * 0.5, otsu_thresh_val)

    cv2.imshow("Edges",edges)

    #edges = cv2.Canny(gray_frame, 50, 150)  # Kenar tespiti
    contours,_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:

        colors = ('b','g','r')
        histograms = {}
        dominantColor = {}
        color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
        }

        for cont in contours:
            #0.026
            epsilon = 0.026*cv2.arcLength(cont,True)
            approx = cv2.approxPolyDP(cont,epsilon,True)
            area = cv2.contourArea(cont)
            if area > 500:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                edge_count = len(approx)
                print("Edge Count: ",edge_count)
              
                (x, y), radius = cv2.minEnclosingCircle(cont)
                circularity = (4 * np.pi * area) / (cv2.arcLength(cont, True) ** 2)

                if(edge_count==3):
                    x,y,w,h = cv2.boundingRect(approx)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    print(f"Axis X: {x} Axis Y: {y} Width: {w} Height: {h}")
                    center_x = x+w/2
                    center_y = y+h/1.5
                    cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                    cv2.putText(frame,"Triangle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                if(edge_count==4):
                    x,y,w,h = cv2.boundingRect(approx)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    center_x = x+w/2
                    center_y = y+h/2
                    cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                    cv2.putText(frame,"Rectangle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                if(edge_count==6):
                    x,y,w,h = cv2.boundingRect(approx)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.putText(frame,"Hexagon",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    cv2.circle(frame,(int(x+w/2),int(y+h/2)),5,(255,0,0),-1)

                if edge_count ==3 or edge_count ==4 or edge_count ==6:
                    #crop_image = clean_frame[x:y+h,x:x+w]
                    crop_image = clean_frame[y:y+h,x:x+w]
                    print(f"X {x}",x)
                    print(f"Y {y}",y)
                    print(f"W {w}",w)
                    #time.sleep(0.08)
                    #cv2.imshow("Cropped",crop_image)

                    crop_image_hist = cv2.calcHist([crop_image],[0],None,[256],ranges=[0,256])

                    for i,color in enumerate(colors):
                        color_hist = cv2.calcHist([crop_image],[i],None,[256],[0,256])
                        histograms[color] = color_hist

                    for color in histograms:
                        max_insensity = np.argmax(histograms[color])
                        dominantColor[color] = max_insensity
                    
                    dominantIndex = max(dominantColor,key=dominantColor.get)
                    print("afjkbajfsaf"+ color_names[dominantIndex])
                    if(color_names[dominantIndex] == 'Red'):
                        cv2.putText(frame,"Red",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    elif(color_names[dominantIndex]  == 'Green'):
                        cv2.putText(frame,"Green",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    elif(color_names[dominantIndex]  == 'Blue'):
                        cv2.putText(frame,"Blue",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                    print(dominantColor)
 


                
    else:
        print("No Contours Found")


    

    cv2.imshow("Original",frame)
    cv2.imshow("Mask",mask)

    if initalizeKeyBinds():
        break


