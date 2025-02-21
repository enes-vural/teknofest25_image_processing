import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

#mv ~/.espressif ~/.espressif_backup


video_path = "assets/object_video5.mp4"
#video read
video = cv2.VideoCapture(video_path)

# cam = cv2.VideoCapture(0),
#test


# if not video.isOpened():
#     print("Error: Could not open video.")
#     exit()

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

print("Video Properties")
print("----------------------")
print("Width: ",frame_width)
print("Height: ",frame_height)
print("Frame Count: ",frame_count)
print("FPS: ",fps)
print("----------------------")

def nothing():
    pass


def getDominantColor(x,y,w,h):
    shape_colors = ('b','g','r')
    shape_dominant_index = None
    border_dominant_index = None
    histograms = {}
    shape_dominant_color = {}
    color_names = {
    'r': "Red",
    'g': "Green",
    'b': "Blue",
    }

    #get crop frame.
    crop_image = clean_frame[y:y+h,x:x+w]

    for i,color in enumerate(shape_colors):
        color_hist = cv2.calcHist([crop_image],[i],None,[256],[0,256])
        histograms[color] = color_hist

    for color in histograms:
        max_insensity = np.argmax(histograms[color])
        shape_dominant_color[color] = max_insensity

    shape_dominant_index = max(shape_dominant_color,key=shape_dominant_color.get)
    return color_names[shape_dominant_index]

    # if(color_names[shape_dominant_index] == 'Red'):
    #     cv2.putText(frame,"Red",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    #     if(border_dominant_index is not None and color_names[border_dominant_index]=="Red"):
    #         print("Second Weight(Red) has dropped into Triangle(Red) field")

    # if(color_names[shape_dominant_index]  == 'Green'):
    #     cv2.putText(frame,"Green",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #     #not necessary

    # if(color_names[shape_dominant_index]  == 'Blue'):
    #     cv2.putText(frame,"Blue",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    #     if(border_dominant_index is not None and color_names[border_dominant_index]=="Blue"):
    #         print("First Weight(Blue) has dropped into Hexagon(Blue) field")

    #     if(border_dominant_index is not None and color_names[border_dominant_index]=="Red"):
    #         print("First Weight(Blue) has dropped into Triangle(Red) field")
    #         print("Bonus Mission has been completed")


def getBorderDominantColor(x, y, w, h, approx):
    border_dominant_color = {}
    histograms = {}
    border_colors = ('b', 'g', 'r')
    border_dominant_index = None
    histograms = {}
    border_dominant_color = {}
    color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
    }

    # İlk kareyi kes
    crop_image_outline = clean_frame[y:h + y, x:w + x]

    # Kenar bölgesi oluştur
    border_width = 100  # Kenar bandı genişliği
    outline = clean_frame[y:y + h, x:x + w]  # Şekli crop'la

    # Maskeyi oluştur ve kenarı izole et
    mask = np.zeros_like(outline, dtype=np.uint8)
    cv2.drawContours(mask, [approx - [x, y]], -1, (255, 255, 255), thickness=border_width)  # Kenar kalınlığı
    edge_only = cv2.bitwise_and(outline, outline, mask=mask[:, :, 0])

    # Kenar histogramını al
    edge_histogram = cv2.calcHist([edge_only], [0], mask[:, :, 0], [256], [0, 256])
    cv2.imshow("Edge Only", edge_only)

    # Border dominant color için histogram
    for i, color in enumerate(border_colors):
        edge_color_hist = cv2.calcHist([crop_image_outline], [i], None, [256], [0, 256])
        histograms[color] = edge_color_hist

    for color in histograms:
        max_intensity = np.argmax(histograms[color])
        border_dominant_color[color] = max_intensity

    # En dominant rengi bul
    border_dominant_index = max(border_dominant_color, key=border_dominant_color.get)

    
    return [color_names[border_dominant_index]]

capture = cv2.VideoCapture(0)
while True:
    state,frame = capture.read()
    #state,frame = video.read() (default)
    if not state:
        print("Video Ended")
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Videoyu başa al
        continue

    clean_frame = frame.copy()
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    #hsv = cv2.medianBlur(hsv,ksize=5)
    #hsv = cv2.erode(hsv,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #hsv = cv2.morphologyEx(hsv.astype(np.float32),cv2.MORPH_CLOSE,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)
    #hsv = cv2.morphologyEx(hsv,cv2.MORPH_OPEN,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)

    #Contours
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #default 5,5 (kernel size)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    #blurred_frame = cv2.medianBlur(gray_frame,ksize=5)
    #blurred_frame = cv2.erode(blurred_frame,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #blurred_frame = cv2.morphologyEx(blurred_frame,cv2.MORPH_OPEN,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)

    #equalized_frame = cv2.equalizeHist(blurred_frame) (detayları manyak belirtiyor paraziti arttırıyor).
    _, otsu_thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh_val, _ = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred_frame, otsu_thresh_val *0.4, otsu_thresh_val)

    cv2.imshow("Edges",edges)

    #edges = cv2.Canny(gray_frame, 50, 150)  # Kenar tespiti
    contours,_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:

        shape_colors = ('b','g','r')
        border_colors = ('b','g','r')
        shape_dominant_index = None
        border_dominant_index = None
        histograms = {}
        border_histograms = {}
        shape_dominant_color = {}
        border_dominant_color = {}
        color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
        }

        for idx,cont in enumerate(contours):
            #0.026
            epsilon = 0.026*cv2.arcLength(cont,True)
            approx = cv2.approxPolyDP(cont,epsilon,True)
            area = cv2.contourArea(cont)
            if area > 250:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                edge_count = len(approx)
                #print("Edge Count: ",edge_count)
              
                (x, y), radius = cv2.minEnclosingCircle(cont)
                circularity = (4 * np.pi * area) / (cv2.arcLength(cont, True) ** 2)

                if(edge_count==3):
                    x,y,w,h = cv2.boundingRect(approx)
                    color = getDominantColor(x,y,w,h)
                    if(color == "Red"):
                        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        #print(f"Axis X: {x} Axis Y: {y} Width: {w} Height: {h}")
                        center_x = x+w/2
                        center_y = y+h/1.5
                        cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                        cv2.putText(frame,"Triangle Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        #print("Second Triangle Target found")
                        

                if(edge_count==4):
                    print(f"AREA: {area}")
                    print(f"IDX: {idx}")
                    #print(f"H: {h}")

                    x,y,w,h = cv2.boundingRect(approx)
                    aspect_ratio = float(w)/h
                    if (0.88<= aspect_ratio <= 1.30):
                        shape_color = getDominantColor(x,y,w,h)
                        #video sonunda üçgene yaklaşırken bütünü kare zannediyor.
                        #color = getBorderDominantColor(x,y,w,h,approx)

                        # if color is not None and shape_color is not None:
                        #     print(f"Color Border: {color}, Color Shape: {shape_color}")
                        
                        cv2.putText(frame,f"Weight {shape_color}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                if(edge_count==6) and (area >300):
                        x,y,w,h = cv2.boundingRect(approx)
                        color = getDominantColor(x,y,w,h)
                        if(color == "Blue"):
                            edge_meter = np.sqrt(((h/2)*(h/2) + (w/2)*(w/2)))
                            print(f"Edge Meter: {edge_meter}")
                            #1.154 expected ratio
                            #width = height*2/np.sqrt(3) =? 1.154 * height
                            calculated_ratio = w/h
                            print(f"Calculated Ratio: {calculated_ratio}")
                            ratio= h*2/np.sqrt(3) *1.154 *h
                            if (1.00 <= calculated_ratio <= 1.20):
                            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                                cv2.putText(frame,"Hexagon Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                                cv2.circle(frame,(int(x+w/2),int(y+h/2)),5,(255,0,0),-1)

                        #print("First Hexagon Target found")

                # if edge_count ==3 or edge_count ==4 or edge_count ==6:
                    #crop_image = clean_frame[x:y+h,x:x+w]
                    # crop_image = clean_frame[y:y+h,x:x+w]
                    
                    #print(f"X {x}",x)
                    #print(f"Y {y}",y)
                    #print(f"W {w}",w)
                    #time.sleep(0.08)
                    #cv2.imshow("Cropped",crop_image)

                    # if(edge_count ==4):
                    #     print("Weight found")

                    #     aspect_ratio = float(w)/h
                    #     # print(f"RATIO {aspect_ratio}")
                    #     # if not (0.88 <= aspect_ratio <= 1.30):
                    #     #     continue
                    #     # else:
                    #     # İlk kareyi kes
                    #     crop_image_outline = clean_frame[y:h+y, x:w+x]
                        
                    #     # İlk kare için histogramı hesapla
                    #     crop_image_hist2 = cv2.calcHist([crop_image_outline], [0], None, [256], [0, 256])

                    #     # Kenar bölgesi oluştur
                    #     border_width = 100  # Kenar bandı genişliği
                    #     outline = clean_frame[y:y+h, x:x+w]  # Şekli crop'la
                        
                    #     # Maskeyi oluştur ve kenarı izole et
                    #     mask = np.zeros_like(outline, dtype=np.uint8)
                    #     cv2.drawContours(mask, [approx - [x, y]], -1, (255, 255, 255), thickness=border_width)  # Kenar kalınlığı
                    #     edge_only = cv2.bitwise_and(outline, outline, mask=mask[:, :, 0])

                    #     # Kenar histogramını al
                    #     edge_histogram = cv2.calcHist([edge_only], [0], mask[:, :, 0], [256], [0, 256])
                    #     cv2.imshow("Edge Only", edge_only)

                    #     # Border dominant color için histogram
                    #     histograms = {}
                    #     for i, color in enumerate(border_colors):
                    #         edge_color_hist = cv2.calcHist([crop_image_outline], [i], None, [256], [0, 256])
                    #         histograms[color] = edge_color_hist

                    #     border_dominant_color = {}
                    #     for color in histograms:
                    #         max_intensity = np.argmax(histograms[color])
                    #         border_dominant_color[color] = max_intensity

                    #     # En dominant rengi bul
                    #     border_dominant_index = max(border_dominant_color, key=border_dominant_color.get)
                    #     print("Edge Border Outline Color: " + color_names[border_dominant_index])

                    #     # İkinci kareyi işle
                    #     # İkinci kareyi bulup onun histogramını ve kenarını işle
                    #     # İkinci konturu (approx2) al, ikinci kareyi kes ve aynı işlemleri uygula
                    #     x2, y2, w2, h2 = cv2.boundingRect(approx2)  # İkinci konturu al
                    #     crop_image_outline2 = clean_frame[y2:h2+y2, x2:w2+x2]
                        
                    #     # İkinci kare için histogramı hesapla
                    #     crop_image_hist2_2 = cv2.calcHist([crop_image_outline2], [0], None, [256], [0, 256])

                    #     # İkinci kenar bölgesi oluştur
                    #     outline2 = clean_frame[y2:y2+h2, x2:x2+w2]  # İkinci şekli crop'la
                        
                    #     # Maskeyi oluştur ve ikinci kenarı izole et
                    #     mask2 = np.zeros_like(outline2, dtype=np.uint8)
                    #     cv2.drawContours(mask2, [approx2 - [x2, y2]], -1, (255, 255, 255), thickness=border_width)  # Kenar kalınlığı
                    #     edge_only2 = cv2.bitwise_and(outline2, outline2, mask=mask2[:, :, 0])

                    #     # İkinci kenar histogramını al
                    #     edge_histogram2 = cv2.calcHist([edge_only2], [0], mask2[:, :, 0], [256], [0, 256])
                    #     cv2.imshow("Edge Only 2", edge_only2)

                    #     # İkinci border dominant color için histogram
                    #     histograms2 = {}
                    #     for i, color in enumerate(border_colors):
                    #         edge_color_hist2 = cv2.calcHist([crop_image_outline2], [i], None, [256], [0, 256])
                    #         histograms2[color] = edge_color_hist2

                    #     border_dominant_color2 = {}
                    #     for color in histograms2:
                    #         max_intensity = np.argmax(histograms2[color])
                    #         border_dominant_color2[color] = max_intensity

                    #     # İkinci en dominant rengi bul
                    #     border_dominant_index2 = max(border_dominant_color2, key=border_dominant_color2.get)
                    #     print("Edge Border Outline Color 2: " + color_names[border_dominant_index2])

                        
                    # crop_image_hist = cv2.calcHist([crop_image],[0],None,[256],ranges=[0,256])


                    # for i,color in enumerate(shape_colors):
                    #     color_hist = cv2.calcHist([crop_image],[i],None,[256],[0,256])
                    #     histograms[color] = color_hist

                    # for color in histograms:
                    #     max_insensity = np.argmax(histograms[color])
                    #     shape_dominant_color[color] = max_insensity

                    # shape_dominant_index = max(shape_dominant_color,key=shape_dominant_color.get)
                    #print("(afjkbajfsaf): Color Names"+ color_names[dominantIndex])
                    #Şekilin renk histogramının koşulu
                    # if(color_names[shape_dominant_index] == 'Red'):
                    #     cv2.putText(frame,"Red",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                    #     if(border_dominant_index is not None and color_names[border_dominant_index]=="Red"):
                    #         print("Second Weight(Red) has dropped into Triangle(Red) field")

                    # if(color_names[shape_dominant_index]  == 'Green'):
                    #     cv2.putText(frame,"Green",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    #     #not necessary

                    # if(color_names[shape_dominant_index]  == 'Blue'):
                    #     cv2.putText(frame,"Blue",(x-100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                    #     if(border_dominant_index is not None and color_names[border_dominant_index]=="Blue"):
                    #         print("First Weight(Blue) has dropped into Hexagon(Blue) field")

                    #     if(border_dominant_index is not None and color_names[border_dominant_index]=="Red"):
                    #         print("First Weight(Blue) has dropped into Triangle(Red) field")
                    #         print("Bonus Mission has been completed")



                    #print(dominantColor)
 


                
    else:
        #print("No Contours Found")
        pass


    

    cv2.imshow("Original",frame)

    if initalizeKeyBinds():
        break




   