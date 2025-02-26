import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


#video path for fetch the video that Azize's created.
video_path = "firtina-iha/assets/object_video5.mp4"

#capture the video with given path as video_path
video = cv2.VideoCapture(video_path)

#test

def checkVideoStartState():
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

checkVideoStartState()


#initializeKeyBinds() function is used to listen keyboard actions
# if you press 'q' key, it will close the window
# if you dont call this function, you can not show your images with opencv
def initalizeKeyBinds()->bool:  
    #keybind is a variable that holds the key value that you pressed ('q')
    keybind = cv2.waitKey(1) &0xFF 
    if keybind == ord('q'):
        #cv2.destroyAllWindows() is used to close the window              
        cv2.destroyAllWindows()
        return True
    return False
   

#gets frame width of video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#gets frame height of video
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#get frame count of video
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#get fps of video
fps = int(video.get(cv2.CAP_PROP_FPS))

#show video properties when soft started
print("Video Properties")
print("----------------------")
#print video width
print("Width: ",frame_width)
#print video height
print("Height: ",frame_height)
#print video total frame count
print("Frame Count: ",frame_count)
#print video fps (one time)
print("FPS: ",fps)
print("----------------------")

#method for passing
def nothing():
    pass

cam = cv2.VideoCapture(0)
#capture the video from camera
#camera assigned to zero
#zero is the default camera of computer.

#function for get domainant color of shape with histogram
def getDominantColor(x,y,w,h):
    #constants list for shape colors with Blue Green Red
    shape_colors:list = ('b','g','r')
    #shape dominant index for detect shape color
    #shape_dominant_index = 1 is equals blue
    #initial value is None
    shape_dominant_index = None
    histograms = {}
    #shape_dominant_color is a dictionary that holds the dominant color of shape
    shape_dominant_color = {}
    #color_names constant for color names
    #converter from shape_colors.
    color_names = {
    'r': "Red",
    'g': "Green",
    'b': "Blue",
    }

    #crop a new window (image) from clean frame
    #the axis are x and y comes from detected shape
    #we inspect and get dominant color from this cropped window.
    crop_image = clean_frame[y:y+h,x:x+w]

    #i = index 
    #color = color
    #gets color names from list with indexs
    for i,color in enumerate(shape_colors):
        #create new instance with named color_hist
        #color_hist assigns the value comes from cv2's calcHist() function
        #calcHist() function gets the histogram of the image
        color_hist = cv2.calcHist([crop_image],[i],None,[256],[0,256])
        #histograms dictionary holds the color_hist values
        histograms[color] = color_hist
    #gets the max intensity of histograms
    #get the intensity of the color and assign it to shape_dominant_color
    for color in histograms:
        max_intensity = np.argmax(histograms[color])
        shape_dominant_color[color] = max_intensity
    #get the maxium intensity and assign it to shape_dominant_index
    shape_dominant_index = max(shape_dominant_color,key=shape_dominant_color.get)
    #return the maximized (most intense & dominant) color of shape 
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

#get border dominant color of shape with histogram
#the border color is need for detect weight's field.
#when the weight has been dropped to the field.
#we need to detect the field that dropped.
#Example: Blue Weight in Red Triangle
#If we want to understand the triangle's color, we need to get border dominant color.
def getBorderDominantColor(x, y, w, h, approx):
    #border colors list for detect border color
    border_dominant_color = {}
    #color names constant for color names
    histograms = {}
    #border_colors constant for color names
    border_colors = ('b', 'g', 'r')
    #border_dominant_index = None
    #same with the shape method
    border_dominant_index = None
    histograms = {}
    border_dominant_color = {}
    #color_names constant for color names
    color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
    }

    #crop the first image from clean frame
    #crop the image with x,y,w,h axis
    crop_image_outline = clean_frame[y:h + y, x:w + x]

    #the border space for detect
    #we need bigger px for here
    #reason: we need to detect the border color with intense
    border_width = 100  # Kenar bandı genişliği
    #get outline frame from clean frame with axis y and height
    outline = clean_frame[y:y + h, x:x + w] 

    #mask the current shape(weight)
    #create mask with zeros
    mask = np.zeros_like(outline, dtype=np.uint8)
    #draw contours for mask
    cv2.drawContours(mask, [approx - [x, y]], -1, (255, 255, 255), thickness=border_width)  # Kenar kalınlığı
    #combine the outline and mask
    #TODO:
    edge_only = cv2.bitwise_and(outline, outline, mask=mask[:, :, 0])

    cv2.imshow("Edge Only", edge_only)

    #Border dominant color için histogram
    #i = index
    #color = color
    #get the color names 
    for i, color in enumerate(border_colors):
        #calculate edge color histogram
        #edge color histogram calculates the border's color histogram
        edge_color_hist = cv2.calcHist([crop_image_outline], [i], None, [256], [0, 256])
        #assign the calculated histogram.
        histograms[color] = edge_color_hist

    for color in histograms:
        #get the max intensity of histograms
        max_intensity = np.argmax(histograms[color])
        #detect the border dominant color with max intensity
        border_dominant_color[color] = max_intensity

    #find the the most dominant(intense) color from border_dominant_color
    border_dominant_index = max(border_dominant_color, key=border_dominant_color.get)
    #return the most dominant(intense) color of border
    return [color_names[border_dominant_index]]

while True:

    #state = status fo capture read function
    #if state is not True, it means video has been ended or not started successfully
    
    # state,frame = cam.read()
    #--------------------------
    #this config is need for video read() function
    state,frame = video.read()


    if not state:
        print("Video Ended")
        #Set video to the beginning
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #continue to read the video
        continue
    
    #create a new frame to copy the original frame
    #----------- HSV CONFIG ------------
    clean_frame = frame.copy()

    hsv_image = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Mavi renk aralığı
    blue_lower = np.array([95, 100, 50])   # Mavi renk için alt sınır
    blue_upper = np.array([140, 255, 255])  # Mavi renk için üst sınır

    # Kırmızı renk aralığı
    red_lower_1 = np.array([-2, 120, 50])     # Kırmızı renk için alt sınır 1
    red_upper_1 = np.array([12, 255, 255])   # Kırmızı renk için üst sınır 1

    red_lower_2 = np.array([165, 120, 50])   # Kırmızı renk için alt sınır 2
    red_upper_2 = np.array([185, 255, 255])  # Kırmızı renk için üst sınır 2

    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Kırmızı için maske oluşturun (iki aralık)
    red_mask_1 = cv2.inRange(hsv_image, red_lower_1, red_upper_1)
    red_mask_2 = cv2.inRange(hsv_image, red_lower_2, red_upper_2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    overlap_mask = cv2.bitwise_and(blue_mask, red_mask)

    final_red = cv2.subtract(red_mask, overlap_mask)
    final_blue = cv2.subtract(blue_mask, overlap_mask)

    final_mask = cv2.bitwise_or(final_red, final_blue)
    cv2.imshow("Final Mask",final_mask)

    hsv_result = cv2.bitwise_and(frame, frame, mask=final_mask)
    
    gray_result = cv2.cvtColor(hsv_result, cv2.COLOR_BGR2GRAY)

    gray_result = cv2.medianBlur(gray_result, 7)
    gray_result = cv2.erode(gray_result, kernel=np.ones((5, 5), dtype=np.uint8), iterations=3)
    gray_result = cv2.morphologyEx(gray_result, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=3)

    #-------------------------------------------

    #---------------- GRAY CONFIG ---------------
    clean2_frame = frame.copy()

    gray_frame = cv2.cvtColor(clean2_frame,cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    equalized_frame = cv2.equalizeHist(blurred_frame)
    _, otsu_thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh_val, _ = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    w_edges = cv2.Canny(blurred_frame, otsu_thresh_val * 0.5, otsu_thresh_val)
    fields = cv2.Canny(gray_result, 100, 200)
 
    #cv2.imshow("Targets",hsv_result)
    #cv2.imshow("Gray",gray_result)
    #cv2.imshow("Edge-2s",edges)
    # cv2.imshow("Result", result_with_white)



    #--------- BLUR ---------- 
    # blurred_frame = gray_frame
    #blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    #blurred_frame = cv2.bilateralFilter(gray_frame,9,75,75)
    # blurred_frame = cv2.medianBlur(gray_frame,ksize=5)
    #blurred_frame = cv2.erode(blurred_frame,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #blurred_frame = cv2.dilate(blurred_frame,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #blurred_frame = cv2.morphologyEx(blurred_frame,cv2.MORPH_OPEN,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)
    # blurred_frame = cv2.morphologyEx(blurred_frame,cv2.MORPH_CLOSE,kernel=np.ones((3,3),dtype=np.uint8),iterations=2)

    #equalized_frame = cv2.equalizeHist(blurred_frame) (detayları manyak belirtiyor paraziti arttırıyor).

    #--------- DILATE ---------
    # dilateKernel = np.ones((3,3),dtype=np.uint8)
    # blurred_frame = cv2.dilate(blurred_frame,dilateKernel,iterations=5)

    #--------- THRESHOLD ----------
    #static threshold
    # _, otsu_thresh_val = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #adaptive threshold
    # adaptive_threshold = cv2.adaptiveThreshold(blurred_frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,9)

    #detects the shape's edges with canny function

    # combined_threshold = cv2.bitwise_and(adaptive_threshold,blurred_frame)
    # blurred_frame = combined_threshold

    weight_contours,_ = cv2.findContours(w_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    field_contours,_ = cv2.findContours(fields,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    filled_frame = np.zeros_like(fields)

    cv2.drawContours(filled_frame,field_contours,-1,(255,255,255),thickness=cv2.FILLED)
    cv2.imshow("Second Frame",w_edges)
    #cv2.imshow("Weight Frame",weight_frame)
    #cv2.imshow("Filled Frame",filled_frame)

    if len(weight_contours) > 0:
        if (weight_contours is None):
            print("OK")
            break

        for w_cont in weight_contours:
            w_epsilon = 0.028*cv2.arcLength(w_cont,True)
            w_approx = cv2.approxPolyDP(w_cont,w_epsilon,True)
            w_area = cv2.contourArea(w_cont)
            w_edge_count = len(w_approx)

            if(w_area > 2500):
                if(w_edge_count == 3):
                    cv2.drawContours(frame,[w_approx],-1,(255,255,0),2)
                    #x = axis x (top left)
                    #y = axis y (top left)
                    #w = shape's width
                    #h = shape's height
                    #get the values from approx with boundingRect function
                    x,y,w,h = cv2.boundingRect(w_approx)
                    #get the dominant color of triangle
                    color = getDominantColor(x,y,w,h)

                    #if triangle was red
                    #this is the what we expected in competition
                    #because the red triangle is the target for the weight
                    if(color == "Red"):
                          #if shape was found, now draw the shape with green color
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                        print(f"Triangle : {area}")

                        #draw the rectangle with red color
                        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                        #print axis x y w h
                        #print(f"Axis X: {x} Axis Y: {y} Width: {w} Height: {h}")

                        #the center axis X of triangle.
                        center_x = x+w/2
                        #the center axis Y of triangle.
                        center_y = y+h/1.5
                        #draw the circle with center of triangle
                        cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                        #draw the text with "Triangle Target" text
                        cv2.putText(frame,"Triangle Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        

                if(w_edge_count==4):

                    #TODO: AREA Değeri Ekrandaki pixel sayısına uygun mu veriyor
                    #bunu test etmek lazım yoksa farklı çözünürlüklerde farklı area
                    #durumlarında ratio ile responsive bir algılama şekli yapılması gerekebilir.
                    #TODO: Enhancement ISSUE #1

                    #print the area of shape
                    print(f"AREA: {w_area}")
                
                    #get the values from approx with boundingRect function
                    x,y,w,h = cv2.boundingRect(w_approx)

                    #aspect ratio is a value that we use for detect the shape's ratio
                    #if the shape is square, the ratio is 1.0 or close to 1.0
                    #we can obviously say that the shape is square
                    #else the shape is rectangle.

                    #aspect_ratio = width/height => convert float
                    aspect_ratio = float(w)/h

                    #the aspect ratio min value: 0.88
                    #the aspect ratio max value: 1.30
                    #if the aspect ratio is between 0.88 and 1.30
                    #we can say that the shape is square
                    #0.88 <= aspect_ratio <= 1.30
                    #configuration settings for video mode.
                    #in real life, we need to change the values
                    if (0.80<= aspect_ratio <= 1.30):
                        print(f"Ratio: {aspect_ratio}")
                        #if shape was found, now draw the shape with green color
                        cv2.drawContours(frame, [w_approx], -1, (0, 255, 0), 2)
                        #get the dominant color of square (weight)
                        shape_color = getDominantColor(x,y,w,h)
                        #TODO:
                        #video sonunda üçgene yaklaşırken bütünü kare zannediyor.
                        #color = getBorderDominantColor(x,y,w,h,approx)

                        #put the text with "Weight" text into square (weight)
                        cv2.putText(frame,f"Weight {shape_color}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    #if contours length is greater than 0, it means we found the shape
    if len(field_contours) > 0:

        for idx,cont in enumerate(field_contours):
            #-------------EPSILON-------------
            #0.026 is a constant value for epsilon
            #if epsion is increase the shape's can found more easy
            #but the software can detect multiple and needless shapes.

            #epsilon is a value that we use for approxPolyDP function
            epsilon = 0.028*cv2.arcLength(cont,True)
            #approx is a value that we use for instance of object
            approx = cv2.approxPolyDP(cont,epsilon,True)
            #area is a value that we use for detect the shape's area
            area = cv2.contourArea(cont)

            #if area is greater than 250, it means we found the shape
            #little shapes are not important for us.
            if area > 2500:

                #edge_count = approx's length
                #edge_count is a value that we use for detect the shape count
                edge_count = len(approx)

                #if expected shape not found, continue and dont draw the shape's contours
                if(edge_count is None or edge_count != 3 and edge_count != 4 and edge_count != 6):
                    continue
                else:
                    #if shape was found, now draw the shape with green color
                    #cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    pass


                #------------ FOR CIRCLE -----------
                # Circularity calculation
                # (x, y), radius = cv2.minEnclosingCircle(cont)
                # circularity = (4 * np.pi * area) / (cv2.arcLength(cont, True) ** 2)

                #If shape was Triangle
                if(edge_count==3) and (area > 500):
                    #x = axis x (top left)
                    #y = axis y (top left)
                    #w = shape's width
                    #h = shape's height
                    #get the values from approx with boundingRect function
                    x,y,w,h = cv2.boundingRect(approx)
                    #get the dominant color of triangle
                    color = getDominantColor(x,y,w,h)

                    #if triangle was red
                    #this is the what we expected in competition
                    #because the red triangle is the target for the weight
                    if(color == "Red"):
                          #if shape was found, now draw the shape with green color
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                        print(f"Triangle : {area}")

                        #draw the rectangle with red color
                        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                        #print axis x y w h
                        #print(f"Axis X: {x} Axis Y: {y} Width: {w} Height: {h}")

                        #the center axis X of triangle.
                        center_x = x+w/2
                        #the center axis Y of triangle.
                        center_y = y+h/1.5
                        #draw the circle with center of triangle
                        cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                        #draw the text with "Triangle Target" text
                        cv2.putText(frame,"Triangle Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        
                #if shape was Square
                #The square is the weight that we dropped
                #The square is the weight that we need to detect in the field
                if(edge_count==4):

                    #TODO: AREA Değeri Ekrandaki pixel sayısına uygun mu veriyor
                    #bunu test etmek lazım yoksa farklı çözünürlüklerde farklı area
                    #durumlarında ratio ile responsive bir algılama şekli yapılması gerekebilir.
                    #TODO: Enhancement ISSUE #1

                    #print the area of shape
                    print(f"AREA: {area}")
                    #print the idx of shape
                    #idx is the number of detected shape (identifier)
                    print(f"IDX: {idx}")
                
                    #get the values from approx with boundingRect function
                    x,y,w,h = cv2.boundingRect(approx)

                    #aspect ratio is a value that we use for detect the shape's ratio
                    #if the shape is square, the ratio is 1.0 or close to 1.0
                    #we can obviously say that the shape is square
                    #else the shape is rectangle.

                    #aspect_ratio = width/height => convert float
                    aspect_ratio = float(w)/h

                    #the aspect ratio min value: 0.88
                    #the aspect ratio max value: 1.30
                    #if the aspect ratio is between 0.88 and 1.30
                    #we can say that the shape is square
                    #0.88 <= aspect_ratio <= 1.30
                    #configuration settings for video mode.
                    #in real life, we need to change the values
                    if (0.80<= aspect_ratio <= 1.30):
                        #if shape was found, now draw the shape with green color
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                        #get the dominant color of square (weight)
                        shape_color = getDominantColor(x,y,w,h)
                        #TODO:
                        #video sonunda üçgene yaklaşırken bütünü kare zannediyor.
                        #color = getBorderDominantColor(x,y,w,h,approx)

                        #put the text with "Weight" text into square (weight)
                        cv2.putText(frame,f"Weight {shape_color}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                #if shape was hexagon
                #the hexagon is the target for the weight
                #the hexagon should be blue
                #and area must be greater than 300
                #because the cv2 detects other circles as hexagon
                #we need to filter the noisy shapes with area
                if(edge_count==6) and (area >300):
                        #get the x,y,w,h axis with boundingRect function
                        x,y,w,h = cv2.boundingRect(approx)
                        #get the dominant color of hexagon
                        #it must be blue
                        color = getDominantColor(x,y,w,h)

                        #if hexagon was blue
                        #blue is expected target in competition
                        if(color == "Blue"):
                            #if shape was found, now draw the shape with green color
                            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                            #get hexagon's one edge length (meter)
                            edge_meter = np.sqrt(((h/2)*(h/2) + (w/2)*(w/2)))
                            #print edge's meter.
                            print(f"Edge Meter: {edge_meter}")

                            #1.154 expected ratio
                            #width = height*2/np.sqrt(3) =? 1.154 * height

                            #calculate hexagon's ratio.
                            #calculated ratio equals = width / height
                            calculated_ratio = w/h
                            #print the calculated ratio
                            print(f"Calculated Ratio: {calculated_ratio}")
                            
                            #calcualte the ratio with given formula
                            #height*2 / 3^2 * 1.154 * height
                            ratio= h*2/np.sqrt(3) *1.154 *h

                            #if the calculated ratio is between 1.00 and 1.20
                            #we can say that the shape is hexagon
                            if (1.00 <= calculated_ratio <= 1.20):
                            #draw rectangle with blue color outline of hexagon
                            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                                #put text with "Hexagon Target" text to field
                                cv2.putText(frame,"Hexagon Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                                #put circle to center of hexagon
                                cv2.circle(frame,(int(x+w/2),int(y+h/2)),5,(0,0,255),-1)
    else:
        #removed print method for cleaner terminal
        #print("No Contours Found")
        pass

    #show original frame with imshow function
    cv2.imshow("Original",frame)

    #if you press 'q' key, it will close the window
    #and exit the loop
    if initalizeKeyBinds():
        break




   