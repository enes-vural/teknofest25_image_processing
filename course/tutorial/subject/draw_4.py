import cv2
import numpy as np


#create image
# 3 => means count color channels ex: B G R (3 channel)
img = np.zeros((512,512,3),np.uint8) #black image with 512x512 sizes

#print size of image
print(img.shape)
cv2.imshow("Window",img)

#draw line
#if you give same window ne, the changes sets automatiically during build.
#else circums your project will create a new window instance for each different changes :)

#(image, (start point), (end point) color, line_width)
#(0,0) => top left
cv2.line(img,(100,100),(412,412),(0,255,0),3)
cv2.imshow("Window",img)

#draw rectangle
#empty rectangle
#(image, (start point), (end point), rectangle_width)

#filled rectangle
#(image, (start point), (end point), cv2.FILLED)
cv2.rectangle(img,(0,0),(100,100),(0,255,0),cv2.FILLED) #(image, (start point), (end point), rectangle_width)
cv2.imshow("Window",img)

#draw circle
#(image, (center point), radius (color), {if filled ? cv2.FILLED else null})
cv2.circle(img,(216,216),60,(255,0,0),cv2.FILLED)
cv2.imshow("Window",img)

#(image, text, (begin point), cv2.FONT, fontSize, color, fontWidth)
cv2.putText(img,"Selam",(216,216),cv2.FONT_HERSHEY_PLAIN,4.0,(255,255,255),2)
cv2.imshow("Window",img)

# Wait for a key press
cv2.waitKey(0)  # Sonsuza kadar bekler, bir tuşa basılmasını bekler
cv2.destroyAllWindows()  # Tüm pencereleri kapatır