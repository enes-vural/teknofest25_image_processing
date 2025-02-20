import cv2
import matplotlib.pyplot as plt
import numpy as np


image1 = cv2.imread("assets/odev1.jpg",0)
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
plt.figure()
plt.title("Image")
plt.axis("off")
plt.imshow(image1)
size = image1.shape

print(f"Shape ${size}")

#x = 568 *0.8
#y = 860 *0.8
# 4:5

image1 = cv2.resize(image1,(int(568*0.8),int(860*0.8)))

# plt.figure()
# plt.title("Image")
# plt.axis("off")
# plt.imshow(image1)
# plt.show()

image1 = cv2.putText(image1,org=(250,250),text="Animal Paint",fontFace=0,fontScale=2.0,thickness=2,color=(0,255,0))

# plt.figure()
# plt.title("Image")
# plt.axis("off")
# plt.imshow(image1)

#burada _ 'i unuttum
#_,image1 = cv2.threshold(image1,thresh=50,maxval=255,type=cv2.THRESH_BINARY)

#image1 = cv2.GaussianBlur(image1,ksize=(51,51),sigmaX=5,sigmaY=5)

#image1 =cv2.Laplacian(image1,ddepth=cv2.CV_16S)

image2 = cv2.calcHist(image1,mask=None,histSize=[256],ranges=[0,255],channels=[0])
plt.figure()
plt.plot(image2)

plt.figure()
plt.axis("off")
plt.imshow(image1)

plt.show()
