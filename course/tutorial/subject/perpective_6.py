import cv2
import numpy as np

#perpektif değiştirme

#read the image
image = cv2.imread("assets/kart.png")
#image width
width = 400
#image height that we set
height = 500


#first we have to declare a point one
#first corner of image
#top left px (203,5)
#top right px (545,150)
#bottom right px (328,624)
#bottom left px (7,475)

#points of treshold image's corners.
points = np.float32([[203,5],[545,150],[328,624],[1,472]])
#points of straight image's corners.
points2 = np.float32([[0,0],[400,0],[400,500],[0,500]])

#we convert our values into matrix by perspective method
matrix = cv2.getPerspectiveTransform(points,points2)
print(matrix)
#After print the latest transformed matrix we had 3 x 3 matrix that includes our values
#[[ 1.41828502e+00  6.13476606e-01 -2.90979243e+02]
# [-3.33802788e-01  7.87314161e-01  6.38253951e+01]
# [ 8.74393358e-04 -2.76836102e-04  1.00000000e+00]]

#now we will use our matrix to create a new image that perspectived
#parameters => (image), (perpectived matrix), (new image's width), (new image's height)
perspectiveImage = cv2.warpPerspective(image,matrix,(width,height))

cv2.imshow("Perspective",perspectiveImage)





cv2.imshow("Original",image)


cv2.waitKey(0)
cv2.destroyAllWindows()