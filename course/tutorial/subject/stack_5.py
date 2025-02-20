import cv2
import numpy as np
image = cv2.imread("assets/background.jpg")
small = cv2.resize(image,(600,600))
cv2.imshow("Original",small)

#numpy horizontal stack
#yan yana ekliyor arrayleri
horizontal = np.hstack((image,image))
#iki resmin birleşmiş hali
cv2.imshow("Original",horizontal)


#üst üste ekliyor arrayleri
vertical = np.vstack((image,image))
cv2.imshow("Original",vertical),
#iki resmin birleşmiş hali




# Wait for a key press
cv2.waitKey(0)  # Sonsuza kadar bekler, bir tuşa basılmasını bekler
cv2.destroyAllWindows()  # Tüm pencereleri kapatır