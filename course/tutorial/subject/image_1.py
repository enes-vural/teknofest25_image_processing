import cv2
import matplotlib.pyplot as plt
#computer vision
basePath = "/Users/sakastudio/development-py/course/"
imgPath = basePath+ "assets/background.jpg"
writePath = basePath+"assets/generated/gray-image.jpg"

#image read anlamına gelir ayrıca 0 ifadesi gray-scale yani resmin siyah ve beyaz olarak fetch etmesini sağlar
image = cv2.imread(imgPath,0)

total_pixel = image.size

# cv2.waitKey(0) detects keyboard actions of user

#imaginization 
if image is None:
    print("Image has occurred with error")
else:
    # Resmi göster
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Eksenleri kapat
    plt.show()

    cv2.imwrite(writePath,image)
