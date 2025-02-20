import cv2
import matplotlib.pyplot as plt


#Gradients = Gradyanlar.
#Görüntüdeki yoğunluk veya renkteki yönlü değişiklikleri temsil eder.
#KENAR Algılamada kullanılır.

#siyah beyaz aktardık
image = cv2.imread("assets/sudoku.jpg",0)
plt.figure()
plt.imshow(image,cmap="gray")
plt.axis("off")
plt.title("Original")


#X eksenindeki gradyanları bulma
#parametreler = (image), derinlik, dx, dy, kernelSize
#x yönünde yaptığımız için dx=1 verdik
#Dik olan kenarları tespit ettik

#16SC1 = signed 16 bit integer Channel 1 (Gray Scale)
#16SC3 = signed 16 bit integer CHannel 3 (RGB)

sobelX = cv2.Sobel(image,ddepth=cv2.CV_16S,dx=1,dy=0,ksize=5)
plt.figure()
plt.imshow(sobelX,cmap="gray")
plt.axis("off")
plt.title("Sobel X")

#Yatay olan kenarları tespit ettik
sobelY = cv2.Sobel(image,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=5)
plt.figure()
plt.imshow(sobelY,cmap="gray")
plt.axis("off")
plt.title("Sobel Y")

#her ikisini de tespit etmek için laplaction gradient i çağırabiliriz.  

laplacian = cv2.Laplacian(image,ddepth=cv2.CV_16S)
plt.figure()
plt.imshow(laplacian,cmap="gray")
plt.axis("off")
plt.title("Sobel XY")





plt.show()