import cv2
import matplotlib.pyplot as plt


#Görüntüleri Karıştırma
#üst üste hafiften soluk bir şekilde koyma :)

img1 = cv2.imread("assets/background.jpg")
firstImage = cv2.resize(img1,(200,200))

img2 = cv2.imread("assets/kart.png")
secondImage = cv2.resize(img2,(200,200))

#openCV normalde BGR desteklediği için
#pyplot ta RGB destekliyor otomatik olarak.
#bu yüzden burada BGR'ı RGB ye çeviriyoruz.
# OpenCV'den gelen görüntüleri RGB formatına dönüştür
firstImage = cv2.cvtColor(firstImage, cv2.COLOR_BGR2RGB)
secondImage = cv2.cvtColor(secondImage, cv2.COLOR_BGR2RGB)

#create
plt.figure()
plt.imshow(firstImage)

plt.figure()
plt.imshow(secondImage)

# karıştırma için denklem
# karıştırılmış resim = alpha * (image1) + betha * (image2)
# output=(src1×α)+(src2×β)+γ

#buradaki alpha ve betaları değiştirerek denkleme göre hangi resimin daha baskın olduğunu ayarlayabilirsin.
#buradaki gama işareti çıktı resmin parlaklığını ayarlamak için kullanılır.
blended = cv2.addWeighted(src1=firstImage,alpha=0.5,src2=secondImage,beta=0.5,gamma=0)
plt.figure()
plt.imshow(blended)


#grafikleri çıkarmak için
plt.show()
