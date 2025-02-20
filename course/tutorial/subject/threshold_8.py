#Python 3.9.6 (venv) enviorment inpreter

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
#Görüntüleri Eşikleme
#aslında eşiklemeden kasıt şu. Tüm pixellerin renk değeri aralığı 0-255 arasında
#burada diyelimki eşik değeri 125 ve siyah beyaz bir resime sahipiz.
#çıktı olan resimde renk değeri 125 in altındaki değerler beyaz olarak gözükecek ve kalan diğer değerler
#kendi renginde çıkacak.
#burada resimdeki ek detayları kaldırarak ana hatları barizleştirmiş olursun.

image = cv2.imread("assets/background.jpg")
#cvtColor = convert Color
#COLOR_BGR2GRAY = Color blue green red to gray
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#resim tam siyah beyaz olmadı burada renk tonları gri skalada oldu o yüzden hafiften yeşil ve morumsu
#renk skalası gri olarak ayarlandı

plt.figure()

#cmap = color map | renk haritası gri olarak ayarlandı
plt.imshow(image,cmap="gray")

#ek axis x ve y yi kapatır
plt.axis("off")

#treshold eşik ayarlama
#iki tip return ediyor 1.si işime yaramadığı için underscore(_) ile atadık
#treshold parametreleri = (image), min eşik = 60, max değer = 255, tip = Thresh Binary
#max değeri düşürürsen maxtan büyük değerler direkt 255 beyaz olarak gözükür
#thresh binary = max ve min değerlerin arasında değerlerin açıp kapanması için bir değer
#bunun farklı tipleri de tam tersini uygulayabiliyor şimdilik işimize gelen çevirme tipi THRESH_BINARY
#THRESH_BINARY_INV tipi de eşik çevirmeyi ters olarak ayarlar.
_,tresholdImage = cv2.threshold(image,thresh=30,maxval=255,type=cv2.THRESH_BINARY)

plt.figure()
plt.imshow(tresholdImage,cmap="gray")
plt.axis("off")

#adaptive treshold
#ışık farklılıklarından dolayı istemediğimiz detaylar kalabilir.
#kaldırılmasını istedğimiz bir detayın belli bir kısmını kaldıramadığımız durumlar olabilir.
#işte bu durum için biz de adaptive treshold methodunu kullanacağız.

#mantık global bir threshold değeri atamaktansa belli parçaların treshold değerini değiştirmektir.
#bu sayede belli alanların komşu pixellerin farklı eşik değerleri ile o bölgeleri gizleyebiliriz veya gösterebiliriz.

#parametreler = (image), max value, adaptive treshold type, treshold type, block boyutu, C sayısı
# C sayısı 

thresh_image_adaptive = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,8)
#bloksize yani 11 değerimizi verdiğimiz alan görüntüyü 11x11 lik değerlere böler.
#her parça için gerekli eşik değeri bulunur bunun da methodu:
#cv2.ADAPTIVE_THRESH_MEAN_C toplu pixellerin ortalama renk değerini alır
#ile yapılır.
#sonrasında bu değerden C sayısını yani 8 i çıkartıp son bir renk değeri elde ederiz.
#buradaki C nin amacı parlak bölgelerde ışık daha fazla vurduğu için detaylandırmayı kontrol etmek amacı ile yapılır.
#Ek Not: blockSize = 11 değerinde blockSize tek sayı olmalıdır. ex: 3, 5, 7, 11, 13

plt.figure()
plt.imshow(thresh_image_adaptive,cmap="gray")
plt.axis("off")
plt.show()




