import cv2
import matplotlib.pyplot as plt
import numpy as np

#Morfolojik Operasyonlar
#Kavramlar: Erozyon, genişleme, açma kapama ve morfolojik radyan grupları

#1. Erozyon: Toprak kayması gibidir foreground (ön yüzey) deki nesnenin sınırlarını aşındırır | İnceltme
#Kalın fonttaki bir yazının ince fonta kayması gibi :)

#2. Genişleme: Erozyonun tam tersi. Görüntütdeki beyaz bölgenin sınırlarını genişletir.| Kalınlaştırma 

#3. Açma : Erozyon + Genişlemenin art arda kullanımıdır. | Beyaz Nokta | White Noise
# Neden art arda tersi kullanılır diye sorarsanız, eğer resim gürültülü ise bu gürültüyü engellemek için
# açma yöntemi kullanılır.

#4. Kapatma: Genişleme + Erzoyonun art arda kullanımıdır. Açmanın tam tersidir. | Siyah Nokta | Black Noise
# Neden art arda tersi kullanılır diye sorarsak tekrardan, resimin ön planındaki nesnelerin yazı vb. gibi bunların
# üzerindeki siyah beyaz delikleri vb. unsurları yok etmek amaçlı kullanılır. (beyaz yazı üzerindeki parazitlerin kaldırılması gibi)

#5. Morfolojik Gradyan: Görüntünün Genişleme ve Erozyonun arasındaki farktır. Görüntüyü genişlettikten sonra erozyondan çıkarıyorsun.
#Görünüş olarak bir yazının etrafında bir yazı kalıbı var gibi duruyor nedeni ise farkı alınan görüntünün pixellerinden dolayıdır.

#resimi siyah beyaz olarak içe aktardık renk kanalı yok.
image = cv2.imread("assets/datai_team.jpg",0)
plt.figure()
plt.imshow(image,cmap="gray")
plt.axis("off")
plt.title("Original")


#Erozyon:
#5x5 şekilde kutucuk oluştur bu kutucuk 1 lerden oluşuyor ve tipi integer
#(x,y) x büyürse yatay çubuklar hızlı silinir, y büyürse dikey
kernel = np.ones((5,5),dtype=np.uint8)
print(kernel)
#iterations = kaç kez erozyon yapacağı
result = cv2.erode(image,kernel,iterations=1)
plt.figure()
plt.imshow(result,cmap="gray")
plt.axis("off")
plt.title("Erozyon Resmi")


#Genişleme Dilation
#5x5 yukarıdaki kernel i kullandık.
result2 = cv2.dilate(image,kernel,iterations=1)
plt.figure()
plt.imshow(result2,cmap="gray")
plt.axis("off")
plt.title("Genişleme Resmi")


#Açılma => Beyaz Gürültüyü Kapatmak İçin Kullanılır.
#image.shape in row ve column size larını aldı :2 ile
#0 ve 2 derken beyaz = 1 olduğu için 1 üretecek veya 0 (siyah üretecek)
whiteNoise = np.random.randint(0,2,size=image.shape[:2])
whiteNoise = whiteNoise*255 #1lerin hepsi 255 oldu
plt.figure()
plt.imshow(whiteNoise,cmap="gray")
plt.axis("off")
plt.title("White Noise Resmi")

#white noise ile asıl orijinal image i birleştirdik
#başta imread ile okunan değer siyah beyaz olduğu için renk kanalı yok yani 2d array
#bu yüzden bu 2d diziler toplanabilir. Aksi durumda ekstra parametre ekleyecektik white_noise a
noise_img = whiteNoise + image
plt.figure()
plt.imshow(noise_img,cmap="gray")
plt.axis("off")
plt.title("White Noise Resmi")

#Açılma:
opening = cv2.morphologyEx(noise_img.astype(np.float32),cv2.MORPH_OPEN,kernel=kernel)
plt.figure()
plt.imshow(opening,cmap="gray")
plt.axis("off")
plt.title("Opening ile White Noise Duzeltildi")


#Kapatma:
#Black Noise gerekmektedir.
#Black Noise ile White Noise arasındaki tek fark kat sayıyı white da 255 ile çarparken burada -255 ile çarpıyoruz
blackNoise = np.random.randint(0,2,size=image.shape[:2])
blackNoise = blackNoise*-255 #1lerin hepsi -255 oldu
noise_img2 = blackNoise+image
#bu resimde -255 olan değerler 0 a atanır siyah noktalar oluşturulur
#1 ve 0 1 ler -255 ile çarpıldı, ve ana resim ile toplandı.
#sonrsaında blackNoise değelrleri -245den küçük olan değerler yani 1 ile -255 in çarpımı sonucu oluşan değerler beyaz noktaya
#çevirildi.
noise_img2[blackNoise<=-245]=0
plt.figure()
plt.imshow(noise_img2,cmap="gray")
plt.axis("off")
plt.title("Black Noise Resmi")

#astype 32 yapılmasının sebebi hassasiyetin arttırılması gerekmekte nedeni ise şu.
#kernel çerçeveyi gezerken sınırlarda ondalıklı sayı alma ihtimail bulunur.
#belirli piksellerin eşik değeri altına veya üstüne çıkması beklerken integer da veri kaybı olabilir hassas değil
#fark değerleri önemli ve hassas olduğu için float daha avantajlı
closing = cv2.morphologyEx(noise_img2.astype(np.float32),cv2.MORPH_CLOSE,kernel=kernel)
plt.figure()
plt.imshow(closing,cmap="gray")
plt.axis("off")
plt.title("Closing ile Black Noise Duzeltildi")



#Morfolojik Gradient | Gradyan
#Genişleme - Erozyon
#Kenar Tespitnin önemli temel yöntemlerinden biridir.

#burada astype(np.float32) kullanılmıyor gerek yok sonuç sağlıklı
gradient = cv2.morphologyEx(image,cv2.MORPH_GRADIENT,kernel=kernel)
plt.figure()
plt.imshow(gradient,cmap="gray")
plt.axis("off")
plt.title("Morph Gradient")



plt.show()



