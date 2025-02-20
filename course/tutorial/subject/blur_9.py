import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings
#Gereksiz uyarıları kaldırır
#warnings.filterwarnings("ignore")

#Görüntü Bulanıklaştırma
#Görüntü bulanıklaştırma aslında görüntünün düşük geçişli filtrelemedir.
#Görüntüdeki gürültüleri (parazit, kenar) tarzında gürültüleri kaldırır

#3 tane bulanıklaştırma yöntemi vardır bunlar:
# 1. Ortalama Bulanıklaştırma => görüntüyü kutu gibi düşün. Kutucuğu sol üst köşeden başladığnı düşün. 3x3 olsun. 
# Bu kutucuk 3x3 şekilde döne döne üzerine geldiği pixellerin renk ortalamasını alır ve bölgenin merkezine renk olarak koyar

#2. GAUSS => Kutu mantığı ile aynıdır. Burada kutu yerine Gauss çekirdeği denir ve biz bu çekirdeğin.
# yükseklik, genişlik, sigma X ve sigma Y değerlerini veririrz. Bu çekirdek ortalama renk değerini almak yerine
# belirlediğimiz değerlere göre işlemler blurlama gerçekleştirilir

#3. Medyan => Ortalama Bulanıklaştırma ile neredeyse aynı tek farkı ortalama değeri almak yerine içindeki pixellerin
# medyan değerini alır ve bu değer ile merkezi pixelin yerine boyar.
# Tuz biber gürültüsüne karşı etkilidir (siyah beyaz noktacıklar)

image = cv2.imread("assets/background.jpg")
#Convert BGR Color to RGB
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#--------------#
#Original Image
plt.figure()
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
#--------------#

#--------------#
#Ortalama Bulanıklaştırma

#Blur çıktılarına dst olarak adlandırılır.
#Tahminimce açıklaması destination => hedef
averageBlr = cv2.blur(image,ksize=(3,3))
plt.figure()
plt.imshow(averageBlr)
plt.axis("off")
plt.title("Average Blur")
#--------------#

#--------------#
#Gauss Bulanıklaştırma
#sigma Y boş bırakılırsa X e ile aynı olur.
gaussianBlr = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=5,sigmaY=5)
plt.figure()
plt.imshow(gaussianBlr)
plt.axis("off")
plt.title("Gaussian Blur")
#--------------#


#--------------#
#[3][3] gezdiği yerlerdeki 7x7 arrayleri alır
#Medyan değerini alır
#Not: Bu değerlere tek sayı vermeye dikkat et
medianBlur = cv2.medianBlur(image,ksize=7)
plt.figure()
plt.imshow(medianBlur)
plt.axis("off")
plt.title("Median Blur")
#--------------#


def gaussianNoise(img):
    #channel renk kodudur RGB ise channel = 3
    row,column,channel = img.shape
    mean = 0 #gürültünün ortalama değeri = 0
    var = 0.05 #varyans = gürültünün yoğunluğu
    sigma = var**0.5 # sigma = varyans'ın karekökü 
    #sigma büyüdükçe gürültü dağılımının genişliği ve yayılımı artar
    #sigma aslında standart sapmadır.

    #gaussian in diger ismi normal dağılımdır
    #np.random.normal => normal (gaussian) dağılıma sahip olacak rastgele sayılar oluşturur
    #mean = gürültünün merkezi
    #sigma = standart sapma
    #(row, column, channel) → Üretilen gürültünün boyutları (görüntüyle aynı).
    gaussian = np.random.normal(mean,sigma,(row,column,channel))
    #tekrardan ek kontrol amaçlı görüntü kendi boyutları ile tekrar şekillendirilir
    gaussian = gaussian.reshape(row,column,channel)
    #elde edilen görüntü ile eski görüntü toplanarak, eski gürültünün pixel değerleri arttırılır.
    #bu da bozulmuş bir görüntü elde etmemizi sağlar

    #Matematiksel Denklem:
    #Noisy Image=Original Image+G(μ,σ)
    # Parazitli Görüntü = Orijinal Resim + G(μ,σ)
    # G => Gaussian Dağılımı
    # μ => Ortalama (mean)
    # μ = 0 ise gürültü negatif ve pozitif olabilir. μ > 0 ise gürültü pozitife daha yakındır bu yüzden görüntü daha parlak olabilir. μ < 0 ise gürültü negatife daha yakındır bu yüzden görüntü daha karanlık olabilir.
    # genel de μ = 0 kullanılır daha rastgele bir bozulma meydana getirilir.
    #Gaussian Çan Eğirisinin Formülü = (1 / √2*pi*σ^2) * e^[-(x-μ)/2σ^2]
    # σ => Standart Sapma (sigma)

    noisyImage = img+gaussian
    return noisyImage

#Gaussian Noise
#--------------
image2 = cv2.imread("assets/background.jpg")
#burada gaussian noise çalışması için image in color değerlerini /255 e bölerek 0-1 arasında tutmanız gerekmektedir.
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)/255

gaussianNoiseImage = gaussianNoise(image2)
plt.figure()
plt.imshow(gaussianNoiseImage)
plt.title("Gaussian Noised Image")
plt.axis("off")
#--------------
#Gaussain Paraziti eklenen resim tekrardan Gaussian Blur ile hatalar azaltıldı
guassian2Normal = cv2.GaussianBlur(gaussianNoiseImage,ksize=(3,3),sigmaX=5,sigmaY=5)
plt.figure()
plt.imshow(guassian2Normal)
plt.axis("off")
plt.title("Gaussian Blur from Gausian Noise")
#--------------#




#Tuz Karabiber Gürültüsü Oluşturma
#Resim üzerine siyah beyaz noktaların rastgele oluşturulması

def saltPepperNoise(img):
    row,column,channel = img.shape
    s_vs_p = 0.5 # salt ve pepper (siyah beyaz) noktaların birbirine göre oranı burada %50 verildi
    amount = 0.04
    #görüntüyü bozmadan noisyImage e atadık
    noisyImage = np.copy(img)
    
    #salt => beyaz noktacıklar
    #ceil ondalıklı sayıyı yukarı aşşağı yuvarlar
    #Formül = Miktar * Resim Boyutu * Siyah ve Beyaz ın oranı
    number_salt = int(np.ceil(amount * img.size * s_vs_p))
    #tuzların ekleneceği kordinatlar belirtilir.
    #kordinatlar şu şekilde belirlenir:
    #image.shape = resim boyutundaki arraylerde
    #0 dan başlayarak son indexe kadar rastgele bir sayı türetilir
    #bu türetme number_salt yani tuz sayısı kadar kordinat üretilir.
    cords = [np.random.randint(0,i ,number_salt)  for i in img.shape]
    #image içindeki kordinatlara beyaz (1) eklenir.
    #cords u tuple, demet'e çevirdik.
    #çoklu dizi kullandığımız için [row,column,channel]
    #bu üçlü veriye erişmek için bunları tuple da toplamamız gerekiyor.
    #bunuda casting tuple() ifadesi ile yaptık
    noisyImage[tuple(cords)] = 1

    

    #pepper => siyah noktacıklar
    #Formül = Miktar * Resim Boyutu * Siyah ve Beyaz ın oranı
    #(1-s_vs_p) | tuzlar 0.03 ise pepperler 0.07 olması için
    number_pepper = int(np.ceil(amount * img.size * (1-s_vs_p)))
    #pepper ekleneceği kordinatlar belirtilir.
    #kordinatlar şu şekilde belirlenir:
    #image.shape = resim boyutundaki arraylerde
    #0 dan başlayarak son indexe kadar rastgele bir sayı türetilir
    #bu türetme number_salt yani tuz sayısı kadar kordinat üretilir.
    cords2 = [np.random.randint(0,i,number_pepper)  for i in img.shape]
    #image içindeki kordinatlara siyah (0) eklenir.
    noisyImage[tuple(cords2)] = 0
    #cords2 yi tuple a çeviridk


    return noisyImage

#Salt Pepper Noise
#--------------#
image3 = cv2.imread("assets/background.jpg")
image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2RGB)
saltPepperImage = saltPepperNoise(image3)
plt.figure()
plt.imshow(saltPepperImage)
plt.title("Salt Pepper Noised Image")
plt.axis("off")
#--------------
#Medyan Düzeltmesi ile Tuz Karabiber Blurlarındaki hatalar azaltıldı
noiseMedianBlur = cv2.medianBlur(image3,ksize=7)
plt.figure()
plt.imshow(noiseMedianBlur)
plt.axis("off")
plt.title("Median Blur from Salt Pepper Noise")
#--------------#




plt.show()