import cv2
import matplotlib.pyplot as plt
import numpy as np

#Histogram
#Görüntü Histogramı görüntüdeki ton dağılımının grafiksel bir temsilidir.
#Her bir ton değer için pixel sayısı içerir
#Belirli bir görüntü için histograma bakılarak ton dağılımı anlaşılabilir


image = cv2.imread("assets/red_blue.jpg")
image_vis = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image_vis)
plt.axis("off")
plt.show()

print(image_vis.shape)

#hist size = 256, max değer 255 renk kodunda o yüzden
#range de aynı nedenden dolayı 0 - 255

image_hist = cv2.calcHist([image],channels=[0],mask=None,histSize=[256],ranges=[0,256])

#renk ayrımsız total 255 ve 0 olan pixelleri gösterir
print(image_hist.shape)
plt.figure()
plt.plot(image_hist)

color = ("b","g","r")
#enumerate = color ı alıyor color içine b nin indexini i ye eşitleyip stringini de c ye eşitler
for i,c in enumerate(color):
    new_hist = cv2.calcHist([image],channels=[i],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(new_hist,color=c)

#Maskeleme ile resmin küçük bir bölümüne odaklanacağız

golden_gate = cv2.imread("assets/goldenGate.jpg")
golden_gate = cv2.cvtColor(golden_gate,cv2.COLOR_BGR2RGB)
print(golden_gate.shape)

#maske oluşturduk siyahlardan ve resmimizin boyutları kadar pixel büyüklüğünde
mask = np.zeros(golden_gate.shape[:2],np.uint8)
#maskenin üstüne delik açıp ana resmi maskenin altına ekleyeceğiz, geri kalan her yer siyah olacak
#maskenin x ekseninde 1500den 2000e kadar, 
#maseknin y ekseninde 1000den 2000e kadar olan pixellerini beyaz olarak ayarladık
mask[1500:2000,1000:2000]=255

#bitwise and operatörü
#bitwise and operatörü ile maskeleme gerçekleşti
#not kullanırsan renkler galiba tam tersine çıkıyor
mask_img_vis = cv2.bitwise_and(golden_gate,golden_gate,mask=mask)

#channel = 0 kırmızıya karşılık geliyor 
#channel = 1 yeşil
#channel = 2 mavi
#RGB
hist_masked = cv2.calcHist([golden_gate],channels=[0],mask=mask,histSize=[256],ranges=[0,255])
plt.figure()
plt.title("Histogram")
plt.plot(hist_masked)

#Histogram Eşitleme
#karşıtlığı arttırıyor.

img2 = cv2.imread("assets/hist_equ.jpg",0)
plt.figure()
plt.imshow(img2,cmap="gray")

img_hist2 = cv2.calcHist([img2],channels=[0],mask=None,histSize=[256],ranges=[0,255])
plt.figure()
plt.plot(img_hist2)

#konstrantı arttırıyor
#155 ler daha fazla 255 e yaklaşırken
#50 60lar daha fazla 0 a yaklaşıyor
eq_img = cv2.equalizeHist(img2)
plt.figure()
plt.imshow(eq_img,cmap="gray")

eq_hist = cv2.calcHist([eq_img],channels=[0],mask=None,histSize=[256],ranges=[0,255])
plt.figure()
plt.plot(eq_hist)


plt.show()