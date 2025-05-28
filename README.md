Raspberry Pi gibi görece düşük güçlü donanımlarda gerçek zamanlı renkli şekil tespiti yapmak için optimize edilmiş, ileri seviye bir Python/OpenCV uygulamasıdır. İşte temel özellikleri ve önemli noktaları:

1. Donanım ve Performans Optimizasyonu
OpenCV optimizasyonları (NEON SIMD gibi) aktif edilir.
CPU performans modu açılır, program bitince eski moda döner (enerji ve hız dengesi için).
İşlem önceliği (nice) ve CPU çekirdeği ataması yapılır (daha hızlı ve stabil çalışsın diye).
HDMI çıkışı kullanılmıyorsa kapatılır (Raspberry Pi’de RAM ve CPU tasarrufu için).
2. Asenkron Video Yakalama (AsyncVideoCapture)
Kamera işlemleri ana iş parçacığından ayrılır, ayrı bir thread’de çalışır.
Küçük bir buffer (kuyruk) ile kareler tutulur, böylece ana program kareleri hızlıca alabilir.
Kamera çözünürlüğü ve FPS’i düşük tutulur (320x240, 30 FPS) — performans için.
Buffer dolarsa en eski kare atılır, yenisi eklenir.
3. Şekil Tespit Sistemi (ShapeDetector)
Renk tabanlı maskeleme: HSV renk uzayında kırmızı ve mavi için ayrı maskeler oluşturulur.
Morfolojik işlemler: Maskelerde küçük boşlukları kapatmak için hızlı dilate işlemi uygulanır.
Gürültü azaltma: Median blur ile hızlı ve etkili gürültü temizliği yapılır.
Kontur ve şekil tespiti: Maskelerden konturlar bulunur, köşe sayısına göre üçgen, kare, altıgen gibi şekiller tespit edilir.
Alan ve oran kontrolleri: Küçük konturlar ve orantısız şekiller elenir.
Etiketleme: Tespit edilen şekillerin üstüne renk ve tip etiketi yazılır.
4. Gerçek Zamanlı FPS Hesabı ve Dinamik Ayar
Her karede işleme süresi ölçülür, son 10 FPS’in ortalaması ekranda gösterilir.
Eğer işleme yavaşlarsa, bazı kareler atlanır (frame skipping) — böylece sistem takılmaz, akıcı kalır.
Eğer işleme hızlanırsa, tekrar her kare işlenmeye başlanır.
5. Kullanıcı Deneyimi
Program başında ve sonunda bilgilendirici mesajlar basılır.
'q' tuşuna basınca güvenli şekilde kapanır, kaynaklar serbest bırakılır.
Hatalı kare alınırsa bekleyip tekrar denenir, program çökmez.
6. Kodun Yapısı ve Modülerlik
Tüm kamera ve şekil tespit işlemleri sınıflara ayrılmıştır.
Kodun her bölümü, Raspberry Pi gibi görece yavaş donanımlarda bile hızlı çalışacak şekilde optimize edilmiştir.
Gereksiz bellek tahsisi ve yavaş işlemler (ör. CLAHE) kullanılmaz.
Kısaca:
Raspberry Pi’de veya benzer donanımlarda, kameradan alınan görüntülerde kırmızı ve mavi renkli üçgen, kare, altıgen gibi şekilleri hızlı ve verimli şekilde tespit edip ekranda gösterir. Donanımın sınırlarını zorlamadan, akıcı ve kararlı bir şekilde çalışacak şekilde tasarlanmıştır. Kodun her aşamasında performans ve kararlılık ön plandadır.
