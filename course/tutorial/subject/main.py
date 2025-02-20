import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Örnek bir rastgele matris oluştur
data = np.random.random((200, 200))

# Renkleri tanımla: Beyaz (1, 1, 1) ve Kırmızı (1, 0, 0)
colors = [(1, 1, 1), (1, 0, 0)]  # Beyaz ve Kırmızı

# LinearSegmentedColormap kullanarak renk haritası oluştur
custom_cmap = LinearSegmentedColormap.from_list('white_red', colors)

# Görüntüyü özel renk haritasıyla göster
plt.figure(figsize=(7,7))
plt.imshow(data, cmap=custom_cmap,)
print(data)
plt.colorbar()  # Renk çubuğunu ekle
plt.title("Beyazdan Kırmızıya Geçiş")
plt.show()