import os
from roboflow import Roboflow

# Roboflow API anahtarınızı buraya ekleyin
API_KEY = "qhKwvgJrKGQCEjX7Ozvd"

# Roboflow nesnesini başlatıyoruz
rf = Roboflow(api_key=API_KEY)

# Workspace ve Project ID bilgilerini alın
workspace_id = 'firtinateknofest'  # Buraya workspace ID'nizi girin
project_id = 'firtina'      # Buraya proje ID'nizi girin

# Proje nesnesini alıyoruz
project = rf.workspace(workspace_id).project(project_id)

# Görsellerin bulunduğu klasör yolu
folder_path = r"output\hexagon" 

# Klasördeki tüm görselleri yüklemek için
for image_filename in os.listdir(folder_path):
    # Yalnızca .jpg, .jpeg, .png uzantılı dosyaları yükleyelim
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, image_filename)  # Görselin tam dosya yolunu alıyoruz
        
        try:
            print(f"{image_filename} yükleniyor...")
            
            # Görseli Roboflow'a yükleyin
            response = project.upload(
                image_path=image_path,          # Yüklemek istediğiniz görselin yolu
                batch_name="hexagon_batch",     # İsteğe bağlı: Görselin ait olduğu batch adı
                split="train",                  # İsteğe bağlı: Görselin ait olduğu split (train/valid/test)
                num_retry_uploads=3,           # Yükleme hatası durumunda deneme sayısı
                tag_names=["Hexagon"],     # İsteğe bağlı: Görsele etiket eklemek
                sequence_number=1,             # İsteğe bağlı: Görsel sırası
                sequence_size=100              # İsteğe bağlı: Görsel sırası toplamı
            )
            
            # Yükleme başarılı olduysa yanıtı yazdırıyoruz
            print(f"{image_filename} başarıyla yüklendi:", response)
        
        except Exception as e:
            print(f"{image_filename} yüklenirken hata oluştu: {e}")
