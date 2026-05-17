import os
import shutil
import random
import yaml

# --- AYARLAR ---
# Kaynak klasörlerimiz
IMG_DIR = "data/raw/images/"
LBL_DIR = "data/intermediate/labels/"

# Hedef YOLO eğitim klasörü
BASE_DIR = "data/processed/"

# Şimdilik tek sınıfımız var
CLASS_NAMES = {0: "patates_kizartmasi"}

# Eğitim yüzdesi
TRAIN_RATIO = 0.8  # %80 Train, %20 Val
# --------------------------------------------------------

# 1. YOLO için gerekli klasör yapısını oluştur
for split in ['train', 'val']:
    os.makedirs(os.path.join(BASE_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'labels', split), exist_ok=True)

# 2. Resim ve TXT eşleşmelerini bul
valid_data = []
for img_file in os.listdir(IMG_DIR):
    if not img_file.endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    # Resim adından txt adını bul
    base_name = os.path.splitext(img_file)[0]
    txt_file = base_name + ".txt"
    
    # Eğer bu resmin txt dosyası labels içinde varsa listeye ekle
    if os.path.exists(os.path.join(LBL_DIR, txt_file)):
        valid_data.append((img_file, txt_file))

print(f"Toplam eşleşen veri: {len(valid_data)} adet")

# 3. Verileri karıştır (Modelin farklı resimleri görmesi için önemli)
random.seed(42) # Her çalıştırdığında aynı karışımı yapması için sabit bir seed
random.shuffle(valid_data)

# 4. Verileri böl
split_idx = int(len(valid_data) * TRAIN_RATIO)
train_data = valid_data[:split_idx]
val_data = valid_data[split_idx:]

# 5. Dosyaları kopyalayan yardımcı fonksiyon
def copy_files(data_list, split_name):
    for img_file, txt_file in data_list:
        # Resimleri kopyala
        shutil.copy(
            os.path.join(IMG_DIR, img_file),
            os.path.join(BASE_DIR, 'images', split_name, img_file)
        )
        # TXT'leri kopyala
        shutil.copy(
            os.path.join(LBL_DIR, txt_file),
            os.path.join(BASE_DIR, 'labels', split_name, txt_file)
        )

# Kopyalama işlemini başlat
print("Dosyalar Train ve Val klasörlerine kopyalanıyor...")
copy_files(train_data, 'train')
copy_files(val_data, 'val')

# 6. data.yaml dosyasını oluştur
# Klasörün tam yolunu alıyoruz (YOLO bazen tam yol ister)
abs_base_dir = os.path.abspath(BASE_DIR)

yaml_content = {
    'path': abs_base_dir,
    'train': 'images/train',
    'val': 'images/val',
    'names': CLASS_NAMES
}

yaml_path = os.path.join(BASE_DIR, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

print(f"\nİşlem Tamamlandı!")
print(f"Eğitim (Train): {len(train_data)} fotoğraf")
print(f"Doğrulama (Val): {len(val_data)} fotoğraf")
print(f"YOLO konfigürasyon dosyası '{yaml_path}' konumunda oluşturuldu.")
print("Artık eğitime başlamaya hazırsın!")