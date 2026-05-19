import os
import shutil
import random
import yaml

RAW_IMAGES_DIR = "data/raw/images/"
LBL_DIR = "data/intermediate/labels/"
BASE_DIR = "data/processed/"

# GELECEĞE YATIRIM: YOLO'nun indeks hatası vermemesi için tüm 10 sınıfı haritaya ekliyoruz.
CLASS_NAMES = {
    0: "baklava",
    1: "hamburger",
    2: "kebap", 
    3: "köfte",
    4: "lahmacun", 
    5: "patates_kizartmasi", 
    6: "pizza",
    7: "sosisli",
    8: "sutlac",
    9: "waffle"
}

TRAIN_RATIO = 0.8

# Hedef klasörleri temizle ve oluştur
for split in ['train', 'val']:
    img_split_dir = os.path.join(BASE_DIR, 'images', split)
    lbl_split_dir = os.path.join(BASE_DIR, 'labels', split)
    
    # Eğer klasörler varsa önce içini temizle (eski verilerle karışmaması için)
    if os.path.exists(img_split_dir):
        shutil.rmtree(img_split_dir)
    if os.path.exists(lbl_split_dir):
        shutil.rmtree(lbl_split_dir)
        
    os.makedirs(img_split_dir, exist_ok=True)
    os.makedirs(lbl_split_dir, exist_ok=True)

valid_data = []

for class_name in os.listdir(RAW_IMAGES_DIR):
    img_class_dir = os.path.join(RAW_IMAGES_DIR, class_name)
    if not os.path.isdir(img_class_dir):
        continue
        
    for img_file in os.listdir(img_class_dir):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + ".txt"
        
        if os.path.exists(os.path.join(LBL_DIR, txt_file)):
            valid_data.append((img_class_dir, img_file, txt_file))

print(f"Toplam eşleşen veri: {len(valid_data)} adet")

random.seed(42)
random.shuffle(valid_data)

split_idx = int(len(valid_data) * TRAIN_RATIO)
train_data = valid_data[:split_idx]
val_data = valid_data[split_idx:]

def copy_files(data_list, split_name):
    for img_class_dir, img_file, txt_file in data_list:
        src_img = os.path.join(img_class_dir, img_file)
        src_txt = os.path.join(LBL_DIR, txt_file)
        
        dst_img = os.path.join(BASE_DIR, 'images', split_name, img_file)
        dst_txt = os.path.join(BASE_DIR, 'labels', split_name, txt_file)
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_txt, dst_txt)

print("Dosyalar Train ve Val klasörlerine kopyalanıyor...")
copy_files(train_data, 'train')
copy_files(val_data, 'val')

abs_base_dir = os.path.abspath(BASE_DIR)
yaml_content = {
    'path': abs_base_dir,
    'train': 'images/train',
    'val': 'images/val',
    'names': CLASS_NAMES
}

yaml_path = os.path.join(BASE_DIR, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

print(f"\nIslem Basariyla Tamamlandi! data.yaml dosyasi 10 sinif formatinda guncellendi.")