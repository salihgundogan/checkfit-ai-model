import os
import xml.etree.ElementTree as ET
import cv2

# --- YENİ KLASÖR YAPISI AYARLARI ---
# Scriptin ana dizinde olduğunu varsayarak yolları buna göre veriyoruz
XML_DIR = "ham_veriler/xml_dosyalari/"
IMAGES_DIR = "ham_veriler/gorseller/patates_kizartmasi/"
LABELS_DIR = "ham_veriler/labels/patates_kizartmasi/"  # TXT'lerin kaydedileceği yer
START_FRAME = 3000

CLASS_MAPPING = {
    "patates_kizartmasi": 0  
}
# --------------------------------------------------------

# Etiketlerin kaydedileceği klasör yoksa otomatik oluştur
if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

# 1. Resimleri klasörden al ve Leksikografik (1, 10, 100...) sırala
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort() 

print(f"Toplam {len(image_files)} görsel bulundu ve sıralandı.")

# 2. Frame ID ile Resim isimlerini eşleştir
frame_to_image = {}
for i, img_name in enumerate(image_files):
    current_frame = START_FRAME + i
    frame_to_image[str(current_frame)] = img_name

# 3. XML Dosyalarını Oku ve Ayrıştır (Klasördeki tüm XML'leri otomatik bulur)
image_annotations = {img_name: [] for img_name in image_files}
xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]

# Performans için: Görsel boyutlarını hafızada tutalım
image_dimensions = {}

for xml_file in xml_files:
    xml_path = os.path.join(XML_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for track in root.findall('track'):
        label_name = track.get('label')
        
        # SADECE patates_kizartmasi etiketlerini al
        if label_name != "patates_kizartmasi":
            continue
            
        class_id = CLASS_MAPPING.get(label_name)
        
        for polygon in track.findall('polygon'):
            frame = polygon.get('frame')
            points = polygon.get('points')
            
            img_name = frame_to_image.get(frame)
            if not img_name:
                continue
                
            # --- DİNAMİK BOYUT HESAPLAMA (Önbellekli) ---
            if img_name not in image_dimensions:
                img_path = os.path.join(IMAGES_DIR, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Uyarı: {img_name} okunamadı, atlanıyor.")
                    continue
                image_dimensions[img_name] = img.shape[:2] # (height, width)
            
            img_height, img_width = image_dimensions[img_name]
            
            # Noktaları normalize et
            yolo_points = []
            point_pairs = points.split(';')
            for pair in point_pairs:
                x_str, y_str = pair.split(',')
                x = float(x_str) / img_width
                y = float(y_str) / img_height
                
                # Sınırların dışına taşmayı engelle
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                
                yolo_points.extend([f"{x:.6f}", f"{y:.6f}"])
                
            yolo_line = f"{class_id} " + " ".join(yolo_points)
            image_annotations[img_name].append(yolo_line)

# 4. TXT dosyalarını oluştur ve kaydet
for img_name, annotations in image_annotations.items():
    txt_filename = os.path.splitext(img_name)[0] + ".txt"
    txt_filepath = os.path.join(LABELS_DIR, txt_filename)
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(ann + "\n")

print(f"İşlem tamam! {len(xml_files)} adet XML dosyası tarandı ve TXT dosyaları '{LABELS_DIR}' klasörüne kaydedildi.")