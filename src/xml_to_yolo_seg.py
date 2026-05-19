import os
import xml.etree.ElementTree as ET
import cv2

# Ana dizinler
RAW_IMAGES_DIR = "data/raw/images/"
RAW_XML_DIR = "data/raw/xml/"
LABELS_DIR = "data/intermediate/labels/"

CLASS_MAPPING = {
    "baklava": 0, 
    "hamburger": 1, 
    "kebap": 2, 
    "köfte": 3,
    "lahmacun": 4, 
    "patates_kizartmasi": 5, 
    "pizza": 6,
    "sosisli": 7, 
    "sutlac": 8, 
    "waffle": 9
}

if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

# Her bir  yemek sınıfı (klasörü) için sırayla işlem yap
for class_name in os.listdir(RAW_IMAGES_DIR):
    img_class_dir = os.path.join(RAW_IMAGES_DIR, class_name)
    xml_class_dir = os.path.join(RAW_XML_DIR, class_name)
    
    # Eğer o yemeğe ait XML klasörü yoksa atla
    if not os.path.isdir(img_class_dir) or not os.path.exists(xml_class_dir):
        continue
        
    print(f"\n--- {class_name.upper()} İşleniyor ---")
    
    # 1. Resimleri al ve leksikografik sırala
    image_files = [f for f in os.listdir(img_class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    print(f"{len(image_files)} adet görsel bulundu.")
    
    # 2. O sınıfa ait XML'leri bul ve en küçük Start Frame'i otomatik tespit et
    xml_files = [f for f in os.listdir(xml_class_dir) if f.endswith('.xml')]
    min_start_frame = float('inf')
    
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(xml_class_dir, xml_file))
        root = tree.getroot()
        # CVAT XML içindeki start_frame etiketini bulur
        start_frame_tag = root.find('.//meta/job/start_frame')
        if start_frame_tag is not None:
            min_start_frame = min(min_start_frame, int(start_frame_tag.text))
            
    if min_start_frame == float('inf'):
        print(f"Uyarı: {class_name} için start_frame bulunamadı, 0 varsayılıyor.")
        min_start_frame = 0
        
    print(f"Başlangıç Frame'i otomatik tespit edildi: {min_start_frame}")
    
    # 3. Frame - Resim Eşleştirmesi (Harita oluştur)
    frame_to_image = {}
    for i, img_name in enumerate(image_files):
        frame_to_image[str(int(min_start_frame) + i)] = img_name

    # 4. XML'lerin içindeki poligonları okuyup TXT'leri oluştur
    image_dimensions = {}
    
    for xml_file in xml_files:
        tree = ET.parse(os.path.join(xml_class_dir, xml_file))
        root = tree.getroot()
        
        for track in root.findall('track'):
            label_name = track.get('label')
            class_id = CLASS_MAPPING.get(label_name)
            
            if class_id is None:
                continue
                
            for polygon in track.findall('polygon'):
                frame = polygon.get('frame')
                points = polygon.get('points')
                
                img_name = frame_to_image.get(frame)
                if not img_name:
                    continue
                    
                # Boyutları dinamik öğren
                if img_name not in image_dimensions:
                    img_path = os.path.join(img_class_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    image_dimensions[img_name] = img.shape[:2]
                
                img_height, img_width = image_dimensions[img_name]
                
                # Normalize et
                yolo_points = []
                point_pairs = points.split(';')
                for pair in point_pairs:
                    x_str, y_str = pair.split(',')
                    x = max(0.0, min(1.0, float(x_str) / img_width))
                    y = max(0.0, min(1.0, float(y_str) / img_height))
                    yolo_points.extend([f"{x:.6f}", f"{y:.6f}"])
                    
                yolo_line = f"{class_id} " + " ".join(yolo_points)
                
                # TXT dosyasını LABELS_DIR içine kaydet
                txt_filepath = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")
                with open(txt_filepath, 'a', encoding='utf-8') as f:
                    f.write(yolo_line + "\n")

print("\nTÜM İŞLEMLER BAŞARIYLA TAMAMLANDI! TXT dosyaları 'labels' klasöründe.")