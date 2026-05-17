import cv2
from ultralytics import YOLO

# 1. Eğittiğin taze modeli yüklüyoruz
# (5 epoch eğittiğin için runs/segment/patates_modeli/weights/ klasöründedir)
model = YOLO('runs/segment/runs/patates_modeli/weights/best.pt')

# 2. Test etmek istediğin bir patates fotoğrafının yolunu yaz
TEST_IMAGE = "ham_veriler/gorseller/patates_kizartmasi/6000 (28).jpg" # Klasöründeki gerçek bir resim adıyla değiştir!
# TEST_IMAGE = "kalibrasyon_kagidi2.jpg" # Klasöründeki gerçek bir resim adıyla değiştir!

# 3. Modelden tahmin iste
results = model.predict(source=TEST_IMAGE, conf=0.25)

# --- MATEMATİKSEL KÖPRÜ (Piksel -> cm² Dönüşüm Oranı) ---
# Şimdilik farazi bir oran veriyoruz: Örneğin her 2000 piksel = 1 cm² olsun.
# Gerçek projede bu oranı tabak boyutuna göre kalibre edeceğiz.
PIXEL_TO_CM2_RATIO = 410.71

total_pixels = 0

# Model bir nesne bulabildiyse ve maske ürettiyse
if results[0].masks is not None:
    # Modelin bulduğu tüm patates poligonlarını (maskelerini) piksel cinsinden alıyoruz
    masks_in_pixels = results[0].masks.xy  
    
    for mask in masks_in_pixels:
        if len(mask) > 0:
            # OpenCV'nin Contour Area fonksiyonu poligonun piksel alanını hesaplar
            area_in_pixels = cv2.contourArea(mask)
            total_pixels += area_in_pixels

    # Pikselleri gerçek cm² değerine bölüyoruz
    calculated_cm2 = total_pixels / PIXEL_TO_CM2_RATIO

    print("\n" + "="*40)
    print(f"📊 ANALİZ SONUCU:")
    print(f"Patateslerin Kapladığı Piksel Alanı: {total_pixels:.2f} piksel")
    print(f"Hesaplanan Gerçek Alan: {calculated_cm2:.2f} cm²")
    print(f"👉 SONUÇ CÜMLESİ: Bu yemekte {calculated_cm2:.1f} cm² patates kızartması vardır.")
    print("="*40 + "\n")

else:
    print("Model bu fotoğrafta patates kızartması tespit edemedi. Eğitim epoch sayısını artırmamız gerekebilir.")