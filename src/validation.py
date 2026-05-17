import cv2
import numpy as np

# Kalibrasyon fotoğrafını yükle
img = cv2.imread("kalibrasyon_kagidi2.jpg")
# Renk uzayını HSV'ye çevirerek rengi izole et
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Örnek renk aralığı (Mavi kağıt için maskeleme)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Piksel alanını hesapla
pixel_area = np.sum(mask > 0)
real_area_cm2 = 100.0  # 10x10 cm kağıt

calibrated_ratio = pixel_area / real_area_cm2
print(f"Senin Sistemin İçin Gerçek Katsayı: {calibrated_ratio}")