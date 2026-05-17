# 🥔 Patates Alan Hesaplama & YOLOv11 Segmentasyon Projesi

Bu proje, görüntüler üzerinden yapay zeka (YOLOv11-segmentation) kullanarak patates kızartmalarını segmentlere ayırır (talan eder), piksel alanlarını hesaplar ve kalibrasyon katsayısı yardımıyla bu alanları gerçek **cm²** değerlerine dönüştürür.

Proje, akademik ve endüstriyel standartlara (MLOps) uygun, temiz, modüler ve profesyonel bir klasör yapısıyla organize edilmiştir.

---

## 🏗️ Proje Klasör Yapısı

Proje dosyaları, veri setleri, modeller ve kodlar tamamen birbirinden ayrıştırılmıştır:

```text
Egitim/
│
├── .venv/                     # Python Sanal Ortamı
├── .vscode/                   # Geliştirici Ortamı Ayarları
│
├── src/                       # Kaynak Kodlar (Python Scriptleri)
│   ├── xml_to_yolo_seg.py     # Ham XML etiketlerini YOLO segmentasyon formatına çevirir
│   ├── split_data.py          # Verileri %80 Train / %20 Val olarak böler
│   ├── train.py               # YOLOv11-seg modelini veri setimizle eğitir
│   ├── validation.py          # Referans mavi kağıt ile kalibrasyon katsayısını hesaplar
│   └── predict_and_calculate.py# Eğitilmiş model ile patates alanlarını ve cm²'yi hesaplar
│
├── data/                      # Projede Kullanılan Tüm Veriler
│   ├── raw/                   # İşlenmemiş Ham Veriler
│   │   ├── images/            # Kamera/telefondan gelen ham patates resimleri (500 adet)
│   │   └── xml/               # CVAT'tan indirilen ham XML koordinat etiketleri
│   │
│   ├── intermediate/          # Ara Dönüşüm Verileri
│   │   └── labels/            # XML'lerden üretilen YOLO formatındaki TXT etiketleri
│   │
│   ├── processed/             # Eğitime Hazır Bölünmüş Veri Seti (split_data çıktısı)
│   │   ├── images/            # train ve val klasörleri
│   │   ├── labels/            # train ve val klasörleri
│   │   └── data.yaml          # YOLOv11 Veri Konfigürasyon Dosyası
│   │
│   ├── calibration/           # Kalibrasyon ve Referans Görselleri
│   │   ├── kalibrasyon_kagidi1.jpg
│   │   └── kalibrasyon_kagidi2.jpg
│   │
│   └── backup_xml/            # Orijinal XML Yedekleri
│
├── models/                    # Model Ağırlıkları (.pt dosyaları)
│   ├── pretrained/            # Hazır indirilmiş YOLO modelleri (yolo11n-seg.pt, yolo26n.pt)
│   └── trained/               # Eğittiğiniz en iyi modellerin yedekleri
│
├── runs/                      # YOLO Eğitim ve Tahmin Çıktıları (Otomatik oluşur)
│
└── docs/                      # Proje Notları ve Dokümantasyon
    ├── yapilacak.docx
    └── cvattan gelenler.txt
```

---

## 🚀 Proje Çalıştırma Adımları (Pipeline)

Projede verileri hazırlamaktan eğitime ve tahmine kadar tüm işlemler sırasıyla şu scriptler çalıştırılarak yapılır:

> [!IMPORTANT]
> Tüm scriptleri terminalde **proje kök dizinindeyken (`Egitim/` klasöründeyken)** çalıştırınız.

### Adım 1: XML Etiketlerini YOLO Formatına Dönüştürme
CVAT koordinat etiketlerini YOLO segmentasyon standardı olan normalize edilmiş `TXT` dosyalarına dönüştürür ve `data/intermediate/labels/` klasörüne kaydeder:
```bash
python src/xml_to_yolo_seg.py
```

### Adım 2: Veri Setini Train/Val Olarak Bölme
Ham resimleri ve oluşturulan TXT etiketlerini rastgele karıştırıp %80 Eğitim, %20 Doğrulama olarak böler ve `data/processed/` klasöründe YOLO formatına uygun olarak kopyalar:
```bash
python src/split_data.py
```

### Adım 3: YOLO Modelini Eğitme
Oluşturulan veri seti konfigürasyonunu (`data.yaml`) referans alarak YOLOv11 segmentasyon modelini eğitir. Sonuçlar otomatik olarak `runs/` klasörüne kaydedilir:
```bash
python src/train.py
```

### Adım 4: Sistem Kalibrasyonu (Piksel -> cm²)
Kameranın tabak alanını doğru hesaplayabilmesi için referans mavi kağıt görseli (`data/calibration/kalibrasyon_kagidi2.jpg`) üzerinden piksel-cm² katsayısını hesaplar:
```bash
python src/validation.py
```

### Adım 5: Patates Alanını ve Gerçek Boyutunu Hesaplama
Eğitilen modeli yükleyerek test görselindeki patates kızartmalarını segmentlere ayırır, toplam piksel alanını bulur ve kalibrasyon katsayısına bölerek **gerçek cm² boyutunu** hesaplar:
```bash
python src/predict_and_calculate.py
```

---

## 🎯 Profesyonel Geliştirici Notları
* Kodlardaki tüm dosya yolları göreceli (`relative path`) olarak projenin ana dizinine göre ayarlanmıştır. Bu sayede projenizi başka bir bilgisayara taşıdığınızda yollar bozulmaz.
* Git versiyon kontrol sistemi kullanırken model çıktılarının (`runs/`) ve sanal ortamın (`.venv/`) git geçmişine yüklenmemesi için `.gitignore` dosyası yapılandırılmıştır.
