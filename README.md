# 🥔 NutriSense: YOLOv11-Segmentation ile Yemek Alanı Hesaplama Pipeline'ı

Bu proje; görüntüler üzerinden yapay zeka (**YOLOv11-segmentation**) kullanarak yemekleri (özellikle patates kızartmalarını) segmentlere ayırır, piksel bazlı alanlarını hesaplar ve referans bir mavi kalibrasyon nesnesi yardımıyla bu alanları hassas bir şekilde gerçek **cm²** değerlerine dönüştürür.

Akademik ve endüstriyel standartlara (**MLOps**) uygun, temiz, modüler ve yüksek performanslı bir yapay zeka boru hattı (pipeline) olarak tasarlanmıştır.

---

## 🍽️ Desteklenen Yemek Sınıfları (10 Sınıf)
Sistemimiz, CVAT üzerinde etiketlenmiş ve YOLOv11 modeliyle eğitilmeye hazır olan aşağıdaki 10 farklı yemek sınıfını desteklemektedir:

| ID | Yemek Sınıfı | İkon | ID | Yemek Sınıfı | İkon |
|:--:|:--|:--:|:--:|:--|:--:|
| **0** | Baklava | 🥐 | **5** | Patates Kızartması | 🍟 |
| **1** | Hamburger | 🍔 | **6** | Pizza | 🍕 |
| **2** | Kebap | 🥙 | **7** | Sosisli | 🌭 |
| **3** | Köfte | 🥩 | **8** | Sütlaç | 🥣 |
| **4** | Lahmacun | 🌯 | **9** | Waffle | 🧇 |

---

## 🏗️ Proje Klasör Yapısı
Proje dosyaları, veri setleri, modeller ve kodlar tamamen birbirinden ayrıştırılarak MLOps prensiplerine uygun şekilde organize edilmiştir:

```text
Egitim/
│
├── .venv/                     # Python Sanal Ortamı
├── .vscode/                   # VS Code Geliştirici Ortamı Ayarları
├── .gitignore                 # Büyük dosyaları (veri seti/modeller) engelleyen Git kuralları
│
├── src/                       # Kaynak Kodlar (Python Modülleri)
│   ├── xml_to_yolo_seg.py     # CVAT XML etiketlerini normalize YOLO TXT formatına çevirir
│   ├── split_data.py          # Verileri %80 Train / %20 Val olarak böler ve data.yaml hazırlar
│   ├── train.py               # YOLOv11-seg modelini veri setimizle otomatik isimlendirmeli eğitir
│   ├── validation.py          # Referans mavi kağıt ile piksel-cm² katsayısını hesaplar
│   └── predict_and_calculate.py # Eğitilmiş model ile alan tespiti ve cm² dönüşümü yapar
│
├── data/                      # Projede Kullanılan Tüm Veriler (Git dışı tutulur)
│   ├── raw/                   # İşlenmemiş Ham Veriler
│   │   ├── images/            # Sınıflara göre ayrılmış ham yemek görselleri
│   │   └── xml/               # CVAT'tan indirilen ham XML koordinat etiketleri
│   │
│   ├── intermediate/          # Ara Dönüşüm Verileri
│   │   └── labels/            # XML'lerden üretilen YOLO segmentasyon TXT etiketleri
│   │
│   ├── processed/             # Eğitime Hazır Bölünmüş Veri Seti (split_data.py çıktısı)
│   │   ├── images/            # train ve val klasörleri
│   │   ├── labels/            # train ve val klasörleri
│   │   └── data.yaml          # YOLOv11 Veri Konfigürasyon Dosyası
│   │
│   └── calibration/           # Kalibrasyon Görselleri (Örn: Mavi kağıt)
│
├── models/                    # Model Ağırlıkları (.pt dosyaları - Git dışı tutulur)
│   ├── pretrained/            # Hazır YOLO modelleri (yolo11n-seg.pt vb.)
│   └── trained/               # En iyi performans gösteren eğitilmiş modellerin yedekleri
│
└── runs/                      # YOLO Eğitim ve Tahmin Çıktıları (Otomatik oluşur - Git dışı tutulur)
```

---

## 🚀 Proje Çalıştırma Adımları (Pipeline)

Tüm işlemleri sırasıyla terminalde **proje kök dizinindeyken (`Egitim/` klasöründeyken)** çalıştırınız.

### 1️⃣ XML Etiketlerini YOLO Formatına Dönüştürme
CVAT çokgen (polygon) etiketlerini YOLO segmentasyon standardı olan normalize edilmiş `TXT` dosyalarına dönüştürür ve `data/intermediate/labels/` klasörüne kaydeder:
```bash
python src/xml_to_yolo_seg.py
```

### 2️⃣ Veri Setini Train / Val Olarak Bölme
Resimleri ve oluşturulan TXT etiketlerini eşleştirerek rastgele karıştırır, `%80 Eğitim` ve `%20 Doğrulama` olarak böler. Ardından YOLO formatına uygun olarak `data/processed/` altına dağıtır ve otomatik olarak `data.yaml` dosyasını oluşturur:
```bash
python src/split_data.py
```

### 3️⃣ YOLOv11 Segmentasyon Modelini Eğitme
Oluşturulan veri seti konfigürasyonunu (`data.yaml`) referans alarak YOLOv11 modelini eğitir. Eğitim sonuçları çakışmayı önlemek için otomatik olarak artan numaralarla (`runs/checkfit-ai1`, `runs/checkfit-ai2` vb.) kaydedilir:
```bash
python src/train.py
```

### 4️⃣ Sistem Kalibrasyonu (Piksel -> cm²)
Kameranın gerçek tabak/yemek alanını doğru hesaplayabilmesi için referans bir mavi nesne (`data/calibration/kalibrasyon_kagidi2.jpg`) üzerinden HSV maskeleme uygulayarak piksel-cm² katsayısını hesaplar:
```bash
python src/validation.py
```

### 5️⃣ Yemek Alanını ve Gerçek Boyutunu Hesaplama
Eğitilen en iyi modeli yükler, test görselindeki nesneleri segmentlere ayırır, toplam piksel alanını hesaplar ve kalibrasyon katsayısına bölerek **gerçek cm² boyutunu** çıktı olarak verir:
```bash
python src/predict_and_calculate.py
```

---

## 🎯 Profesyonel MLOps Yaklaşımları
* **Taşınabilirlik:** Kodlardaki tüm dosya yolları göreceli (`relative path`) olarak projenin ana dizinine göre ayarlanmıştır. Başka bilgisayarlarda doğrudan çalışır.
* **Hafif Git Yapısı:** Büyük veri setleri (`data/`), model ağırlıkları (`models/` ve `*.pt`) ile eğitim çıktıları (`runs/`) `.gitignore` ile filtrelenerek Git deposuna yüklenmesi engellenmiştir. Bu sayede kod tabanı her zaman hafif ve hızlı kalır.

