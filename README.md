# NutriSense: YOLOv11-Segmentation Tabanlı Görüntü İşleme ve Alan Hesaplama Boru Hattı

Bu proje, bilgisayarlı görü ve derin öğrenme (YOLOv11-segmentation) tekniklerini kullanarak gıda nesnelerinin (örneğin patates kızartması) otonom olarak segmentasyonunu gerçekleştirir, pikseller üzerinden alan hesaplaması yapar ve referans bir kalibrasyon nesnesi aracılığıyla bu değerleri metrik (cm²) ölçülere dönüştürür.

Sistem; akademik araştırmalara ve endüstriyel MLOps (Makine Öğrenimi Operasyonları) standartlarına uygun, modüler, ölçeklenebilir ve yüksek performanslı bir mimari temel alınarak geliştirilmiştir.

---

## Desteklenen Sınıflar
Model, veri seti (CVAT üzerinden etiketlenmiş) kapsamında 10 farklı sınıfı tespit ve segmente edebilmektedir:

- Baklava
- Hamburger
- Kebap
- Köfte
- Lahmacun
- Patates Kızartması
- Pizza
- Sosisli
- Sütlaç
- Waffle

---

## Proje Mimarisi ve Klasör Yapısı
Proje bileşenleri, veri seti yönetimini ve model eğitimini birbirinden izole edecek şekilde yapılandırılmıştır:

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
│   └── calibration/           # Kalibrasyon Görselleri
│
├── models/                    # Model Ağırlıkları (.pt dosyaları - Git dışı tutulur)
│   ├── pretrained/            # Hazır YOLO modelleri (yolo11n-seg.pt vb.)
│   └── trained/               # En iyi performans gösteren eğitilmiş modellerin yedekleri
│
└── runs/                      # YOLO Eğitim ve Tahmin Çıktıları (Otomatik oluşur - Git dışı tutulur)
```

---

## Kurulum ve Çalıştırma Adımları

Tüm komut dosyaları proje kök dizininde (`Egitim/` klasöründe) çalıştırılmalıdır.

### 1. Etiket Dönüştürme (XML to YOLO)
CVAT üzerinden elde edilen çokgen (polygon) etiketlerini, YOLO formatına uygun normalize edilmiş `TXT` dosyalarına dönüştürür.
```bash
python src/xml_to_yolo_seg.py
```

### 2. Veri Seti Bölümlendirme (Train / Val Split)
Görüntüleri ve karşılık gelen TXT etiketlerini %80 Eğitim (Train) ve %20 Doğrulama (Validation) kümelerine ayırır. Eğitim konfigürasyon dosyası (`data.yaml`) otomatik olarak oluşturulur.
```bash
python src/split_data.py
```

### 3. Model Eğitimi
Veri seti konfigürasyonu (`data.yaml`) kullanılarak YOLOv11 segmentasyon modeli eğitilir. Eğitim çıktıları `runs/` dizininde versiyonlanarak saklanır.
```bash
python src/train.py
```

### 4. Sistem Kalibrasyonu
Kamera perspektifinden kaynaklanan alan değişimlerini hesaplayabilmek için referans bir nesne (örneğin mavi bir kalibrasyon kâğıdı) üzerinden piksel-cm² oranını (katsayıyı) hesaplar.
```bash
python src/validation.py
```

### 5. Alan Hesaplama ve Tahmin
Eğitilmiş model test görüntüsü üzerinde çalıştırılarak nesne segmentasyonlarını gerçekleştirir. Elde edilen toplam piksel alanı, kalibrasyon katsayısı kullanılarak metrik cm² değerine dönüştürülür.
```bash
python src/predict_and_calculate.py
```

---

## MLOps ve Geliştirici Standartları
- **Bağımsız Çalışma Ortamı (Portability):** Kod yapısındaki dizin yolları tamamen göreceli (relative path) olarak tasarlanmıştır. Bu sayede proje, ortam bağımsız olarak çalıştırılabilir.
- **Optimizasyon ve Sürüm Kontrolü:** Model ağırlıkları (`.pt`), eğitim çıktıları (`runs/`) ve hacimli veri setleri (`data/`) Git takibinden dışlanarak (`.gitignore` aracılığıyla) deponun hafif ve yönetilebilir kalması sağlanmıştır.
