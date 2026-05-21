from ultralytics import YOLO
import os
import glob
import re

if __name__ == '__main__':
    # 1. Modeli Yükle
    model = YOLO('models/pretrained/yolo11n-seg.pt') 

    # KESİN ÇÖZÜM: Windows ortamında PyTorch hatalarını önlemek için 
    # tüm işlemler 'if __name__ == '__main__':' bloğunun İÇİNDE (girintili) olmalıdır.
    project_runs_dir = os.path.join(os.getcwd(), 'runs')
    os.makedirs(project_runs_dir, exist_ok=True)
    
    existing_runs = glob.glob(os.path.join(project_runs_dir, 'checkfit-ai*'))
    
    max_num = 0
    for run in existing_runs:
        match = re.search(r'checkfit-ai(\d+)', os.path.basename(run))
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    run_name = f'checkfit-ai{max_num + 1}'
    
    # 2. Optimize Edilmiş Eğitimi Başlat
    results = model.train(
        data='data/processed/data.yaml',
        epochs=2,              # 2 epoch'luk testten ana eğitime geçiş
        patience=30,             # 30 epoch boyunca mAP artmazsa eğitimi otomatik durdur
        imgsz=640,
        batch=16,
        project=project_runs_dir,          
        name=run_name,        
        exist_ok=False,       
        device='0',              # RTX 3050 GPU'yu aktif eder
        workers=0,               # Windows 11'de çoklu işlem (multiprocessing) çökmesini engeller
        
        # --- CHECKFIT-AI MODEL OPTİMİZASYONLARI ---
        image_weights=True,      # Çok önemli: Veri setinde az bulunan Hamburger, Sosisli, Waffle gibi sınıflara ağırlık verir
        degrees=10.0,            # Görselleri rastgele +/- 10 derece döndürerek farklı tabak açılarını öğretir
        hsv_s=0.4,               # Renk doygunluğunu değiştirerek farklı ortam ışıklarını simüle eder
        hsv_v=0.4,               # Parlaklık değişimi ile gölgeli fotoğraflara karşı direnci artırır
        mosaic=1.0               # Mozaikleme (farklı yemekleri tek karede birleştirme) yöntemini %100 uygular
    )

    print(f"\n✅ Eğitim tamamlandı! Sonuçlar proje dizinindeki 'runs/{run_name}' klasörüne kaydedildi.")