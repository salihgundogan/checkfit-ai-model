from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-seg.pt') 

    results = model.train(
        data='dataset/data.yaml',
        epochs=50,              # Gerçek eğitim için epoch sayısını artırdık
        imgsz=640,
        batch=8,
        project='runs',          # ÖNEMLİ: Çıktıları direkt bu proje klasöründeki 'runs' içine kaydeder
        name='patates_modeli',   # Sonuçlar 'runs/patates_modeli' altında toplanır
        exist_ok=True,           # Her seferinde yeni klasör açmaz, üzerine yazar
        device='0',
        workers=0 
    )

    print("Eğitim tamamlandı! Sonuçlar proje dizinindeki 'runs/patates_modeli' klasörüne kaydedildi.")