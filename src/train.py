from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('models/pretrained/yolo11n-seg.pt') 

# KESİN ÇÖZÜM: Çalıştığın dizinin tam yolunu al ve 'runs' klasörüne sabitle
    project_runs_dir = os.path.join(os.getcwd(), 'runs')
    results = model.train(
        data='data/processed/data.yaml',
        epochs=50,              # Gerçek eğitim için epoch sayısını artırdık
        imgsz=640,
        batch=8,
        project=project_runs_dir,          # ÖNEMLİ: Çıktıları direkt bu proje klasöründeki 'runs' içine kaydeder
        name='checkfit-ai',   # Sonuçlar 'runs/checkfit-ai' altında toplanır
        exist_ok=True,           # Her seferinde yeni klasör açmaz, üzerine yazar
        device='0',
        workers=0 
    )

    print("Eğitim tamamlandı! Sonuçlar proje dizinindeki 'runs/checkfit-ai' klasörüne kaydedildi.")