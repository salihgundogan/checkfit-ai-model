from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('models/pretrained/yolo11n-seg.pt') 

# KESİN ÇÖZÜM: Çalıştığın dizinin tam yolunu al ve 'runs' klasörüne sabitle
    project_runs_dir = os.path.join(os.getcwd(), 'runs')
    
    # checkfit-ai1, checkfit-ai2 şeklinde sırayla kaydetmesi için otomatik isimlendirme
    import glob
    import re
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
    
    results = model.train(
        data='data/processed/data.yaml',
        epochs=200,              
        imgsz=640,
        batch=16,
        project=project_runs_dir,          # Çıktıları bu klasöre kaydeder
        name=run_name,        # Sonuçlar dinamik isme (checkfit-ai1, checkfit-ai2 vb.) kaydedilir
        exist_ok=False,       # Klasörün üzerine yazmasını engeller
        device='0',
        workers=0 
    )

    print(f"Eğitim tamamlandı! Sonuçlar proje dizinindeki 'runs/{run_name}' klasörüne kaydedildi.")