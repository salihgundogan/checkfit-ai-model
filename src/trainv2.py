from ultralytics import YOLO
import os
import glob
import re

if __name__ == '__main__':
    # 1. Çalışma dizinini ve runs klasörünü tanımla
    project_runs_dir = os.path.join(os.getcwd(), 'runs')
    
    # Mevcut tüm checkfit-ai klasörlerini tara
    existing_runs = glob.glob(os.path.join(project_runs_dir, 'checkfit-ai*'))
    
    max_num = 0
    for run in existing_runs:
        match = re.search(r'checkfit-ai(\d+)', os.path.basename(run))
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    # 2. Eğer hiç eğitim klasörü yoksa uyarı ver, varsa en sonuncunun last.pt dosyasını seç
    if max_num == 0:
        print("❌ Hata: 'runs' dizini altında devam ettirilecek bir 'checkfit-ai' klasörü bulunamadı!")
    else:
        last_checkpoint = os.path.join(project_runs_dir, f'checkfit-ai{max_num}', 'weights', 'last.pt')
        
        # last.pt dosyasının varlığını kontrol et ve eğitimi başlat
        if os.path.exists(last_checkpoint):
            print(f"🔄 CheckFit-AI son eğitimi tespit edildi: checkfit-ai{max_num}")
            print(f"Ağırlık yükleniyor: {last_checkpoint}")
            print("Eğitim kaldığı epoch'tan itibaren 200'e kadar devam ettiriliyor...\n")
            
            # Modeli en son kaydedilen durumla yüklüyoruz
            model = YOLO(last_checkpoint)
            
            # KESİN KURAL: resume=True yapıldığında YOLO önceki tüm parametreleri (box=7.5, cls=1.5, scale=0.5 vb.)
            # otomatik olarak klasördeki ayarlardan çeker. Yeniden parametre yazmaya gerek yoktur.
            results = model.train(resume=True)
            
            print(f"\n✅ CheckFit-AI Small Eğitimi Kaldığı Yerden Başarıyla Tamamlandı!")
        else:
            print(f"❌ Hata: '{last_checkpoint}' dosyası bulunamadı! Eğitimin yarıda kesildiğinden emin olun.")