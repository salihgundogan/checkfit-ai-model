from ultralytics import YOLO

# 1. 106 epoch'luk eğitimin sonucunda oluşan en iyi modeli yükle
# Not: 'checkfit-ai1' kısmını, runs klasöründeki son eğitiminin adıyla değiştir.
model_path = 'runs/checkfit-ai7/weights/best.pt'
model = YOLO(model_path)

print(f"{model_path} yükleniyor ve ONNX formatına dönüştürülüyor...")

# 2. Modeli ONNX formatında dışa aktar
# simplify=True: Model mimarisini gereksiz işlemlerden arındırıp mobil için hafifletir.
export_path = model.export(
    format='onnx', 
    imgsz=640,        # Eğitimde kullandığın çözünürlük
    simplify=True,    # ONNX grafiğini optimize eder (mobilde hız kazandırır)
    opset=12          # Mobil platformlarla en uyumlu ONNX versiyonlarından biri
)

print(f"\n✅ Dönüşüm Başarılı! Arkadaşına göndereceğin dosya: {export_path}")
