from ultralytics import YOLO
model = YOLO(r"yolov_8n_model_vAlpha\runs\detect\train3\weights\best.pt")
model.export(format="onnx", dynamic=True, simplify=True, opset=12)