import torch
import os
from ultralytics import YOLO

cpu_count = os.cpu_count()

torch.set_num_threads(cpu_count)


model = YOLO("yolov8n.pt")

model.train(data="C:/Users/alper/Downloads/firtina.v1-alpha.yolov8/data.yaml", epochs=20, imgsz=320, batch=8)
