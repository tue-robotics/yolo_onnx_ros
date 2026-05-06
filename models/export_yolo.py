from ultralytics import YOLO
model = YOLO("yolo26m.pt")
# Default: NMS-free one-to-one head — works with updated code
model.export(format="onnx")
# Or for the one-to-many head (requires NMS — also works):
# model.export(format="onnx", end2end=False)