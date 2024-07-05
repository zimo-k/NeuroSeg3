from ultralytics import YOLO

# Load model
model = YOLO("runs/segment/ABO_9/train/yolo-s-C2f-Faster-EMA-bifpn-175/weights/best.pt")
# model = YOLO("./weights/segmentation/yolov8s-seg.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'
