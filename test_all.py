from ultralytics import YOLO

# layer = '275'
# model = YOLO(f'runs/segment/train_yolos_abo_6_{layer}/weights/best.pt')
# model.val(data = f'neuron-seg-abo-1-{layer}.yaml', device = 0, iou = 0.5, conf = 0.25, max_det = 1000, batch = 16, split = 'val')

model_size = 's'  # n s m l x
layer = '175'
# layer = '275'
# modified_model = 'adam'
modified_model = 'C2f-Faster-EMA-bifpn'


data_type = 'ABO_9'  # ABO Neurofinder
# model = YOLO(f'runs/Neurofinder/train/yolo-{model_size}-{dataset}/weights/best.pt')
model = YOLO(f'runs/segment/{data_type}/train/yolo-s-{modified_model}-{layer}/weights/best.onnx')
# model = YOLO(f'runs/segment/{data_type}/train/yolo-s-{modified_model}-{layer}/weights/best.engi/ne')
# model = YOLO(f'weights/segmentation/yolov8s-seg.pt')

# metric = model.val(data=f'dataset_cfg/{data_type}/neuron-seg-{data_type}-{layer}.yaml',
metric = model.val(data=f'dataset_cfg/{data_type}/neuron-seg-{data_type}-{layer}.yaml',
                   project=f'runs/segment/{data_type}/val',
                   name=f'yolo-{model_size}-{modified_model}-{layer}',
                   save_json=True,  # slow speed high accuracy
                   # save_hybrid=False,
                   conf=0.001,
                   batch=1,
                   iou=0.5,
                   max_det=300,  #
                   half=True,  # high accuracy
                   rect=False,  # False is better
                   split='val')
