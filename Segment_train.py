from ultralytics import YOLO

model_size = 's'
batch_size = '8'

modified_model = 'C2f-Faster-EMA-bifpn'
layer = '175'
data_type = 'ABO_9'  # ABO Neurofinder Mixed dataset
weight_path = ''  # weights/segmentation/yolov8s-seg.pt
# model = YOLO(f'ultralytics/cfg/models/v8/yolov8{model_size}-seg-{modified_model}.yaml').load(weight_path)

if not weight_path:
    project_folder = 'segment'
else:
    project_folder = 'segment_weights'

model = YOLO(f'ultralytics/cfg/models/v8/yolov8{model_size}-seg-{modified_model}.yaml')

model.train(data=f'dataset_cfg/{data_type}/neuron-seg-{data_type}-{layer}.yaml',
            cache=True,  # Use cache for data loading
            imgsz=640,
            epochs=500,
            patience=50,
            batch=int(batch_size),
            close_mosaic=10,
            # workers=32,
            device='0',
            # resume=True,
            amp=False,  # close amp
            project=f'runs/{project_folder}/{data_type}/train',
            name=f'yolo-{model_size}-{modified_model}-{layer}',
            # name=f'yolo-{model_size}-{modified_model}',
            plots=True,
            )

