import os
import numpy as np
import yaml
import shutil
import pandas as pd
from collections import OrderedDict
from ultralytics import YOLO

project_folder = 'segment'
modified_model = 'C2f-Faster-EMA-bifpn'
# for data_type in ['Neurofinder_']:
for data_type in ['ABO_9']:
    for layer in ['175', '275']:
        model = YOLO(f'runs/{project_folder}/{data_type}/train/yolo-s-{modified_model}-{layer}/weights/best.pt')
        # model = YOLO(f'runs/segment_finetune/{data_type}/train/yolo-s-{modified_model}-{layer}-TiCo-freeze/weights/best.pt')
        dataset = f'train_{layer}'
        log = OrderedDict([
            ('experiment', []),
            ('precision', []),
            ('recall', []),
            ('F1', []),
        ])
        for coco in ['images', 'labels']:
            for engine in ['val']:
                root_dir = f'dataset/{data_type}/{dataset}/{engine}/{coco}'
                for avg, ind in enumerate(os.listdir(root_dir)):
                    if ind.endswith("jpg"):
                        id_ = ind.split('.jpg')[0]
                    elif ind.endswith("txt"):
                        id_ = ind.split('.txt')[0]
                    # create dataset
                    ori_path = os.path.join(root_dir, ind)
                    val_dir = f'dataset_val/{data_type}/{dataset}/{id_}/{engine}/{coco}'
                    os.makedirs(val_dir, exist_ok=True)
                    new_path = os.path.join(val_dir, ind)
                    if not os.path.exists(new_path):
                        shutil.copy(ori_path, new_path)
                    if coco=='images':
                        # create yaml config
                        cfg_dir = f'dataset_val_cfg/{data_type}/{layer}/'
                        os.makedirs(cfg_dir, exist_ok=True)
                        cfg_file = os.path.join(cfg_dir, f'neuron-seg-{data_type}-{layer}-{id_}.yaml')
                        cfg_data = {
                            "path": f"/media/user1/477137b6-5640-470f-80f7-ec6dd6a1d8c5/"
                                    f"ultralytics-main-neuroseg/dataset_val/{data_type}/train_{layer}/{id_}",
                            "train": None,
                            "val": "val",
                            "test": None,
                            "names": {
                                0: "neuron"
                            }
                        }
                        if not os.path.exists(cfg_file):
                            # 将数据写入YAML文件
                            with open(cfg_file, 'w') as yaml_file:
                                yaml.dump(cfg_data, yaml_file, default_flow_style=False)

                        projection_dir = f'runs/segment/{data_type}/metric/{layer}/yolo-s-{modified_model}/'
                        val_projection_dir = os.path.join(projection_dir, 'SingleResults')
                        os.makedirs(val_projection_dir, exist_ok=True)

                        metric = model.val(data=cfg_file,
                                           project=val_projection_dir,
                                           name=f'yolo-s-{dataset}-{id_}',
                                           save_json=True,
                                           conf=0.001,
                                           iou=0.5,
                                           max_det=300,  #
                                           half=True,  # high accuracy
                                           rect=False,  # False is better
                                           split='val')
                        precision = metric.seg.mp
                        recall = metric.seg.mr
                        F1 = metric.seg.f1

                        log['experiment'].append(id_)
                        log['precision'].append(precision)
                        log['recall'].append(recall)
                        log['F1'].append(F1)

                        if avg==len(os.listdir(root_dir))-1:
                            log['experiment'].append('AVG')
                            log['precision'].append(np.array(log['precision']).mean())
                            log['recall'].append(np.array(log['recall']).mean())
                            log['F1'].append(np.array(log['F1']).mean())

        csv_path = os.path.join(projection_dir, f'{data_type}_{layer}.csv')
        pd.DataFrame(log).to_csv( csv_path, index=False)
