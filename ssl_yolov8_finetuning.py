from ultralytics import YOLO
import torch


class FinetuneYolo(YOLO):
    def load_backbone(self, ckptPath):
        """
        Transfers backbone parameters with matching names and shapes from 'weights' to model.
        """
        backboneWeights = torch.load(ckptPath)
        self.model.load_state_dict(backboneWeights, strict=False)
        return self

    def freeze_backbone(self, freeze):
        # Freeze backbone params
        freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze):
                v.requires_grad = False
        return self

    def unfreeze_backbone(self):
        # unfreeze backbone params
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train_yolos_ABO_20_train275 all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        return self


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=f'ABO', help='model.yaml')
    parser.add_argument('--split', type=str, default=f'10', help='model.yaml')
    parser.add_argument('--layer', type=str, default='175', help='')
    parser.add_argument('--model_size', type=str, default='s', help='model.yaml')
    parser.add_argument('--modified_model', type=str, default='C2f-Faster-EMA-2-bifpn', help='')
    parser.add_argument('--ssl_model', type=str, default='TiCo', help='model.yaml')
    parser.add_argument('--ssl_dir', type=str, default='20240122T0934', help='model.yaml')
    parser.add_argument('--freeze', type=bool, default=False, help='model.yaml')
    parser.add_argument('--freeze_layer', type=int, default=10, help='model.yaml')

    opt = parser.parse_args()
    ckptPath = f"runs/ssl/{opt.dataset}/20/{opt.ssl_model}/{opt.ssl_model}_128_{opt.ssl_dir}/yolov8{opt.model_size}-seg-{opt.ssl_model}-best.pt"

    cfg = f'ultralytics/cfg/models/v8/yolov8{opt.model_size}-seg-{opt.modified_model}.yaml'
    # model = FinetuneYolo(cfg).load_backbone(ckptPath)  # build from YAML and transfer weights
    model = FinetuneYolo(cfg).load_backbone(ckptPath)  # build from YAML and transfer weights
    if opt.freeze:
        model.freeze_backbone(opt.freeze_layer)
        model_name = f'yolo-{opt.model_size}-{opt.modified_model}-{opt.layer}'
        pro_dir = f'runs/segment_/{opt.dataset}_{opt.split}/train_{opt.ssl_model}_unfreeze'
    else:
        model.unfreeze_backbone()
        model_name = f'yolo-{opt.model_size}-{opt.modified_model}-{opt.layer}'
        pro_dir = f'runs/segment_/{opt.dataset}_{opt.split}/train_{opt.ssl_model}_freeze'

    model.train(data=f'dataset_cfg/{opt.dataset}_{opt.split}/neuron-seg-{opt.dataset}_{opt.split}-{opt.layer}.yaml',
                cache=True,
                epochs=500,
                patience=50,
                batch=8,
                workers=16,
                device='0',
                # rect=False,
                optimizer='auto',  # using SGD
                amp=True,  # close amp
                project=pro_dir,
                name=model_name,
                )