# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import os
import copy
import torch
import time
import tqdm
import warnings
from torch import nn
import pandas as pd
from collections import OrderedDict
from prefetch_generator import BackgroundGenerator
from utils.process import cal_time, make_log_dir
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.utils.scheduler import cosine_schedule

warnings.filterwarnings("ignore", category=UserWarning)

class TiCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = TiCoProjectionHead(512, 4096, 256)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


if __name__ == "__main__":
    import argparse
    # from ultralytics.nn.tasks import DetectionModel, YoloBackbone, SegmentationModel
    from ultralytics import YOLO
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TiCo', help='the name of the self-learning-model')
    parser.add_argument('--dataset', type=str, default='ABO', help='the name of the dataset  ABO/Neurofinder')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_frequency', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--best_avg_loss', type=int, default=1000)

    parser.add_argument('--cfg', type=str, default=f'ultralytics/cfg/models/v8/yolov8s-seg-C2f-Faster-EMA-bifpn.yaml',
                        help='model.yaml')
    opt = parser.parse_args()

    # Create model
    model = YOLO(opt.cfg)
    backbone = nn.Sequential(*list(model.model.children())[0][:10], nn.AdaptiveAvgPool2d((1, 1)))
    model = TiCo(backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform_ = SimCLRTransform(input_size=512)
    root_dir = '/media/user1/477137b6-5640-470f-80f7-ec6dd6a1d8c5/'

    dataset = LightlyDataset(os.path.join(root_dir, f"./Dataset/ssl/{opt.dataset}_all/"), transform=transform_)  # ABO_ssl

    collate_fn = MultiViewCollate()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    criterion = TiCoLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2 * opt.batch_size / 256, weight_decay=1.5 * 1e-6)
    # optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    print("Starting Training")
    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
    ])
    check_point_dir = f'./runs/ssl/{opt.dataset}/{opt.model_name}/'
    os.makedirs(check_point_dir, exist_ok=True)
    check_point_dir_ = make_log_dir(check_point_dir, opt.model_name, opt.batch_size)

    for epoch in range(opt.epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, opt.epochs, 0.996, 1)
        for (x0, x1), _, _ in tqdm.tqdm(BackgroundGenerator(dataloader)):
            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(
                model.projection_head, model.projection_head_momentum, m=momentum_val
            )
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model.forward_momentum(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        log['epoch'].append(epoch)
        log['loss'].append(f'{avg_loss.item():.4f}')
        pd.DataFrame(log).to_csv(os.path.join(check_point_dir_, 'log.csv'), index=False)

        if opt.best_avg_loss > avg_loss:
            # torch.save(check_point_dir, "best_yoloV8_BackboneV4.pth")
            torch.save(model.state_dict(), os.path.join(check_point_dir_, f"yolov8s-seg-{opt.model_name}-best.pt"))
            print(f"Finding optimal model params. Loss is dropping from {opt.best_avg_loss:.4f} to {avg_loss:.4f}")
            opt.best_avg_loss = avg_loss

    end_time = time.time()
    cal_time(start_time, end_time)
