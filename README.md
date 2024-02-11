# NeuroSeg3
NeuroSeg3 is an self-supervised learning approach  designed to achieve fast and precise segmentation of neurons in imaging data. This approach consists of two modules: a self-supervised pre-training network and a segmentation network. After pre-training the encoder of the segmentation network via a self-supervised learning method without any annotated masks, we only need to fine-tune the segmentation network with a small amount of annotated data. The segmentation network is designed with YOLOv8s, FasterNet,  EMA and BiFPN, which enhanced the model's segmentation accuracy while reducing the computational cost and parameters. 

##### The code of Neuroseg-Ⅲ implements the following functionalities:

- Improve the segment work, based on  YOLOv8s, with FasterNet,  EMA and BiFPN.
- Training the encoder of the segmentation network via TiCo without any annotated masks.
- Integrating the pre-trained encoder with the segment network for fine-tuning.
- Evaluation of the Neuroseg-Ⅲ framework with standard metrics.

## System Requirements

- A CUDA compatible GPU
- Anaconda with Python 3.8
- Pytorch 1.13.1 (CUDA Toolkit 11.6 and cuDNN v8.3.2 required)
- Lightly for self-supervised learning.

You can compile the environment as the following steps:

```bash
conda env create -f Neuroseg3_environment.yaml
python setup.py intall
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Link to ABO Dataset:

[Allen Brain Observatory dataset](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-–-Visual-Coding-on-AWS)

## Contact information

If you have any questions about this project, please feel free to contact us. Email address: [wuu_yukun@163.com](
