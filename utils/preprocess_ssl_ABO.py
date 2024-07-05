import os
import cv2
import h5py
import numpy as np


def data_normal(float_data):
    """data normalization
    The purpose is to convert images of the float data type to uint8 format and store them.
    input:  float64,float32,float16 np.array
    output: uint8                   np.array
    """
    uint8_data = (float_data - np.min(float_data)) / (np.max(float_data) - np.min(float_data)) * 255
    uint8_data = uint8_data.astype('uint8')
    return uint8_data


def gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    assert s > 0, 'stride size has to be bigger than 0'
    for j in range(0, i - k + 1, s):
        yield j
        if j + k > i:
            yield i - k


for layer in ['275', '175']:
    ori_data_dir = f'/storage2/kk/Data/ABO/{layer}/20_percent_6Hz/H5'  # The path where you store the raw h5 data
    for abo_h5 in os.listdir(ori_data_dir):
        if abo_h5.endswith('.h5'):
            print(abo_h5)
            abo_h5_id = abo_h5.split('.h5')[0]
            abo_h5_path = os.path.join(ori_data_dir, abo_h5)
            abo_h5_data = h5py.File(abo_h5_path, 'r')['mov'][:]
            num_frames, height, width = abo_h5_data.shape
            patch = stride = 10
            z_steps = gen_indices(num_frames, patch, stride)
            for i, z in enumerate(z_steps):
                # print(i, z, z+10)
                # print(z)
                max_projection_images = np.max(abo_h5_data[z:z+10], axis=0)
                max_projection_images = data_normal(max_projection_images)
                max_projection_images_dir = f'../dataset/ssl/ABO'
                os.makedirs(max_projection_images_dir, exist_ok=True)
                max_projection_images_path = os.path.join(max_projection_images_dir, abo_h5_id+f'_{i}.png')
                cv2.imwrite(max_projection_images_path, max_projection_images)



