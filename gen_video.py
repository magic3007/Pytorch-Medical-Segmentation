#%%
import cv2
import numpy as np
import glob
import os.path as osp
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk

#%%
input_dir = './thyroid_dataset/val/inputs'
mask_dir = './results/deepabv3'
ouput_dir = './seg/deeplabv3/val'

#%%
def gen_video(input_dir, mask_dir, ouput_dir):
    os.makedirs(ouput_dir, exist_ok=True)
    #%%
    inputs = sorted(glob.glob(osp.join(input_dir, '*.png')))
    masks = sorted(glob.glob(osp.join(mask_dir, '*.nii.gz')))
    assert len(inputs) == len(masks)

    seg_imgs = []
    for input_path, mask_path in zip(inputs, masks):
        img = cv2.imread(input_path, 1)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        mask = mask.squeeze(0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
        # cv2 BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[..., 2] = np.where(mask == 1, 255, img[..., 2])
        seg_imgs.append(img)
        out_path = osp.join(ouput_dir, osp.basename(input_path))
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
# %%
