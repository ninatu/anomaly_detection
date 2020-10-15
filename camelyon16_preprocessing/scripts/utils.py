import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import skimage.io
import re


TUMOR_LABEL = 127
NORMAL_LABEL = 255
PATCH_NAME_PAT = re.compile('(?P<image_name>.*)_(?P<crop_type>.*)_x_(?P<x>\d+)_y_(?P<y>\d+)_w_(?P<w>\d+)_h_(?P<h>\d+)')
PATCH_NAME_FORMAT = '{image_name}_{crop_type}_x_{x}_y_{y}_w_{w}_h_{h}.jpg'


def get_tissue_mask(img):
    gimg = np.array(rgb2gray(np.array(img)))  # convert to grayscale
    try:
        thresh = threshold_otsu(gimg)
        binary = gimg > thresh
        mask = (binary == 0).astype(np.uint8)  # tissie pixels are black
        return mask
    except Exception as e:
        return np.zeros((gimg.shape[0], gimg.shape[1]), dtype=np.uint8)


def preprocess_normal_mask_decrease_thrice(tissue_mask):
    # remove boarders
    h, w = tissue_mask.shape
    if h > 700:
        tissue_mask[:100] = 0
        tissue_mask[-100:] = 0
        tissue_mask[:, :50] = 0
    else:
        tissue_mask[:30] = 0
        tissue_mask[-30:] = 0
        tissue_mask[:, :30] = 0
        tissue_mask[:, -30:] = 0

    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3)))
    tissue_mask = (skimage.transform.downscale_local_mean(tissue_mask, (3, 3)) == 1).astype(
        np.uint8)
    return tissue_mask


def preprocess_tumor_mask_decrease_thrice(tissue_mask):
    # remove boarders
    h, w = tissue_mask.shape
    if h > 700:
        tissue_mask[:100] = 0
        tissue_mask[-100:] = 0
        tissue_mask[:, :50] = 0
    else:
        tissue_mask[:30] = 0
        tissue_mask[-30:] = 0
        tissue_mask[:, :30] = 0
        tissue_mask[:, -30:] = 0

    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3)))
    tissue_mask = (skimage.transform.downscale_local_mean(tissue_mask, (3, 3)) == 1).astype(
        np.uint8)
    return tissue_mask


def sample_tissue_pixels(binary, max_count=-1):
    """
    return list of (w, h)
    """
    indexs = enumerate(binary.flatten())
    tissue_indexs = filter(lambda x: x[1] == 1, indexs)
    ncols = binary.shape[1]
    tissue_coords = [(index % ncols, index // ncols) for index, _ in tissue_indexs]

    np.random.shuffle(tissue_coords)
    if max_count > 0:
        return tissue_coords[:max_count]
    else:
        return tissue_coords


def get_tissue_bb(mask):
    # return: x, y, w, h
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        return cv2.boundingRect(np.concatenate(contours, axis=0))
    return 0, 0, 0, 0
