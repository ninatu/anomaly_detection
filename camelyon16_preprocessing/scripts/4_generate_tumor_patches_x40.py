"""
Extract tumor patches from tumor slides
"""

import openslide
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import skimage.io
import skimage.transform
import argparse
import sys

sys.path.append('./')

from utils import get_tissue_mask, preprocess_tumor_mask_decrease_thrice, \
    sample_tissue_pixels, PATCH_NAME_FORMAT, TUMOR_LABEL


# maximum number of patches extracted from one WSI
MAX_PATCHES_PER_IMAGE = 50


def sample_patches_from_tumor_image(image_filename, image_path, annotation_path, max_count, output_dir):
    slide = openslide.OpenSlide(image_path)
    level = 8
    down_scale = int(2 ** 8)
    slide_w, slide_h = slide.dimensions

    down_img = slide.read_region((0, 0), level, (slide_w // down_scale, slide_h // down_scale)).convert('RGB')
    tissue_mask = get_tissue_mask(down_img)
    tissue_mask = preprocess_tumor_mask_decrease_thrice(tissue_mask)

    annotation_mask = skimage.io.imread(annotation_path)
    annotation_mask = skimage.transform.downscale_local_mean(annotation_mask, (3, 3)).astype(np.uint8)

    tumor_mask = tissue_mask * (annotation_mask == TUMOR_LABEL) > 0

    tumor_coords = sample_tissue_pixels(tumor_mask, max_count)

    for w, h in tumor_coords:
        shift_w, shift_h = w * down_scale * 3, h * down_scale * 3
        img = slide.read_region((shift_w, shift_h), 0, (3 * down_scale, 3 * down_scale)).convert('RGB')
        image_name = os.path.splitext(image_filename)[0]
        name = PATCH_NAME_FORMAT.format(image_name=image_name,
                                        crop_type='tumor',
                                        x=shift_h,
                                        y=shift_w,
                                        w=3 * down_scale,
                                        h=3 * down_scale)
        out_path = os.path.join(output_dir, name)
        img.save(out_path)


def process_all_images(image_dir, masks_dir, image_filenames, output_dir):
    for image_filename in tqdm(image_filenames):
        image_path = os.path.join(image_dir, image_filename)
        annotation_path = os.path.join(masks_dir, os.path.splitext(image_filename)[0] + '.png')
        sample_patches_from_tumor_image(image_filename, image_path, annotation_path, MAX_PATCHES_PER_IMAGE, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--tumor_train_dir",
                        type=str,
                        default='/data/camelyon16_original/training/tumor',
                        help='path to train tumor WSI')
    parser.add_argument("--test_dir",
                        type=str,
                        default='/data/camelyon16_original/testing/images',
                        help='path to test WSI')
    parser.add_argument("--test_reference_path",
                        type=str,
                        default='/data/camelyon16_original/testing/reference.csv',
                        help='path to references.csv file containing information (normal/tumor) about test samples')
    parser.add_argument("--train_masks_dir",
                        type=str,
                        default='/data/camelyon16/masks/train',
                        help='directory with tumor masks')
    parser.add_argument("--test_masks_dir",
                        type=str,
                        default='/data/camelyon16/masks/test',
                        help='directory with tumor masks')

    parser.add_argument("--output_tumor_patches_train_dir",
                        type=str,
                        default='/data/camelyon16/train/tumor_patches_x40',
                        help='directory for saving train tumor patches')
    parser.add_argument("--output_tumor_patches_test_dir",
                        type=str,
                        default='/data/camelyon16/test/tumor_patches_x40',
                        help='directory for saving test tumor patches')
    args = parser.parse_args()

    # Process train WSI's

    print("Starting to sample 768x768 patches from the train tumor images ... ")

    train_filenames = os.listdir(args.tumor_train_dir)

    os.makedirs(args.output_tumor_patches_train_dir, exist_ok=True)
    process_all_images(args.tumor_train_dir,args. train_masks_dir, train_filenames, args.output_tumor_patches_train_dir)
    print("Done!")

    # Process test WSI's

    print("Starting to sample 768x768 patches from test tumor images ... ")

    test_info = pd.read_csv(args.test_reference_path, usecols=[0, 1], names=['filename', 'type'])
    test_tumor_filenames = test_info[test_info['type'] == 'Tumor']['filename'].tolist()
    test_tumor_filenames = [filename + '.tif' for filename in test_tumor_filenames]

    # delete file test_114 (according to dataset README this file "does not have exhaustive annotations")
    test_tumor_filenames.remove('test_114.tif')

    os.makedirs(args.output_tumor_patches_test_dir, exist_ok=True)
    process_all_images(args.test_dir, args.test_masks_dir, test_tumor_filenames, args.output_tumor_patches_test_dir)

    print('Done!')
