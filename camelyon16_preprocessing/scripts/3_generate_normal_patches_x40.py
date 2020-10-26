"""
Extract patches from normal slides
"""

import openslide
import os
from tqdm import tqdm
import pandas as pd
import argparse
import sys

sys.path.append('./')
from utils import get_tissue_mask, preprocess_normal_mask_decrease_thrice, \
    sample_tissue_pixels, PATCH_NAME_FORMAT


# maximum number of patches extracted from one WSI
MAX_PATCHES_PER_IMAGE = 50

# we excluded several WSI from training
# * normal_027.tif (a very small region of tissue,
#                   algorithm for creating a tissue mask does not work correctly in this case)
# * normal_045.tif (the same)
# * normal_108.tif (it is a completely black image)
# * normal_144.tif (suddenly background color is black instead of white,
#                   algorithm for creating a tissue mask does not work correctly)
# * normal_150.tif (it is a completely black image)
# * normal_158.tif (almost full image is blurred)

EXCLUDE_TRAIN_WSI = [
    'normal_027.tif',
    'normal_045.tif',
    'normal_108.tif',
    'normal_144.tif',
    'normal_150.tif',
    'normal_158.tif'
]


def sample_patches_from_normal_image(image_filename, image_path, max_count, output_dir):
    """
    Sampling patches from normal WSI, saving them in output_dir
    """

    slide = openslide.OpenSlide(image_path)
    LEVEL = 8
    down_scale = int(2 ** LEVEL)
    slide_w, slide_h = slide.dimensions
    down_img = slide.read_region((0, 0), LEVEL, (slide_w // down_scale, slide_h // down_scale)).convert('RGB')

    tissue_mask = get_tissue_mask(down_img)
    tissue_mask = preprocess_normal_mask_decrease_thrice(tissue_mask)

    coords = sample_tissue_pixels(tissue_mask, max_count)
    for w, h in coords:
        shift_w, shift_h = w * down_scale * 3, h * down_scale * 3
        img = slide.read_region((shift_w, shift_h), 0, (3 * down_scale, 3 * down_scale)).convert('RGB')
        image_name = os.path.splitext(image_filename)[0]
        name = PATCH_NAME_FORMAT.format(image_name=image_name,
                                        crop_type='normal',
                                        x=shift_h,
                                        y=shift_w,
                                        w=3 * down_scale,
                                        h=3 * down_scale)
        out_path = os.path.join(output_dir, name)
        img.save(out_path)


def process_all_images(image_dir, image_filenames, output_dir):
    for image_filename in tqdm(image_filenames):
        image_path = os.path.join(image_dir, image_filename)
        sample_patches_from_normal_image(image_filename, image_path, MAX_PATCHES_PER_IMAGE, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--normal_train_dir",
                        type=str,
                        default='/data/camelyon16_original/training/normal',
                        help='path to train normal WSI')
    parser.add_argument("--test_dir",
                        type=str,
                        default='/data/camelyon16_original/testing/images',
                        help='path to test WSI')
    parser.add_argument("--test_reference_path",
                        type=str,
                        default='/data/camelyon16_original/testing/reference.csv',
                        help='path to references.csv file containing information (normal/tumor) about test samples')
    parser.add_argument("--output_normal_patches_train_dir",
                        type=str,
                        default='/data/camelyon16/train/normal_patches_x40',
                        help='directory for saving train normal patches')
    parser.add_argument("--output_normal_patches_test_dir",
                        type=str,
                        default='/data/camelyon16/test/normal_patches_x40',
                        help='directory for saving test normal patches')

    args = parser.parse_args()

    # Process train WSI's

    print("Starting to sample 768x768 patches from the train normal images ... ")
    train_filenames = os.listdir(args.normal_train_dir)
    for filename in EXCLUDE_TRAIN_WSI:
        train_filenames.remove(filename)

    os.makedirs(args.output_normal_patches_train_dir, exist_ok=True)
    process_all_images(args.normal_train_dir, train_filenames, args.output_normal_patches_train_dir)
    print("Done!")

    # Process test WSI's

    print("Starting to sample 768x768 patches from the test normal images ... ")

    os.makedirs(args.output_normal_patches_test_dir, exist_ok=True)

    test_info = pd.read_csv(args.test_reference_path, usecols=[0, 1], names=['filename', 'type'])
    test_filenames = test_info[test_info['type'] == 'Normal']['filename'].tolist()
    test_filenames = [filename + '.tif' for filename in test_filenames]

    process_all_images(args.test_dir, test_filenames, args.output_normal_patches_test_dir)
    print('Done!')
