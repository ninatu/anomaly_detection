import os
import numpy as np
import cv2
import skimage.io
import openslide
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tqdm import tqdm
import argparse
import sys

sys.path.append('./')

from utils import PATCH_NAME_PAT, TUMOR_LABEL, NORMAL_LABEL, \
    preprocess_normal_mask_decrease_thrice, preprocess_tumor_mask_decrease_thrice, \
    get_tissue_mask, get_tissue_bb


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


def plot_crops(downscaled_image, image_name, patches_dir):
    LEVEL = 8
    downscale = 2 ** LEVEL

    marked_down_image = downscaled_image.copy()
    for patch_name in os.listdir(patches_dir):
        match = PATCH_NAME_PAT.match(patch_name)
        if match['image_name'] == image_name:
            x = int(match['x'])
            y = int(match['y'])

            x = int(x / downscale)
            y = int(y / downscale)
            if match['crop_type'] == 'norm':
                marked_down_image[x:x + 3, y:y + 3] = (0, 255, 0)
            else:
                marked_down_image[x:x + 3, y:y + 3] = (255, 0, 0)

    return marked_down_image


def save_visualization(image_dir, image_filenames, pathes_dir, output_dir, is_normal, masks_dir=None):
    for image_filename in tqdm(image_filenames):
        image_path = os.path.join(image_dir, image_filename)
        image_name = os.path.splitext(image_filename)[0]

        slide = openslide.OpenSlide(image_path)
        LEVEL = 8
        downscale = 2 ** LEVEL
        slide_w, slide_h = slide.dimensions
        down_image = np.array(
            slide.read_region((0, 0), LEVEL, (slide_w // downscale, slide_h // downscale)).convert('RGB'))
        tissue_mask = np.array(get_tissue_mask(down_image))
        marked_down_image = plot_crops(down_image, image_name, pathes_dir)

        if is_normal:
            tissue_mask = preprocess_normal_mask_decrease_thrice(tissue_mask)
            tissue_mask = skimage.transform.rescale(tissue_mask, (3, 3), order=0) > 0
            h, w, _ = down_image.shape
            tissue_mask = tissue_mask[:h, :w]

            x, y, w, h = get_tissue_bb(tissue_mask.astype(np.uint8))
            rect_img = cv2.rectangle(down_image.copy(), (x, y), (x + w, y + h), (0, 0, 255), 3)

            tissue_mask = np.stack([np.zeros(tissue_mask.shape), tissue_mask, np.zeros(tissue_mask.shape)], axis=2)
        else:
            tissue_mask = preprocess_tumor_mask_decrease_thrice(tissue_mask)
            tissue_mask = skimage.transform.rescale(tissue_mask, (3, 3), order=0) > 0

            h, w, _ = down_image.shape
            tissue_mask = tissue_mask[:h, :w]

            x, y, w, h = get_tissue_bb(tissue_mask.astype(np.uint8))

            rect_img = cv2.rectangle(down_image.copy(), (x, y), (x + w, y + h), (0, 0, 255), 3)

            annotation_mask = skimage.io.imread(os.path.join(masks_dir, image_name + '.png'))

            tissue_mask = np.stack([
                annotation_mask == TUMOR_LABEL,
                tissue_mask,
                annotation_mask == NORMAL_LABEL], axis=2).astype(np.float)

        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)

        n, m = 2, 3

        ax = fig.add_subplot(n, m, 1)
        ax.imshow(down_image), ax.axis('off'), ax.set_title('Original')

        ax = fig.add_subplot(n, m, 2)

        ax.imshow(tissue_mask), ax.axis('off'), ax.set_title('Tissue mask')

        ax = fig.add_subplot(n, m, 3)

        ax.imshow(rect_img), ax.axis('off'), ax.set_title('Original')

        if w != 0 and h != 0:
            ax = fig.add_subplot(n, m, 4)
            crop = down_image[y:y + h, x: x + w]
            ax.imshow(crop), ax.axis('off'), ax.set_title('Original')

            ax = fig.add_subplot(n, m, 5)
            mask_crop = tissue_mask[y:y + h, x: x + w]
            ax.imshow(mask_crop, cmap='gray'), ax.axis('off'), ax.set_title('Tissue mask')

            ax = fig.add_subplot(n, m, 6)
            crop = np.asarray(marked_down_image)[y:y + h, x: x + w]
            ax.imshow(crop), ax.axis('off'), ax.set_title('Places of crops')
        fig.tight_layout()

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        skimage.io.imsave(os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.png'), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--normal_train_dir",
                        type=str,
                        default='/data/camelyon16_original/training/normal',
                        help='path to train normal WSI')
    parser.add_argument("--tumor_train_dir",
                        type=str,
                        default='/data/camelyon16_original/training/tumor',
                        help='path to train tumor WSI')
    parser.add_argument("--test_dir",
                        type=str,
                        default='/data/camelyon16_original/testing/images',
                        help='path to test WSI')
    parser.add_argument("--normal_patches_train_dir",
                        default='/data/camelyon16/train/normal_patches_x40')
    parser.add_argument("--normal_patches_test_dir",
                        default='/data/camelyon16/test/normal_patches_x40')
    parser.add_argument("--tumor_patches_train_dir",
                        default='/data/camelyon16/train/tumor_patches_x40')
    parser.add_argument("--tumor_patches_test_dir",
                        default='/data/camelyon16/test/tumor_patches_x40')
    parser.add_argument("--train_masks_dir",
                        type=str,
                        default='/data/camelyon16/masks/train',
                        help='directory with tumor masks')
    parser.add_argument("--test_masks_dir",
                        type=str,
                        default='/data/camelyon16/masks/test',
                        help='directory with tumor masks')
    parser.add_argument("--test_reference_path",
                        type=str,
                        default='/data/camelyon16_original/testing/reference.csv',
                        help='path to references.csv file containing information (normal/tumor) about test samples')
    parser.add_argument("--output_dir",
                        type=str,
                        default='/data/camelyon16/visualization')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # test tumor
    # delete file test_114 (according to dataset README this file "does not have exhaustive annotations")
    test_info = pd.read_csv(args.test_reference_path, usecols=[0, 1], names=['filename', 'type'])
    test_tumor_filenames = test_info[test_info['type'] == 'Tumor']['filename'].tolist()
    test_tumor_filenames = [filename + '.tif' for filename in test_tumor_filenames]
    test_tumor_filenames.remove('test_114.tif')

    save_visualization(args.test_dir, test_tumor_filenames, args.tumor_patches_test_dir, args.output_dir,
                       is_normal=False, masks_dir=args.test_masks_dir)

    # test normal

    test_info = pd.read_csv(args.test_reference_path, usecols=[0, 1], names=['filename', 'type'])
    test_normal_filenames = test_info[test_info['type'] == 'Normal']['filename'].tolist()
    test_normal_filenames = [filename + '.tif' for filename in test_normal_filenames]

    save_visualization(args.test_dir, test_normal_filenames, args.normal_patches_test_dir, args.output_dir,
                       is_normal=True)

    # train tumor

    train_tumor_filenames = os.listdir(args.tumor_train_dir)
    save_visualization(args.tumor_train_dir, train_tumor_filenames, args.tumor_patches_train_dir, args.output_dir,
                       is_normal=False, masks_dir=args.train_masks_dir)

    # train normal

    train_normal_filenames = os.listdir(args.normal_train_dir)

    for filename in EXCLUDE_TRAIN_WSI:
        train_normal_filenames.remove(filename)

    save_visualization(args.normal_train_dir, train_normal_filenames, args.normal_patches_train_dir, args.output_dir,
                       is_normal=True)
