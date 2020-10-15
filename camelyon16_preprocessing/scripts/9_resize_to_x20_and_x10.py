# path to target image (source for stain normalization)

# paths to normalized patches
NORMED_NORMAL_PATCHES_TRAIN_DIR = '/data_hdd1/ninatu/data/camelyon16/miccai/train/normal_patches_x40_vah_norm'
NORMED_NORMAL_PATCHES_TEST_DIR = '/data_hdd1/ninatu/data/camelyon16/miccai/test/normal_patches_x40_vah_norm'
NORMED_TUMOR_PATCHES_TRAIN_DIR = '/data_hdd1/ninatu/data/camelyon16/miccai/train/tumor_patches_x40_vah_norm'
NORMED_TUMOR_PATCHES_TEST_DIR = '/data_hdd1/ninatu/data/camelyon16/miccai/test/tumor_patches_x40_vah_norm'

# paths to not normalized patches
OUTPUT_NORMED_NORMAL_PATCHES_TRAIN_DIR_X20 = '/data_hdd1/ninatu/data/camelyon16/miccai/train/normal_patches_x20_vah_norm'
OUTPUT_NORMED_NORMAL_PATCHES_TEST_DIR_X20 = '/data_hdd1/ninatu/data/camelyon16/miccai/test/normal_patches_x20_vah_norm'
OUTPUT_NORMED_TUMOR_PATCHES_TRAIN_DIR_X20 = '/data_hdd1/ninatu/data/camelyon16/miccai/train/tumor_patches_x20_vah_norm'
OUTPUT_NORMED_TUMOR_PATCHES_TEST_DIR_X20 = '/data_hdd1/ninatu/data/camelyon16/miccai/test/tumor_patches_x20_vah_norm'


import skimage.io
import skimage.transform
import os
from tqdm import tqdm
import argparse


def process_all_files(input_dir, output_dir_x20, output_dir_x10):
    for image_filename in tqdm(os.listdir(input_dir)):
        image = skimage.io.imread(os.path.join(input_dir, image_filename))

        image_x20 = skimage.transform.rescale(image, (0.5, 0.5, 1))
        image_x20 = skimage.img_as_ubyte(image_x20)
        skimage.io.imsave(os.path.join(output_dir_x20, image_filename), image_x20)

        image_x10 = skimage.transform.rescale(image, (0.25, 0.25, 1))
        image_x10 = skimage.img_as_ubyte(image_x10)
        skimage.io.imsave(os.path.join(output_dir_x10, image_filename), image_x10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--image_dir_x40",
                        default='/data/camelyon16/x40')
    parser.add_argument("--output_dir_x20",
                        default='/data/camelyon16/x20')
    parser.add_argument("--output_dir_x10",
                        default='/data/camelyon16/x10')
    args = parser.parse_args()

    print("Starting the resizing process..")

    os.makedirs(args.output_dir_x20, exist_ok=True)
    os.makedirs(args.output_dir_x10, exist_ok=True)
    process_all_files(args.image_dir_x40, args.output_dir_x20, args.output_dir_x10)

    print("Done!")
