import skimage.io
import os
from tqdm import tqdm
import sys
import argparse

sys.path.append('./')

from stain_normalization.normalizer import VahadaneNormalizer


def process_all_files(input_dir, output_dir):
    image_filenames = os.listdir(input_dir)

    # find files that we have already processed
    processed_image_filenames = os.listdir(output_dir)
    not_processed_image_filenames = list(set(image_filenames).difference(processed_image_filenames))

    for image_filename in tqdm(not_processed_image_filenames):
        image = skimage.io.imread(os.path.join(input_dir, image_filename))
        norm_image = normalizer.transform(image)
        skimage.io.imsave(os.path.join(output_dir, image_filename), norm_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--normal_patches_train_dir",
                        default='/data/camelyon16/train/normal_patches_x40')
    parser.add_argument("--normal_patches_test_dir",
                        default='/data/camelyon16/test/normal_patches_x40')
    parser.add_argument("--tumor_patches_train_dir",
                        default='/data/camelyon16/train/tumor_patches_x40')
    parser.add_argument("--tumor_patches_test_dir",
                        default='/data/camelyon16/test/tumor_patches_x40')

    parser.add_argument("--output_normal_patches_train_dir",
                        default='/data/camelyon16/train/normal_patches_x40_vah_norm')
    parser.add_argument("--output_normal_patches_test_dir",
                        default='/data/camelyon16/test/normal_patches_x40_vah_norm')
    parser.add_argument("--output_tumor_patches_train_dir",
                        default='/data/camelyon16/train/tumor_patches_x40_vah_norm')
    parser.add_argument("--output_tumor_patches_test_dir",
                        default='/data/camelyon16/test/tumor_patches_x40_vah_norm')

    parser.add_argument("--stain_target_image_path",
                        type=str,
                        default='/data/camelyon16/stain_target_image.jpg',
                        help='path to the target image (a source for stain normalization)')

    args = parser.parse_args()

    stain_target_img = skimage.io.imread(args.stain_target_image_path)
    normalizer = VahadaneNormalizer()
    normalizer.fit(stain_target_img)

    for input_dir, output_dir, name in [
        (args.normal_patches_train_dir, args.output_normal_patches_train_dir, 'normal (train)'),
        (args.normal_patches_test_dir, args.output_normal_patches_test_dir, 'normal (test)'),
        (args.tumor_patches_train_dir, args.output_tumor_patches_train_dir, 'tumor (train)'),
        (args.tumor_patches_test_dir, args.output_tumor_patches_test_dir, 'tumor (test)'),
    ]:
        print("Starting stain normalization of {} images".format(name))

        os.makedirs(output_dir, exist_ok=True)
        process_all_files(input_dir, output_dir)

        print("Done!")
