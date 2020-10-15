import shutil
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--normal_patches_train_dir",
                        default='/data/camelyon16/train/normal_patches_x40_vah_norm',
                        help='a path to normalized patches')
    parser.add_argument("--normal_patches_test_dir",
                        default='/data/camelyon16/test/normal_patches_x40_vah_norm',
                        help='a path to normalized patches')
    parser.add_argument("--tumor_patches_train_dir",
                        default='/data/camelyon16/train/tumor_patches_x40_vah_norm',
                        help='a path to normalized patches')
    parser.add_argument("--tumor_patches_test_dir",
                        default='/data/camelyon16/test/tumor_patches_x40_vah_norm',
                        help='a path to normalized patches')
    parser.add_argument("--output_dir",
                        type=str,
                        default='/data/camelyon16/x40')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for input_dir in [args.normal_patches_train_dir, args.normal_patches_test_dir,
                      args.tumor_patches_train_dir, args.tumor_patches_test_dir]:
        for file_name in os.listdir(input_dir):
            full_file_name = os.path.join(input_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, args.output_dir)
