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

    parser.add_argument("--output_split_root",
                        type=str,
                        default='/data/camelyon16/train_test_split/healthy')

    args = parser.parse_args()

    for _type, train_dir, test_dir in [
        ('normal', args.normal_patches_train_dir, args.normal_patches_test_dir),
        ('anomaly', args.tumor_patches_train_dir, args.tumor_patches_test_dir)
    ]:
        output_dir = os.path.join(args.output_split_root, _type)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'train'), 'w') as fin:
            fin.writelines([filename + '\n' for filename in os.listdir(train_dir)])

        with open(os.path.join(output_dir, 'test'), 'w') as fin:
            fin.writelines([filename + '\n' for filename in os.listdir(test_dir)])
