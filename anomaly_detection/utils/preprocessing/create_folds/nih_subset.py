import os
import shutil
import argparse


def _save_split(src_path, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    shutil.copy(src_path, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subset_split_root",
                        required=True,
                        type=str,
                        help='subset_split_root')
    parser.add_argument("-o", "--output_root",
                        required=True,
                        type=str,
                        help='output_root')
    parser.add_argument("-n", "--n_folds",
                        type=int,
                        default=3,
                        help='n_folds')

    args = parser.parse_args()

    "===================== CREATE K-FOLD CROSS-VALIDATION SPLIT ========================"

    folds_dir = os.path.join(args.output_root, 'folds', 'nih')
    os.makedirs(folds_dir, exist_ok=True)

    # fake splits (all splits are the same)
    for i_fold in range(args.n_folds):
        _save_split(os.path.join(args.subset_split_root, 'normal_train.txt'),
                    os.path.join(folds_dir,  'subset', str(i_fold), 'normal', 'train'))
        _save_split(os.path.join(args.subset_split_root, 'normal_val.txt'),
                    os.path.join(folds_dir, 'subset', str(i_fold), 'normal', 'val'))
        _save_split(os.path.join(args.subset_split_root, 'normal_val.txt'),
                    os.path.join(folds_dir,  'subset', str(i_fold), 'normal', 'test'))
        _save_split(os.path.join(args.subset_split_root, 'abnormal_val.txt'),
                    os.path.join(folds_dir, 'subset', str(i_fold), 'anomaly', 'test'))
