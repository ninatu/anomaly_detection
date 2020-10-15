import os
import pandas as pd
import argparse


def _save_split(filenames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f_out:
        f_out.writelines([filename + '\n' for filename in filenames])


def save_split_for_view(output_root, train_annotation, test_annotation, view):
    train_annotation = train_annotation[train_annotation['View Position'] == view]
    test_annotation = test_annotation[test_annotation['View Position'] == view]

    train_normal_image_names = train_annotation[train_annotation['Finding Labels'] == 'No Finding']['Image Index'].values
    test_normal_image_names = test_annotation[test_annotation['Finding Labels'] == 'No Finding']['Image Index'].values
    test_anomaly_image_names = test_annotation[test_annotation['Finding Labels'] != 'No Finding']['Image Index'].values

    _save_split(train_normal_image_names, os.path.join(output_root, view, 'normal', 'train'))
    _save_split(test_normal_image_names, os.path.join(output_root, view, 'normal', 'test'))
    _save_split(test_anomaly_image_names, os.path.join(output_root, view, 'anomaly', 'test'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path_to_data_entry",
                        type=str,
                        default='./data/data/nih/Data_Entry_2017.csv',
                        help='path_to_data_entry')
    parser.add_argument("--path_to_train_val_list",
                        type=str,
                        default='./data/data/nih/train_val_list.txt',
                        help='path_to_train_val_list')
    parser.add_argument("--path_to_test_list",
                        type=str,
                        default='./data/data/nih/test_list.txt',
                        help='path_to_test_list')
    parser.add_argument( "--output_root",
                         required=True,
                         type=str)

    args = parser.parse_args()

    annotation = pd.read_csv(args.path_to_data_entry)

    with open(args.path_to_train_val_list) as fin:
        train_image_names = list(map(lambda x: x.strip(), fin.readlines()))

    with open(args.path_to_test_list) as fin:
        test_image_names = list(map(lambda x: x.strip(), fin.readlines()))

    train_annotation = annotation[annotation['Image Index'].isin(train_image_names)]
    test_annotation = annotation[annotation['Image Index'].isin(test_image_names)]

    save_split_for_view(args.output_root,train_annotation, test_annotation, 'PA')
    save_split_for_view(args.output_root,train_annotation, test_annotation, 'AP')
