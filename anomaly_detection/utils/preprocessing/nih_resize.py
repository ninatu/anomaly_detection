import pandas as pd
import os
import PIL.Image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path_to_data_entry", type=str,
                        default='./data/data/nih/Data_Entry_2017.csv', help='path_to_data_entry')
    parser.add_argument("--input_image_root", type=str,
                        default='./data/data/nih/', help='input_image_root')
    parser.add_argument("--output_root", type=str,
                        default='./data/data/nih_300/', help='output_root')
    parser.print_help()
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    annotation = pd.read_csv(args.path_to_data_entry)
    image_names = annotation['Image Index'].tolist()

    for image_name in image_names:
        image_path = os.path.join(args.input_image_root, image_name)
        image = PIL.Image.open(image_path)
        image = image.convert('L')
        image = image.resize((300, 300), PIL.Image.BILINEAR)

        output_path = os.path.join(args.output_root, image_name)
        image.save(output_path)
