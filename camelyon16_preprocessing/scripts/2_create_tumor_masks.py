import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import openslide
import argparse
import sys

sys.path.append('./')
from utils import TUMOR_LABEL, NORMAL_LABEL


def create_mask_x40(annotation, slide_w, slide_h):
    mask = Image.new('L', (slide_w, slide_h), 0)
    for poligon in annotation['tumor']:
        vertices = poligon['vertices']
        vertices = [(x, y) for x, y in vertices]
        ImageDraw.Draw(mask).polygon(vertices, outline=TUMOR_LABEL, fill=TUMOR_LABEL)
    for poligon in annotation['normal']:
        vertices = poligon['vertices']
        vertices = [(x, y) for x, y in vertices]
        ImageDraw.Draw(mask).polygon(vertices, outline=NORMAL_LABEL, fill=NORMAL_LABEL)
    return mask


def process_all_annotations(annotation_dir, image_dir, output_dir):
    for annotation_filename in tqdm(os.listdir(annotation_dir)):
        with open(os.path.join(annotation_dir, annotation_filename)) as f:
            annotation = json.load(f)

        name = os.path.splitext(annotation_filename)[0]
        image_path = os.path.join(image_dir, '{}.tif'.format(name))
        slide = openslide.OpenSlide(image_path)
        slide_w, slide_h = slide.dimensions

        mask = create_mask_x40(annotation, slide_w, slide_h)

        # save image at level 8 (downscaling image with x40 magnification in 2^8 times)
        # (levels are numbered from 0 (highest resolution) to 8 (lowest resolution))
        LEVEL = 8
        downscale = int(2 ** LEVEL)
        mask = mask.resize((int(slide_w / downscale), int(slide_h // downscale)))
        output_path = os.path.join(output_dir, '{}.png'.format(name))
        mask.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--train_json_dir",
                        type=str,
                        default='/data/camelyon16/annotation/train',
                        help='directory containing json annotations')
    parser.add_argument("--test_json_dir",
                        type=str,
                        default='/data/camelyon16/annotation/test',
                        help='directory containing json annotations')

    parser.add_argument("--train_image_dir",
                        type=str,
                        default='/data/camelyon16_original/training/tumor',
                        help='directory containing WSI images')
    parser.add_argument("--test_image_dir",
                        type=str,
                        default='/data/camelyon16_original/testing/images',
                        help='directory containing WSI images')

    parser.add_argument("--output_train_mask_dir",
                        type=str,
                        default='/data/camelyon16/masks/train',
                        help='output directory for saving masks')
    parser.add_argument("--output_test_mask_dir",
                        type=str,
                        default='/data/camelyon16/masks/test',
                        help='output directory for saving masks')

    args = parser.parse_args()

    # Process annotations of train split

    print("Start to create tumor masks for the train split ... ")
    os.makedirs(args.output_train_mask_dir, exist_ok=True)
    process_all_annotations(args.train_json_dir, args.train_image_dir, args.output_train_mask_dir)
    print('Done!')

    # Process annotations of test split

    print("Start to create tumor masks for the test split ... ")
    os.makedirs(args.output_test_mask_dir, exist_ok=True)
    process_all_annotations(args.test_json_dir, args.test_image_dir, args.output_test_mask_dir)
    print('Done!')
