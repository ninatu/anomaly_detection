import xml.etree.ElementTree
import numpy as np
import json
import os
from tqdm import tqdm
import argparse


def convert_camelyon_annotation_to_json(input_path, output_path):
    root = xml.etree.ElementTree.parse(input_path).getroot()
    annotations_tumor = \
        root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
    annotations_0 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
    annotations_1 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
    annotations_2 = \
        root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
    annotations_tumor = \
        annotations_tumor + annotations_0 + annotations_1
    annotations_normal = annotations_2

    json_dict = {}
    json_dict['tumor'] = []
    json_dict['normal'] = []

    for annotation in annotations_tumor:
        X = list(map(lambda x: float(x.get('X')),
                 annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')),
                 annotation.findall('./Coordinates/Coordinate')))
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        name = annotation.attrib['Name']
        json_dict['tumor'].append({'name': name, 'vertices': vertices})

    for annotation in annotations_normal:
        X = list(map(lambda x: float(x.get('X')),
                 annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')),
                 annotation.findall('./Coordinates/Coordinate')))
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        name = annotation.attrib['Name']
        json_dict['normal'].append({'name': name, 'vertices': vertices})

    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=1)


def convert_all_files(input_dir, output_dir):
    filenames = os.listdir(input_dir)
    for filename in tqdm(filenames):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
        convert_camelyon_annotation_to_json(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_train_xml_dir",
                        type=str,
                        default='/data/camelyon16_original/training/lesion_annotations')
    parser.add_argument("--input_test_xml_dir",
                        type=str,
                        default='/data/camelyon16_original/testing/lesion_annotations')
    parser.add_argument("--output_train_json_dir",
                        type=str,
                        default='/data/camelyon16/annotation/train')
    parser.add_argument("--output_test_json_dir",
                        type=str,
                        default='/data/camelyon16/annotation/test')

    args = parser.parse_args()

    print("Starting annotation conversion for the train split ... ")
    os.makedirs(args.output_train_json_dir, exist_ok=True)
    convert_all_files(args.input_train_xml_dir, args.output_train_json_dir)
    print('Done!')

    print("Start of annotation conversion for test split ... ")
    os.makedirs(args.output_test_json_dir, exist_ok=True)
    convert_all_files(args.input_test_xml_dir, args.output_test_json_dir)
    print('Done!')

