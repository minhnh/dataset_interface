#!/usr/bin/env python3
import yaml
import argparse


def yaml_classes_to_pbtxt(yaml_filename, pbtxt_filename):
    """
    Function to parse a yaml file containing id and classes into a labelMap in
    protobuf format.

    args:
    (1) yaml_filename: Name of the input yaml file
    (2) pbtxt_filename:  Name of the output protobuf file
    """
    with open(yaml_filename, 'r') as yaml_file:
        class_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

    with open(pbtxt_filename, 'w') as pbtxt_file:
        for class_id, class_name in class_dict.items():
            pbtxt_item_str = "item {\n" + \
                            "  id: {}\n".format(class_id) + \
                            "  name: '{}'\n".format(class_name) + \
                            "}\n\n"
            pbtxt_file.write(pbtxt_item_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to convert YAML class annotations to pbtxt format")
    parser.add_argument('yaml_file', help="input YAML file containing class annotations")
    parser.add_argument('pbtxt_file', help="output pbtxt file to write to")
    args = parser.parse_args()

    yaml_classes_to_pbtxt(args.yaml_file, args.pbtxt_file)
