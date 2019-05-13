#!/usr/bin/env python3
"""
Script to segment objects from background for data augmentation using the green box in b-it-bots@Home.

Available methods:
1. bg_sub(default): simple background subtraction
"""
import argparse
import os
import sys
import yaml
from dataset_interface.utils import RawDescriptionAndDefaultsFormatter, case_insensitive_glob, TerminalColors


ALLOWED_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']


def handle_image_folder(segmentation_method, data_dir, class_annotations):
    """
    segment objects from backgrounds in object images and save the generated masks
    - will look for object images in <data_dir>/images/<object_name>
    - <object_name> comes from <class_annotations> file
    - will write object masks to <data_dir>/object_masks/<object_name>
    """
    if not os.path.exists(data_dir):
        TerminalColors.formatted_print("data directory does not exist: " + data_dir, TerminalColors.FAIL)
        return

    if not os.path.exists(class_annotations):
        TerminalColors.formatted_print("class annotation YAML does not exist: " + class_annotations,
                                       TerminalColors.FAIL)
        return

    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        TerminalColors.formatted_print("image directory does not exist: " + image_dir, TerminalColors.FAIL)
        return

    # load object classes
    with open(class_annotations) as infile:
        class_dict = yaml.load(infile, Loader=yaml.FullLoader)

    for class_name in class_dict.values():
        obj_img_path = os.path.join(image_dir, class_name)

        # warn if object directory does not exist
        if not os.path.exists(obj_img_path):
            TerminalColors.formatted_print("skipping class '{}': directory '{}' does not exist"
                                           .format(class_name, obj_img_path), TerminalColors.WARNING)
            continue

        # glob all supported image files
        image_paths = []
        for extension in ALLOWED_IMAGE_EXTENSIONS:
            image_paths.extend(case_insensitive_glob(os.path.join(obj_img_path, extension)))
        if class_name == 'test':
            print(image_paths)

        # warn if no image found
        if not image_paths:
            TerminalColors.formatted_print("skipping class '{}': directory '{}' does not contain any supported images"
                                           .format(class_name, obj_img_path), TerminalColors.WARNING)
            continue

        # perform segmentation

    print(segmentation_method, data_dir, class_annotations)


if __name__ == '__main__':
    # Note: using builtin var '__doc__' as script description
    parser = argparse.ArgumentParser(formatter_class=RawDescriptionAndDefaultsFormatter, description=__doc__)

    parser.add_argument('--method', '-m', choices=['bg_sub'], default='bg_sub',
                        help='background segmentation method')
    parser.add_argument('--image-source', '-s', choices=['folder', 'image_topic'], default='folder',
                        help="where to get the images from, either 'sensor_msgs/Image' ROS topic or a system folder")
    parser.add_argument('--data-directory', '-d', required=True,
                        help='directory where the script will look for images, backgrounds and save object masks')
    parser.add_argument('--class-annotations', '-c', required=True,
                        help='file containing mapping from class ID to class name')
    args = parser.parse_args()

    if args.image_source == 'folder':
        handle_image_folder(args.method, args.data_directory, args.class_annotations)
    else:
        TerminalColors.formatted_print('image source not supported: ' + args.image_source, TerminalColors.FAIL)
