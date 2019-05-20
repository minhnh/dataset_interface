#!/usr/bin/env python3
import argparse
import os
from dataset_interface.utils import TerminalColors, prompt_for_float, prompt_for_yes_or_no
from dataset_interface.augmentation.image_augmentation import ImageAugmenter


def generate_masks_and_annotations(data_dir, class_annotation_file, output_dir, output_annotation_dir, display_boxes):
    augmenter = ImageAugmenter(data_dir, class_annotation_file)
    if not output_dir:
        output_dir = os.path.join(data_dir, 'synthetic_images')
    if not output_annotation_dir:
        output_annotation_dir = os.path.join(data_dir, 'annotations')
    TerminalColors.formatted_print("begin generating images under '{}' and annotation files under '{}'"
                                   .format(output_dir, output_annotation_dir), TerminalColors.OKBLUE)
    if not os.path.isdir(output_dir):
        print("creating directory: " + output_dir)
        os.mkdir(output_dir)
    if not os.path.isdir(output_annotation_dir):
        print("creating directory: " + output_annotation_dir)
        os.mkdir(output_annotation_dir)

    while True:
        # ask user for split name, number of images to generate per background, and maximum
        # number of objects per background
        split_name = None
        while not split_name:
            split_name = input("please enter split name (e.g. 'go2019_train'): ")

        num_image_per_bg = -1
        while num_image_per_bg < 0:
            num_image_per_bg = int(prompt_for_float("please enter the number of images to be generated"
                                                    " for each background"))

        max_obj_num_per_bg = -1
        max_obj_num_per_bg = int(prompt_for_float("please enter the maximum number of objects to be projected"
                                                  " onto each background"))

        # generate images
        augmenter.generate_detection_data(split_name, output_dir, output_annotation_dir, num_image_per_bg,
                                          max_obj_num_per_bg, display_boxes=display_boxes)

        if not prompt_for_yes_or_no("do you want to generate images for another dataset split?"):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Script to generate training images and annotations for bounding box based object detection."
                    " This is done by randomly projecting segemented object pixels onto backgrounds, then"
                    " calculating the corresponding bounding boxes.")
    parser.add_argument('--data-directory', '-d', required=True,
                        help='directory where the script will look for images, backgrounds and saved object masks')
    parser.add_argument('--class-annotations', '-c', required=True,
                        help='file containing mappings from class ID to class name')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='(optional) directory to store generated images')
    parser.add_argument('--output-annotation-dir', '-a', default=None,
                        help='(optional) directory to store the generated YAML annotations')
    parser.add_argument('--display-boxes', '-b', action='store_true',
                        help='(optional) whether to display the synthetic images with visualized bounding boxes')
    args = parser.parse_args()

    try:
        generate_masks_and_annotations(args.data_directory, args.class_annotations, args.output_dir,
                                       args.output_annotation_dir, args.display_boxes)
        TerminalColors.formatted_print('image and annotation generation complete', TerminalColors.OKGREEN)
    except KeyboardInterrupt:
        TerminalColors.formatted_print('\nscript interrupted', TerminalColors.WARNING)
    except Exception as e:
        TerminalColors.formatted_print(e, TerminalColors.FAIL)
        raise
