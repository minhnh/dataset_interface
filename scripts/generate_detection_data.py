#!/usr/bin/env python3
import argparse
import os
from dataset_interface.utils import TerminalColors, prompt_for_float, prompt_for_yes_or_no
from dataset_interface.augmentation.image_augmentation import ImageAugmenter


def generate_masks_and_annotations(data_dir, background_dir, class_annotation_file,
                                   num_objects_per_class, output_dir, output_annotation_dir,
                                   invert_mask, prob_rand_trans):
    augmenter = ImageAugmenter(data_dir, background_dir, class_annotation_file, num_objects_per_class)
    if not output_dir:
        output_dir = os.path.join(data_dir, 'synthetic_images')
    else:
        output_dir = os.path.join(output_dir,'synthetic_images')
    output_dir_images = os.path.join(output_dir, 'images')
    output_dir_masks = os.path.join(output_dir, 'masks')

    if not output_annotation_dir:
        output_annotation_dir = os.path.join(data_dir, 'annotations')
    else:
        output_annotation_dir = os.path.join(output_annotation_dir, 'annotations')

    TerminalColors.formatted_print("begin generating images under '{}' and annotation files under '{}'"
                                   .format(output_dir, output_annotation_dir), TerminalColors.OKBLUE)
    if not os.path.isdir(output_dir):
        print("creating directory: " + output_dir)
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir_images):
        print("creating directory: " + output_dir_images)
        os.mkdir(output_dir_images)
    if not os.path.isdir(output_dir_masks):
        print("creating directory: " + output_dir_masks)
        os.mkdir(output_dir_masks)

    if not os.path.isdir(output_annotation_dir):
        print("creating directory: " + output_annotation_dir)
        os.mkdir(output_annotation_dir)

    while True:
        # ask user for split name, number of images to generate per background, and maximum
        # number of objects per background
        split_name = None
        while not split_name:
            split_name = input("please enter split name (e.g. 'go2019_train'): ")

        # num_image_per_bg = -1
        # while num_image_per_bg < 0:
        num_images_per_bg = int(prompt_for_float("please enter the number of images to be generated"
                                                " for each background"))

        max_obj_num_per_bg = int(prompt_for_float("enter the maximum number of objects per background"))
        # generate images
        augmenter.generate_detection_data(split_name, output_dir_images, output_dir_masks,
                                          output_annotation_dir, max_obj_num_per_bg,
                                          invert_mask=invert_mask,
                                          num_images_per_bg=num_images_per_bg,
                                          prob_rand_trans=prob_rand_trans)

        if not prompt_for_yes_or_no("do you want to generate images for another dataset split?"):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Script to generate training images and annotations for"
                    " bounding box based object detection. This is done by"
                    " randomly projecting segmented object pixels onto backgrounds,"
                    " then calculating the corresponding bounding boxes.")
    parser.add_argument('--data-directory', '-d', required=True,
                        help='directory where the script will look for images and object masks')
    parser.add_argument('--background-directory', '-bg', required=True,
                        help='directory where the script will look for backgrounds images')
    parser.add_argument('--class-annotations', '-c', required=True,
                        help='file containing mappings from class ID to class name')
    parser.add_argument('--num-objects-per-class', '-n', required=True, type=int,
                        help='number of object per class')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='(optional) directory to store generated images')
    parser.add_argument('--output-annotation-dir', '-a', default=None,
                        help='(optional) directory to store the generated YAML annotations')
    parser.add_argument('--invert_mask', '-im', action='store_true',
                        help='whether to invert the mask (Required for YCB)')
    parser.add_argument('--prob_rand_trans', '-pt', required=True, type=float,
                        help='probability of a random transformation (1.0 == No transformation applied)')
    args = parser.parse_args()

    try:
        generate_masks_and_annotations(args.data_directory, args.background_directory, args.class_annotations,
                                       args.num_objects_per_class, args.output_dir,
                                       args.output_annotation_dir, args.invert_mask, args.prob_rand_trans)
        TerminalColors.formatted_print('image and annotation generation complete', TerminalColors.OKGREEN)
    except KeyboardInterrupt:
        TerminalColors.formatted_print('\nscript interrupted', TerminalColors.WARNING)
    except Exception as e:
        TerminalColors.formatted_print(e, TerminalColors.FAIL)
        raise
