#!/usr/bin/env python3
"""Convert robocup dataset to TFRecord for object_detection"""
import os
import yaml
import argparse
import contextlib2
from dataset_interface.tf_utils import create_bbox_detection_tf_example, open_sharded_output_tfrecords, tfrecords_exist
from dataset_interface.utils import TerminalColors, prompt_for_yes_or_no, print_progress


def create_tf_record(image_annotation_path, class_annotation_path, output_path, image_dir, num_shards):
    if tfrecords_exist(output_path) \
            and not prompt_for_yes_or_no("shards for '{}' exists, overwrite?".format(args.output_file)):
        # make sure user want to overwrite TFRecord
        TerminalColors.formatted_print('not overwriting TFRecord, exiting..', TerminalColors.WARNING)
        return

    # Load class file
    if not os.path.exists(class_annotation_path):
        TerminalColors.formatted_print('class annotation file does not exist: ' + class_annotation_path,
                                       TerminalColors.FAIL)
        return
    with open(class_annotation_path, 'r') as yml_file:
        class_dict = yaml.load(yml_file, Loader=yaml.FullLoader)
    TerminalColors.formatted_print("\nfound '{}' classes in file '{}'".format(len(class_dict), class_annotation_path),
                                   TerminalColors.OKBLUE)

    # Load annotations file
    if not os.path.exists(image_annotation_path):
        TerminalColors.formatted_print('image annotation file does not exist: ' + image_annotation_path,
                                       TerminalColors.FAIL)
        return
    with open(image_annotation_path, 'r') as annotations_f:
        annotations = yaml.load(annotations_f, Loader=yaml.FullLoader)
    num_annotations = len(annotations)
    TerminalColors.formatted_print("found '{}' image annotations in file '{}'"
                                   .format(num_annotations, image_annotation_path), TerminalColors.OKBLUE)

    TerminalColors.formatted_print('number of TFRecord shards: {}\n'.format(num_shards),
                                   TerminalColors.OKBLUE)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, output_path, num_shards)

        for idx, example in enumerate(annotations):
            output_shard_index = idx % num_shards
            print_progress(idx + 1, num_annotations, prefix="Generating TFRecord for image ")

            image_path = example['image_name']
            if image_dir:
                # prepend image directory if specified
                image_path = os.path.join(image_dir, image_path)

            try:
                tf_example = create_bbox_detection_tf_example(image_path, example, class_dict)
            except RuntimeError as e:
                TerminalColors.formatted_print(str(e), TerminalColors.FAIL)
                continue

            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to generate TFRecord (optionally sharded) from images and"
                                                 " bounding box annotations.")
    parser.add_argument('annotation_file', help="YAML file containing image location and bounding box annotations")
    parser.add_argument('class_file', help="YAML file containing the mapping from class ID to class name")
    parser.add_argument('output_file', help="path where TFRecord's should be written to,"
                                            " e.g. './robocup_train.record'")
    parser.add_argument('--image_dir', '-d', default=None,
                        help="if specified, will prepend to image paths in annotation file")
    parser.add_argument('--num_shards', '-n', default=1, help="number of fragments to split the TFRecord into")
    args = parser.parse_args()

    create_tf_record(args.annotation_file, args.class_file, args.output_file, args.image_dir, args.num_shards)
