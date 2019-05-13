#!/usr/bin/env python3
"""Convert robocup dataset to TFRecord for object_detection"""
import os
import yaml
import argparse
import contextlib2
from dataset_interface.tf_utils import create_bbox_detection_tf_example, open_sharded_output_tfrecords, tfrecords_exist
from dataset_interface.utils import TerminalColors, prompt_for_yes_or_no


def create_tf_record(annotations_file, classes_filename, output_path, image_dir, num_shards):

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        # Load annotations file
        with open(annotations_file, 'r') as annotations_f:
            annotations = yaml.load(annotations_f, Loader=yaml.FullLoader)

        # Load class file
        with open(classes_filename, 'r') as yml_file:
            classes_dict = yaml.load(yml_file, Loader=yaml.FullLoader)

        print('Number of classes ', len(classes_dict))
        print('Number of annotations ', len(annotations))
        for idx, example in enumerate(annotations):
            output_shard_index = idx % num_shards
            print('Generating tf example for image {} of {} images'.format(idx + 1, len(annotations)))

            image_path = example['image_name']
            if image_dir:
                # prepend image directory if specified
                image_path = os.path.join(image_dir, image_path)

            try:
                tf_example = create_bbox_detection_tf_example(image_path, example, classes_dict)
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

    create_record = True
    if tfrecords_exist(args.output_file):
        create_record = prompt_for_yes_or_no("shards for '{}' exists, overwrite?".format(args.output_file))

    if create_record:
        create_tf_record(args.annotation_file, args.class_file, args.output_file, args.image_dir, args.num_shards)
    else:
        print('Not creating TFRecord, exiting.')
