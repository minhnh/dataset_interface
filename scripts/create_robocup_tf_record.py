"""Convert robocup dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
import os
from dataset_interface.tf_utils import create_bbox_detection_tf_example

import tensorflow as tf
import pandas as pd
import yaml
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

# It is required to run beforehand
# from models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
from object_detection.utils import dataset_util


flags = tf.app.flags
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations CSV file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations CSV file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

tf.flags.DEFINE_string('classes_filename', '', 'Classes yaml file.')
FLAGS = flags.FLAGS


def create_tf_record_from_yaml(annotations_file, image_dir, classes_filename, output_path, num_shards):

    # writer = tf.python_io.TFRecordWriter(output_path)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        # Load annotations file
        with open(annotations_file, 'r') as annotations_f:
            annotations = yaml.load(annotations_f, Loader=yaml.FullLoader)

        # Load yaml file
        with open(classes_filename, 'r') as yml_file:
            classes_dict = yaml.load(yml_file, Loader=yaml.FullLoader)

        print('Number of classes ', len(classes_dict))
        print('Number of annotations ', len(annotations))

        for idx, example in enumerate(annotations):
            output_shard_index = idx % num_shards
            print('Generating tf example for image {} of {} images'.format(idx+1, len(annotations)))

            filename = example['image_name']
            file_path = os.path.join(image_dir,filename)
            try:
                tf_example = create_bbox_detection_tf_example(file_path, example, classes_dict)
            except RuntimeError as e:
                print(e)
                continue

            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

    # writer.close()


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    assert FLAGS.classes_filename, '`classes_filename` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_output_path = os.path.join(FLAGS.output_dir, 'robocup_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'robocup_val.record')

    print('Generating validation shards ... ')

    create_tf_record_from_yaml(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        FLAGS.classes_filename,
        val_output_path,
        num_shards=10)

    print('Generating training shards ... ')

    create_tf_record_from_yaml(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        FLAGS.classes_filename,
        train_output_path,
        num_shards=20)




if __name__ == '__main__':
    tf.app.run()
