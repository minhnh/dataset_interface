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


import hashlib
import tensorflow as tf
import os
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
                img = cv2.imread(file_path)
                height = img.shape[0]
                width = img.shape[1]
            except:
                print('Failed with image ', file_path)
                continue

            with tf.gfile.GFile(file_path, 'rb') as fid:
                encoded_image_data = fid.read()

            if filename.split('.')[1] == 'jpg':
                image_format = 'jpeg'
            else:
                print('Image {} is not jpg'.format(filename ))

            key = hashlib.sha256(encoded_image_data).hexdigest()

            xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
            xmaxs = [] # List of normalized right x coordinates in bounding box
                       # (1 per box)
            ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
            ymaxs = [] # List of normalized bottom y coordinates in bounding box
                       # (1 per box)
            classes_text = [] # List of string class name of bounding box (1 per box)
            classes = [] # List of integer class id of bounding box (1 per box)

            objects = example['objects']

            for object_ in objects:
                class_id = object_['class_id']
                classes.append(class_id)
                classes_text.append(classes_dict[class_id].encode('utf8'))

                xmins.append(float(object_['xmin'])/width)
                xmaxs.append(float(object_['xmax'])/width)
                ymins.append(float(object_['ymin'])/height)
                ymaxs.append(float(object_['ymax'])/height)

                if float(object_['xmin'])/width > 1.0 or float(object_['xmax'])/width > 1.0 or \
                    float(object_['ymin'])/height > 1.0 or float(object_['ymax'])/height > 1.0:
                    print('=============================================')
                    print('Invalid bounding box')
                    print('Object: ', class_id)
                    print('Image name: ', filename)
                    print('Bounding box coordinates: \n xmin = {} xmax = {}  ymin = {} ymax = {}'
                        .format(float(object_['xmin'])/width,float(object_['xmax'])/width,float(object_['ymin'])/height,float(object_['ymax'])/height))

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))

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
