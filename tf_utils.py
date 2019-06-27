import glob
import io
import os
import hashlib
import PIL.Image as pil
import numpy as np
import tensorflow as tf
from dataset_interface.utils import is_box_valid, split_path


def int64_feature(value):
    """taken directly from tensorflow/models repo, file research/object_detection/utils/dataset_util.py"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """taken directly from tensorflow/models repo, file research/object_detection/utils/dataset_util.py"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """taken directly from tensorflow/models repo, file research/object_detection/utils/dataset_util.py"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    """taken directly from tensorflow/models repo, file research/object_detection/utils/dataset_util.py"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """
    taken from tensorflow/models, file research/object_detection/dataset_tools/tf_record_creation_util.py

    Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
        exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
        base_path: The base path for all shards
        num_shards: The number of shards
    Returns:
        The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def tfrecords_exist(base_bath):
    """check for TFRecord shards as created by open_sharded_output_tfrecords()"""
    matchs = glob.glob(base_bath + '-*-of-*')
    return len(matchs) > 0


def create_bbox_detection_tf_example(image_path, image_annotations, class_dict):
    if not os.path.exists(image_path):
        raise RuntimeError('image does not exist: ' + image_path)

    # file name handling
    _, _, image_extension = split_path(image_path)

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # extract image dimensions
    encoded_image_io = io.BytesIO(encoded_image_data)
    pil_image = pil.open(encoded_image_io)
    pil_image = np.asarray(pil_image)
    height, width, num_channel = pil_image.shape
    if num_channel != 3:
        raise RuntimeError("image '{}' has shape '{}' - unexpected number of channels: {}"
                           .format(image_path, pil_image.shape, num_channel))
    width = int(width)
    height = int(height)

    # hash image TODO(minhnh) confirm
    image_hash = hashlib.sha256(encoded_image_data).hexdigest()

    # create expected bounding box annotations
    xmins = []          # List of normalized left x coordinates for each box
    xmaxs = []          # List of normalized right x ...
    ymins = []          # List of normalized top y ...
    ymaxs = []          # List of normalized bottom y ...
    class_names = []    # List of string class names for each box
    class_ids = []      # List of integer class id's for each box
    invalid_box_messages = []   # contain one error message for each invalid bounding box found

    objects = image_annotations['objects']

    for obj_box in objects:
        cls_id = obj_box['class_id']

        # normalize box vertices
        x_min_norm = float(obj_box['xmin']) / width
        x_max_norm = float(obj_box['xmax']) / width
        y_min_norm = float(obj_box['ymin']) / height
        y_max_norm = float(obj_box['ymax']) / height

        # check for invalid box
        if not is_box_valid(x_min_norm, y_min_norm, x_max_norm, y_max_norm, 1.0, 1.0):
            invalid_box_messages.append("  Object ID: {}; ".format(cls_id) +
                                        "normalized box (xmin, xmax, ymin, ymax): ({:.3f}, {:.3f}, {:.3f}, {:.3f})."
                                        .format(x_min_norm, x_max_norm, y_min_norm, y_max_norm))
            continue

        # add to annotation lists
        class_ids.append(cls_id)
        class_names.append(class_dict[cls_id].encode('utf8'))
        xmins.append(x_min_norm)
        xmaxs.append(x_max_norm)
        ymins.append(y_min_norm)
        ymaxs.append(y_max_norm)

    if invalid_box_messages:
        err_msg = "=====\nInvalid box(es) for image '{}':\n".format(image_path) + '\n'.join(invalid_box_messages)
        raise RuntimeError(err_msg)

    # create tf example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(image_path.encode('utf8')),
        'image/source_id': bytes_feature(image_path.encode('utf8')),
        'image/key/sha256': bytes_feature(image_hash.encode('utf8')),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature(image_extension.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(class_names),
        'image/object/class/label': int64_list_feature(class_ids),
    }))
    return tf_example
