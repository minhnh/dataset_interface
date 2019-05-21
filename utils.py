from builtins import input      # for Python 2 compatibility
import argparse
import glob
import os
import numpy as np
import cv2
from xml.etree import ElementTree
from dataset_interface.common import BoundingBox, NormalizedBox


ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']


class RawDescriptionAndDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Custom formatter for preserving both indentation and displaying default argument values"""
    pass


class TerminalColors(object):
    """https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def formatted_print(string, format_enum):
        print('{}{}{}'.format(format_enum, string, TerminalColors.ENDC))


def is_box_valid(x_min, y_min, x_max, y_max, img_width, img_height):
    if x_min < 0 or y_min < 0:
        return False
    if x_max > img_width or y_max > img_height:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True


def print_progress(current_index, total, prefix="", fraction=0.1):
    """
    Print progress every time the current_index match a multiple of a specified fraction of the total count
    For example, with the default fraction 0.1, the function would only print at 10%, 20%, and so on
    @return True if printing, False otherwise
    """
    increment = int(total * fraction)
    if increment == 0:
        # essentially print all indices if there area less than 10 entries in totalss
        increment = 1

    if current_index % increment == 0:
        print(prefix + "{}/{}".format(current_index, total))
        return True
    return False


def prompt_for_yes_or_no(promt_string, suffix=' [(y)es/(n)o]: ', blocking=True):
    """prompt for user [(y)es/(n)o] input, by default with append [(y)es/(n)o] to 'prompt_string'"""
    while True:
        # Note: input is
        reply = str(input(promt_string + suffix)).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
        if not blocking:
            raise ValueError('invalid user input for yes/no prompt: ' + reply)
        print("invalid input: '{}', please retry".format(reply))


def prompt_for_float(prompt_string, suffix=': ', blocking=True):
    while True:
        reply = str(input(prompt_string + suffix)).strip()
        try:
            reply_float = float(reply)
            return reply_float
        except ValueError as e:
            TerminalColors.formatted_print(e, TerminalColors.FAIL)
            if not blocking:
                raise
            continue


def case_insensitive_glob(pattern):
    """
    glob certain file types ignoring case

    :param pattern: file types (i.e. '*.jpg')
    :return: list of files matching given pattern
    """
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))


def glob_extensions_in_directory(dir_name, extensions, file_pattern='*'):
    """glob all files with specified extensions in given directory"""
    file_paths = []
    for ext in extensions:
        file_paths.extend(case_insensitive_glob(os.path.join(dir_name, file_pattern + '.' + ext)))
    return file_paths


def split_path(file_path):
    """return (directory, file_name, extension) from a string file path"""
    file_dir = os.path.dirname(file_path)
    file_basename = os.path.basename(file_path)
    file_name, extension = os.path.splitext(file_basename)
    return file_dir, file_name, extension


def display_image_and_wait(cv_image, window_name, escape_key=27):
    """Display image and wait for a specific user input key (default: 'Esc')"""
    cv2.imshow(window_name, cv_image)
    while cv2.waitKey(0) != escape_key:
        continue
    cv2.destroyWindow(window_name)


def cleanup_mask(mask, morph_kernel_size, morph_iter_num):
    """apply morphology and contour detection for mask cleanup"""
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter_num)

    # use contour detection to clean up the mask - use only the largest contour as the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    max_area_cnt = max(contours, key=cv2.contourArea)
    mask = cv2.drawContours(np.zeros(mask.shape, dtype=np.uint8), [max_area_cnt], 0, 255, cv2.FILLED)
    return mask


def draw_labeled_boxes(image, box_annotations, class_dict, font_scale=1, thickness=2, color=(255, 0, 0), copy=True):
    if copy:
        image = image.copy()

    for box_ann in box_annotations:
        if isinstance(box_ann, BoundingBox):
            cls_id = box_ann.class_id
            if cls_id is None:
                raise ValueError("'BoundingBox' instance does not have 'class_id' value specified")
            cls_name = class_dict[cls_id]
            x_min, x_max, y_min, y_max = box_ann.x_min, box_ann.x_max, box_ann.y_min, box_ann.y_max
        elif isinstance(box_ann, dict):
            cls_name = class_dict[box_ann['class_id']]
            x_min, x_max, y_min, y_max = box_ann['xmin'], box_ann['xmax'], box_ann['ymin'], box_ann['ymax']
        else:
            raise ValueError("Unexpected annotation type: {}".format(type(box_ann)))

        # Note: no checking of box dimensions at the moment, presuming they're within image boundaries
        (txt_width, txt_height), baseline = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)

        # adjust for top left corner so there's space for label
        if y_min < txt_height:
            y_min = txt_height

        # draw box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # draw label with background;
        txt_top_left = (x_min, y_min - txt_height)
        txt_bottom_right = (x_min + txt_width, y_min + baseline)
        cv2.rectangle(image, txt_top_left, txt_bottom_right, color, cv2.FILLED)
        cv2.putText(image, cls_name, (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 555), 1)

    return image


def get_bounding_boxes_from_labelimg_xml(xml_path, class_dict):
    """
    Parse bounding box and image annotations from the XML files generated using the open source annotation tool
    https://github.com/tzutalin/labelImg

    @param xml_path: path to the XML annotation file
    @class_dict: dictionary that map from class_id to class_name as often used in the rest of the package
    @return: path to the image and a list of NormalizedBox containing bounding box annotations
    """
    def get_xml_child_or_raise(xml_elem, child_name):
        xml_child = xml_elem.find(child_name)
        if xml_child is None:
            raise ValueError("element '{}' does not have child '{}'".format(xml_elem.tag, child_name))
        return xml_child

    xml_root = ElementTree.parse(xml_path).getroot()

    # get image path
    img_path = get_xml_child_or_raise(xml_root, 'path').text

    # get image dimensions
    size_elem = get_xml_child_or_raise(xml_root, 'size')
    img_width = get_xml_child_or_raise(size_elem, 'width').text
    img_height = get_xml_child_or_raise(size_elem, 'height').text
    img_width, img_height = int(img_width), int(img_height)

    # get boxes
    boxes = []
    class_dict_rev = dict((v, k) for k, v in class_dict.items())
    for obj_elem in xml_root.findall('object'):
        cls_name = get_xml_child_or_raise(obj_elem, 'name').text
        if cls_name not in class_dict_rev:
            TerminalColors.formatted_print('unrecognized class: ' + cls_name, TerminalColors.WARNING)
            continue
        cls_id = class_dict_rev[cls_name]

        box_elem = get_xml_child_or_raise(obj_elem, 'bndbox')
        x_min = int(get_xml_child_or_raise(box_elem, 'xmin').text)
        x_max = int(get_xml_child_or_raise(box_elem, 'xmax').text)
        y_min = int(get_xml_child_or_raise(box_elem, 'ymin').text)
        y_max = int(get_xml_child_or_raise(box_elem, 'ymax').text)
        boxes.append(NormalizedBox((img_height, img_width), x_min, y_min, x_max=x_max, y_max=y_max, class_id=cls_id))

    return img_path, boxes
