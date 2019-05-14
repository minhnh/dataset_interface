"""
Module containing different approaches for segmenting objects from green background, i.e. the green box
"""
from enum import Enum
from abc import ABC, abstractmethod
import os
import yaml
import numpy as np
import cv2
from dataset_interface.utils import TerminalColors, prompt_for_yes_or_no, glob_extensions_in_directory


ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']


class SegmentationMethods(Enum):
    BACKGROUND_SUBTRACTION = 'bg_sub'

    @staticmethod
    def values():
        return list(map(lambda e: e.value, SegmentationMethods))


class BackgroundSegmentation(ABC):
    """
    segment objects from backgrounds in object images and save the generated masks
    - will look for object images in <_data_dir>/images/<object_name>
    - <object_name> comes from <class_annotation_file>
    - will write object masks to <_data_dir>/object_masks/<object_name>
    """
    _data_dir = None
    _image_dir = None
    _mask_dir = None
    _class_dict = None
    _confirmation = None

    def __init__(self, data_dir, class_annotation_file, confirmation=True):
        self._confirmation = confirmation

        # check data directory
        self._data_dir = data_dir
        TerminalColors.formatted_print("Data directory: " + data_dir, TerminalColors.OKBLUE)
        if not os.path.exists(self._data_dir):
            raise ValueError("data directory does not exist: " + self._data_dir)

        # check image directory
        self._image_dir = os.path.join(data_dir, 'images')
        if not os.path.exists(self._image_dir):
            raise ValueError("image directory does not exist: " + self._image_dir)

        # check directory to store object masks
        self._mask_dir = os.path.join(data_dir, 'object_masks')
        TerminalColors.formatted_print("Will generate object masks to: " + self._mask_dir, TerminalColors.OKBLUE)
        if not os.path.exists(self._mask_dir):
            print("creating directory '{}'".format(self._mask_dir))
            os.mkdir(self._mask_dir)
        elif os.listdir(self._mask_dir) and self._confirmation:
            if not prompt_for_yes_or_no("Mask directory '{}' not empty, overwrite?".format(self._mask_dir)):
                raise ValueError("directory '{}' not empty, not overwriting".format(self._mask_dir))

        # load class annotation file
        if not os.path.exists(class_annotation_file):
            raise ValueError("class annotation YAML does not exist: " + class_annotation_file)
        with open(class_annotation_file, 'r') as infile:
            self._class_dict = yaml.load(infile, Loader=yaml.FullLoader)
        TerminalColors.formatted_print("Found '{}' classes in annotation file '{}'"
                                       .format(len(self._class_dict), class_annotation_file), TerminalColors.OKBLUE)

    def segment(self):
        for class_name in self._class_dict.values():
            # check directory containing object images
            class_img_dir = os.path.join(self._image_dir, class_name)
            if not os.path.exists(class_img_dir):
                TerminalColors.formatted_print("skipping class '{}': directory '{}' does not exist"
                                               .format(class_name, class_img_dir), TerminalColors.WARNING)
                continue

            # glob all supported image files & warn if no image found
            image_paths = glob_extensions_in_directory(class_img_dir, ALLOWED_IMAGE_EXTENSIONS)
            if not image_paths:
                TerminalColors.formatted_print(
                    "skipping class '{}': directory '{}' does not contain any supported images"
                    .format(class_name, class_img_dir), TerminalColors.WARNING)
                continue

            TerminalColors.formatted_print("Generating masks for class '{}' from '{}' images"
                                           .format(class_name, len(image_paths)), TerminalColors.BOLD)
            self._segment_class(class_name, image_paths)

    @abstractmethod
    def _segment_class(self, class_name, image_paths):
        raise NotImplementedError("abstract method 'segment' must be implemented")


class BackgroundSubtraction(BackgroundSegmentation):
    _background_dir = None

    def __init__(self, data_dir, class_annotation_file, confirmation=True):
        super(BackgroundSubtraction, self).__init__(data_dir, class_annotation_file, confirmation=confirmation)

        # check background directory for BACKGROUND_SUBTRACTION
        self._background_dir = os.path.join(data_dir, 'backgrounds')
        if not os.path.exists(self._background_dir):
            raise ValueError("background directory does not exist: " + self._background_dir)

    def _segment_class(self, class_name, image_paths):
        # find background image for object
        obj_bg_paths = glob_extensions_in_directory(self._background_dir, ALLOWED_IMAGE_EXTENSIONS,
                                                    file_pattern=class_name)
        if not obj_bg_paths:
            TerminalColors.formatted_print("skipping class '{}': directory '{}' contain no background for this class"
                                           .format(class_name, self._background_dir), TerminalColors.WARNING)
            return


def background_subtraction(image, background, background_threshold):
    img_diff = np.clip(np.abs(image - background), 0, 255)
    img_diff = np.array(img_diff, dtype=np.uint8)

    small_brightness_pixels = np.where(img_diff < background_threshold)
    img_diff[small_brightness_pixels] = 0
    img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
    return img_diff_gray
