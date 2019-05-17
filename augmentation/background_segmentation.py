"""
Module containing different approaches for segmenting objects from green background, i.e. the green box
"""
from enum import Enum
from abc import ABC, abstractmethod
import os
import yaml
import numpy as np
import cv2
from dataset_interface.utils import TerminalColors, prompt_for_yes_or_no, prompt_for_float, \
                                    glob_extensions_in_directory, split_path, ALLOWED_IMAGE_EXTENSIONS, \
                                    display_image_and_wait, cleanup_mask


def get_image_mask_path(image_path, mask_dir):
    """enforces name convention for generated object masks"""
    _, filename, extension = split_path(image_path)
    return os.path.join(mask_dir, filename + '_mask' + extension)


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
        if not os.path.isdir(self._data_dir):
            raise RuntimeError("'{}' is not an existing directory".format(self._data_dir))

        # check directory with green box images
        self._image_dir = os.path.join(data_dir, 'green_box_images')
        if not os.path.isdir(self._image_dir):
            raise RuntimeError("'{}' is not an existing directory".format(self._image_dir))

        # check directory to store object masks
        self._mask_dir = os.path.join(data_dir, 'object_masks')
        TerminalColors.formatted_print("Will generate object masks to: " + self._mask_dir, TerminalColors.OKBLUE)
        if not os.path.exists(self._mask_dir):
            print("creating directory '{}'".format(self._mask_dir))
            os.mkdir(self._mask_dir)
        elif os.listdir(self._mask_dir) and self._confirmation:
            if not prompt_for_yes_or_no("Mask directory '{}' not empty, overwrite?".format(self._mask_dir)):
                raise RuntimeError("directory '{}' not empty, not overwriting".format(self._mask_dir))

        # load class annotation file
        if not os.path.exists(class_annotation_file):
            raise RuntimeError("class annotation YAML does not exist: " + class_annotation_file)
        with open(class_annotation_file, 'r') as infile:
            self._class_dict = yaml.load(infile, Loader=yaml.FullLoader)
        TerminalColors.formatted_print("Found '{}' classes in annotation file '{}'"
                                       .format(len(self._class_dict), class_annotation_file), TerminalColors.OKBLUE)

    def segment(self):
        for class_name in self._class_dict.values():
            # check directory containing object images
            class_img_dir = os.path.join(self._image_dir, class_name)
            if not os.path.isdir(class_img_dir):
                TerminalColors.formatted_print("skipping class '{}': '{}' is not an existing directory"
                                               .format(class_name, class_img_dir), TerminalColors.WARNING)
                continue

            # glob all supported image files & warn if no image found
            image_paths = glob_extensions_in_directory(class_img_dir, ALLOWED_IMAGE_EXTENSIONS)
            if not image_paths:
                TerminalColors.formatted_print(
                    "skipping class '{}': directory '{}' does not contain any supported images"
                    .format(class_name, class_img_dir), TerminalColors.WARNING)
                continue

            # Handle algorithm-specific segmentation
            TerminalColors.formatted_print("Generating masks for class '{}' from '{}' images"
                                           .format(class_name, len(image_paths)), TerminalColors.BOLD)
            self._segment_class(class_name, image_paths)

    @abstractmethod
    def _segment_class(self, class_name, image_paths):
        raise NotImplementedError("abstract method 'segment' must be implemented")


class BackgroundSubtraction(BackgroundSegmentation):
    """
    Extension which use background subtraction for segmentation.
    Will look for empty backgrounds in <data_dir>/green_box_backgrounds
    """
    DEFAULT_VAR_THRESHOLD = 300
    _backgrounds = None

    def __init__(self, data_dir, class_annotation_file, confirmation=True):
        super(BackgroundSubtraction, self).__init__(data_dir, class_annotation_file, confirmation=confirmation)

        # check background directory for BACKGROUND_SUBTRACTION
        background_dir = os.path.join(data_dir, 'green_box_backgrounds')
        if not os.path.isdir(background_dir):
            raise RuntimeError("'{}' is not an existing directory".format(background_dir))

        # load all backgrounds
        bg_paths = glob_extensions_in_directory(background_dir, ALLOWED_IMAGE_EXTENSIONS)
        self._backgrounds = [cv2.imread(bg) for bg in bg_paths]
        if not self._backgrounds:
            raise RuntimeError("found no background in directory: " + background_dir)
        print("found '{}' background images in '{}'".format(len(self._backgrounds), background_dir))

    def _segment_class(self, class_name, image_paths):
        # fine-tune threshold for background subtraction
        bg_threshold = self.find_bg_threshold(image_paths[0])
        print('using background threshold: {}'.format(bg_threshold))

        # create mask directory for the current object if doesn't exist
        obj_mask_dir = os.path.join(self._mask_dir, class_name)
        if not os.path.isdir(obj_mask_dir):
            print("creating directory '{}'".format(obj_mask_dir))
            os.mkdir(obj_mask_dir)

        # generate and save masks
        for img_path in image_paths:
            img = cv2.imread(img_path)
            mask = self.subtract_background(img, bg_threshold)
            mask_path = get_image_mask_path(img_path, obj_mask_dir)
            print("saving mask to '{}'".format(mask_path))
            cv2.imwrite(mask_path, mask)

    def find_bg_threshold(self, test_image_path):
        """handle user interaction for fine-tuning the ver_threshold parameter for subtraction operation"""
        if not self._confirmation:
            return BackgroundSubtraction.DEFAULT_VAR_THRESHOLD

        bg_threshold = BackgroundSubtraction.DEFAULT_VAR_THRESHOLD
        while True:
            print("current threshold: {}; press 'Esc' to exit image window.".format(bg_threshold))
            # calculate mask
            test_image = cv2.imread(test_image_path)
            mask = self.subtract_background(test_image, bg_threshold)

            # display mask and image for fine-tuning threshold
            mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            image_vstack = np.vstack((test_image, mask_3_channel))
            display_image_and_wait(image_vstack, 'vstack mask and object image')

            # ask if new threshold is needed
            if not prompt_for_yes_or_no("Is a new background threshold needed?"):
                break
            bg_threshold = prompt_for_float('please input a numeric new background threshold')

        return bg_threshold

    def subtract_background(self, image, var_threshold, morph_iter_num=10, morph_kernel_size=3):
        """
        use MOG2 background substration algorithm to generate object mask, then perform morphological transformations
        to clean up the masks. Relevant documentation:
        - https://docs.opencv.org/master/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
        - https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html

        @param var_threshold: from OpenCV documentation, is the threshold on the squared Mahalanobis distance
                              between the pixel and the model to decide whether a pixel is well described by
                              the background model
        @param image: image to perform background subtraction on
        @param morph_iter_num: number of iterations to apply morphology operations. We use MORPH_CLOSE, which means
                               a dilation followed by an erosion
        @param morph_kernel_size: size of the kernel for the morphology operation
        @return: segmented mask for the object
        """
        # perform background subtraction
        subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=var_threshold, detectShadows=False)
        for bg in self._backgrounds:
            subtractor.apply(bg)
        mask = subtractor.apply(image)

        return cleanup_mask(mask, morph_kernel_size, morph_iter_num)
