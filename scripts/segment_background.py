#!/usr/bin/env python3
"""
Script to segment objects from background for data augmentation using the green box in b-it-bots@Home.

Available methods:
1. bg_sub(default): simple background subtraction
"""
import argparse
from dataset_interface.augmentation.background_segmentation import SegmentationMethods, BackgroundSubtraction
from dataset_interface.utils import RawDescriptionAndDefaultsFormatter, TerminalColors


def segment_background(arguments):
    # parse segmentation method
    try:
        seg_method_enum = SegmentationMethods(arguments.method)
        TerminalColors.formatted_print("Segmentation method: " + seg_method_enum.name,
                                       TerminalColors.OKBLUE)
    except ValueError:
        raise ValueError("invalid segmentation method: " + arguments.method)

    # create segmentation object and perform segmentation
    if seg_method_enum == SegmentationMethods.BACKGROUND_SUBTRACTION:
        bg_segmentor = BackgroundSubtraction(arguments.data_directory, arguments.class_annotations)
    else:
        raise ValueError('unsupported segmentation method: ' + seg_method_enum.name)
    bg_segmentor.segment()


if __name__ == '__main__':
    # Note: using builtin var '__doc__' as script description
    parser = argparse.ArgumentParser(formatter_class=RawDescriptionAndDefaultsFormatter, description=__doc__)

    parser.add_argument('--method', '-m', choices=SegmentationMethods.values(),
                        default=SegmentationMethods.BACKGROUND_SUBTRACTION.value,
                        help='background segmentation method')
    parser.add_argument('--data-directory', '-d', required=True,
                        help='directory where the script will look for images, backgrounds and save object masks')
    parser.add_argument('--class-annotations', '-c', required=True,
                        help='file containing mapping from class ID to class name')

    try:
        segment_background(parser.parse_args())
        TerminalColors.formatted_print('segmentation complete', TerminalColors.OKGREEN)
    except KeyboardInterrupt:
        TerminalColors.formatted_print('\nscript interrupted', TerminalColors.WARNING)
    except Exception as e:
        TerminalColors.formatted_print(e, TerminalColors.FAIL)
