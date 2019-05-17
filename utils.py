from builtins import input      # for Python 2 compatibility
import argparse
import glob
import os
import numpy as np
import cv2


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
    """
    increment = int(total * fraction)
    if increment == 0:
        # essentially print all indices if there area less than 10 entries in totalss
        increment = 1

    if current_index % increment == 0:
        print(prefix + "{}/{}".format(current_index, total))


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
    return (file_dir, *os.path.splitext(file_basename))


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
