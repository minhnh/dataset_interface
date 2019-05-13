"""
Module containing different approaches for segmenting objects from green background, i.e. the green box
"""
import numpy as np
import cv2


def background_subtraction(image, background, background_threshold):
    img_diff = np.clip(np.abs(image - background), 0, 255)
    img_diff = np.array(img_diff, dtype=np.uint8)

    small_brightness_pixels = np.where(img_diff < background_threshold)
    img_diff[small_brightness_pixels] = 0
    img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
    return img_diff_gray
