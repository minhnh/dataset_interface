#!/usr/bin/env python3
import sys
import os

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import numpy as np
from imageio import imread, imwrite
import cv2


def generate_object_masks(image_dir_name, background_image_name, background_threshold):
    '''
    Method to generate a mask of an object by performing background substraction

    Inputs:
        * image_dir_name
            -> String
            -> Name of the directory where the images are stored
        * background_image_name
            -> String
            -> Name of the background image
        * background_threshold
            -> Int
            -> Threshold to identify the pixels with small brigthness
    '''
    background = np.array(imread(background_image_name), dtype=int)

    os.chdir(image_dir_name)
    files = os.listdir('.')
    if not os.path.isdir('object_masks'):
        os.mkdir('object_masks')
    for f in files:
        if not os.path.isfile(f):
            continue
        try:
            img = np.array(imread(f), dtype=int)
        except:
            print('Corrupted file ', f)
            continue
        img_diff = np.clip(np.abs(img - background), 0, 255)
        img_diff = np.array(img_diff, dtype=np.uint8)

        small_brightness_pixels = np.where(img_diff < background_threshold)
        img_diff[small_brightness_pixels] = 0
        img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)

        img_name, img_extension = f.split('.')
        mask_file_path = os.path.join('object_masks', img_name + '_mask.' + img_extension)
        imwrite(mask_file_path, img_diff_gray)

if __name__ == '__main__':
    '''
    Arguments:
        (1) Name of the folder to store the images
        (2) Name of the background image (including path)
        (3) Threshold to regulate the background substraction
    '''

    image_dir_name = sys.argv[1]
    background_image_name = sys.argv[2]
    background_threshold = int(sys.argv[3])

    print('\033[1;35m========================================================================================\033[0;37m')
    print('\033[1;35m Generating object masks...\033[0;37m')

    generate_object_masks(image_dir_name, background_image_name, background_threshold)

    print('\033[1;36m Object masks successfully generated\033[0;37m')
    print('\033[1;35m========================================================================================\033[0;37m')
