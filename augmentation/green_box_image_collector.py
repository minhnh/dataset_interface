#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import signal
import cv2

def max_number_of_images_saved(dir_name, number_of_images):
    '''
    Method to verify whether or not the desired number of images has been reached

    Inputs:
        * dir_name
            -> String
            -> Name of the directory where the images are stored
        * number_of_images
            -> int
            -> Number of images desired
    Output:
        Boolean
    '''
    files = os.listdir('.')
    img_counter = 0
    for f in files:
        if '.jpg' in f.lower():
            try:
                _ = cv2.imread(f)
                img_counter += 1
            except:
                os.remove(f)
        if img_counter == number_of_images:
            return True
    return False

def rename_img_files(obj_name):
    '''
    Method to rename the images inside a directory

    Input:
        * obj_name
            -> String
            -> Name of the object
    '''
    files = os.listdir('.')
    img_counter = 0

    for f in files:
        if '.jpg' in f.lower():
            new_file_name = '{0}_{1}.jpg'.format(obj_name, img_counter)
            os.rename(f, new_file_name)
            img_counter += 1
        elif '.ini' in f.lower():
            os.remove(f)
if __name__ == '__main__':
    '''
    Arguments:
        (1) ros_topic of the camera to use to collect the images
        (2) Name of the folder to store the images
        (3) Number of images desired
    '''
    image_topic_name = sys.argv[1]
    image_dir_name = sys.argv[2]
    obj_name = sys.argv[3]
    number_of_images = int(sys.argv[4])

    if not os.path.isdir(image_dir_name):
        os.mkdir(image_dir_name)
        # os.makedirs(image_dir_name, exist_ok=True)
    os.chdir(image_dir_name)

    # Execute image_view ros node
    process = subprocess.Popen(['rosrun', 'image_view', 'image_saver', 'image:=' + image_topic_name])

    print('\033[1;35m========================================================================================\033[0;37m')
    print('\033[1;35m Waiting for {0} images on topic {1}\033[0;37m'.format(number_of_images, image_topic_name))

    # Verify the desired number of images has been reached
    while not max_number_of_images_saved(image_dir_name, number_of_images):
        time.sleep(0.1)

    print('\033[1;36m Done saving images \033[0;37m')
    print('\033[1;35m========================================================================================\033[0;37m')

    # Kill image_view ros node
    process.kill()
    subprocess.call(['pkill', 'image_saver'])

    rename_img_files(obj_name)
