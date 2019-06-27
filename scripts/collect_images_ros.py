#!/usr/bin/env python
import argparse
import cv2
import os
import re
import sys
import time
import yaml
from dataset_interface.utils import TerminalColors, prompt_for_yes_or_no, prompt_for_float, \
                                    glob_extensions_in_directory, ALLOWED_IMAGE_EXTENSIONS


def find_largest_img_num(cls_image_dir, class_name):
    img_paths = glob_extensions_in_directory(cls_image_dir, ALLOWED_IMAGE_EXTENSIONS)
    pattern = os.path.join(cls_image_dir, class_name + r'_(\d+)\.[jpg|png|jpeg]')
    max_img_idx = 0
    for img_path in img_paths:
        match = re.match(pattern, img_path)
        if not match:
            continue

        # get the integer image index from the matched filename, and compare to max_img_idx
        img_idx = int(match.group(1))
        if img_idx > max_img_idx:
            max_img_idx = img_idx

    return max_img_idx


def collect_images_single_class(bridge_obj, topic_name, cls_image_dir, class_name, num_image, sleep_time, timeout=5.):
    img_idx = find_largest_img_num(cls_image_dir, class_name) + 1

    for _ in range(num_image):
        # get one image from specified and convert to OpenCV image
        try:
            img_msg = rospy.wait_for_message(topic_name, ImageMsg, timeout=timeout)
            cv_image = bridge_obj.imgmsg_to_cv2(img_msg, "bgr8")
        except rospy.ROSException:
            raise RuntimeError("failed to wait for 'sensor_msgs/Image' on topic '{}' after '{}' seconds"
                               .format(topic_name, timeout))
        except cv_bridge.CvBridgeError as e:
            raise RuntimeError("failed to convert image message to OpenCV message: " + str(e))

        # write OpenCV image to path
        img_path = os.path.join(cls_image_dir, "{}_{}.png".format(class_name, img_idx))
        print("saving image '{}'".format(img_path))
        cv2.imwrite(img_path, cv_image)
        img_idx += 1
        time.sleep(sleep_time)


def collect_images(topic_name, image_dir, class_ann_file, sleep_time):
    TerminalColors.formatted_print('Image directory: ' + image_dir, TerminalColors.OKBLUE)
    if not os.path.isdir(image_dir):
        if prompt_for_yes_or_no("'{}' does not exist, create it?".format(image_dir)):
            print('creating ' + image_dir)
            os.mkdir(image_dir)
        else:
            TerminalColors.formatted_print("Not creating '{}'. Exiting program.".format(image_dir),
                                           TerminalColors.WARNING)
            return

    TerminalColors.formatted_print('class annotation file: ' + class_ann_file, TerminalColors.OKBLUE)
    if not os.path.exists(class_ann_file):
        raise RuntimeError("Class annotation file does not exist: " + class_ann_file)
    with open(class_ann_file, 'r') as infile:
        class_dict = yaml.load(infile, Loader=yaml.FullLoader)

    bridge = cv_bridge.CvBridge()
    for cls_id, cls_name in class_dict.items():
        cls_img_dir = os.path.join(image_dir, cls_name)
        TerminalColors.formatted_print("collecting images for class '{}' in '{}'".format(cls_name, cls_img_dir),
                                       TerminalColors.BOLD)
        if not os.path.isdir(cls_img_dir):
            print("creating '{}'".format(cls_img_dir))
            os.mkdir(cls_img_dir)
        while True:
            num_image = int(prompt_for_float('please enter number of images to take'))
            if num_image < 1:
                TerminalColors.formatted_print('please input a positive integer for the number of images',
                                               TerminalColors.WARNING)
                continue
            collect_images_single_class(bridge, topic_name, cls_img_dir, cls_name, num_image, sleep_time)
            if not prompt_for_yes_or_no(
                    "do you want to take more pictures for class '{}' (e.g. different pespective)?".format(cls_name)):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="helper script to collect images from a ROS topic given a class YAML annotation")
    parser.add_argument('--topic-name', '-t', required=True,
                        help="'sensor_msgs/Image' topic to subscribe to")
    parser.add_argument('--image-dir', '-d', required=True,
                        help="Location to store the taken images")
    parser.add_argument('--class-file', '-c', required=True,
                        help="YAML file which contains the mappings from class ID to class names")
    parser.add_argument('--sleep-time', '-s', default=0.5,
                        help="Delay in seconds between taking each picture")
    args = parser.parse_args()

    try:
        import rospy
        import cv_bridge
        from sensor_msgs.msg import Image as ImageMsg
    except ImportError:
        TerminalColors.formatted_print(
            'This script is meant to work with ROS, please run it from a ROS enabled system.', TerminalColors.FAIL)
        sys.exit(1)

    rospy.init_node('image_collector', anonymous=True)
    try:
        collect_images(args.topic_name, args.image_dir, args.class_file, args.sleep_time)
        TerminalColors.formatted_print('image collection complete', TerminalColors.OKGREEN)
    except KeyboardInterrupt:
        TerminalColors.formatted_print('\nscript interrupted', TerminalColors.WARNING)
    except Exception as e:
        TerminalColors.formatted_print(e, TerminalColors.FAIL)
