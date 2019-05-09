# Green box scripts

This repository contains a set of scripts used to collect images, generate masks of the images, and generate artificial images using any of the cameras in the *HSR robot*

**NOTE:**

* The scripts are themselves robot independent, but the instructions here given are specific for HSR Toyota robot.
* It is preferred to use any of the stereo cameras rather than the RGB-D because the stereo cameras are HD.

## Content

* `green_box_image_collector.py`
* `green_box_mask_generator.py`
* `artificial_images_generator.py`
* `create_robocup_tf_record.py`
* `protobuf_generator.py`
* `rename_images.py`
* `green_box_object_mask_creator.ipynb`

## Requirements

If collecting images remotely:
    - Connect to the network **b-it-bots@home**
    - Set up external ros master by executing
        `export ROS_MASTER_URI=http://hsrb.local:11311`

## Setup

TODO: Include images of the setup

### Camera options

TODO: Include information about the cameras in the robot

### Lighting conditions

TODO: Include images about the lighting configurations

## Instructions

### Positioning the camera

1. Open in a web browser the address:
    `http://192.168.50.201/admin/`
2. Access *Servo States* page
3. Introduce a value inside the *command* cell (or dragging the *command bar*) for the joint required to move in order to position the desired camera in the right position relative to the base of the box

### Collecting images

#### Background image

* Command

    `python3 green_box_image_collector.py image_topic_name image_dir_name image_class number_of_images`

    Example

    `python3 green_box_image_collector.py /hsrb/head_l_stereo_camera/image_raw background_1_light background 1`

#### Object images

* Command

    `python3 green_box_image_collector.py image_topic_name image_dir_name image_class number_of_images`

    Example

    `python3 green_box_image_collector.py /hsrb/head_l_stereo_camera/image_raw robocup_images/objects_top_down/cereal/crunchy_flakes crunchy_flakes 20`

Notes:
    - image_topic_name: rostopic of the desired camera
    - image_dir_name: directory where the image will be stored
    - image_class: name that will be assigned to the image
    - number_of_images: number of images desired
    *Background image is used to generate the segmentation mask. It is required to collect a background image often due to changes in ambient lighting conditions*

### Generating object masks

* Command

    `python3 green_box_mask_generator.py image_dir_name background_image_name background_threshold`

    Example

    `python3 green_box_mask_generator.py sponge_02_04_19_front background_02_04_19/background_02_04_19_0.jpg 30`

Notes:
    - image_dir_name: path to directory where object images are located
    - background_image_name: path of the background image
    - background_threshold: Threshold used to regulate the generation of the mask

### Generating artificial images

#### Training images

* Command
    `python3 artificial_images_generator.py img_dir_name background_img_dir images_per_background annotations_file output_dir`

    Example

    `python3 artificial_images_generator.py objects_perspectives augmentation_backgrounds 5000 train_annotations.yml robocup_images`

#### Validation images

* Command
    `python3 artificial_images_generator.py img_dir_name background_img_dir images_per_background annotations_file output_dir`

    Example

    `python3 artificial_images_generator.py objects_perspectives augmentation_backgrounds 1000 val_annotations.yml robocup_images`

Notes:
    - img_dir_name: path to directory where object images are located
    - background_img_dir: path to directory where augmentation backgrounds are located
    - images_per_background: desired number of images per background
    - annotations_file: YAML file where the labels are going to be generated
    - output_dir: path to the directory where the generated images are going to be stored

## Maintainer

    * Erick Kramer
    * Email: erickkramer@gmail.com
