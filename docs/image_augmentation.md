# Image augmentation

- [Image augmentation](#image-augmentation)
  - [Background segmentation](#background-segmentation)
  - [Generating images and bounding box annotations from segmented masks](#generating-images-and-bounding-box-annotations-from-segmented-masks)
  - [Collect images from a ROS topic](#collect-images-from-a-ros-topic)

## Background segmentation

Currently we use the Gaussian Mixture-based Background/Foreground Segmentation implementation in OpenCV
([documentation](https://docs.opencv.org/master/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)) for
background subtraction.

The script [`segment_background.py`](../scripts/segment_background.py) handles the necessary operations and
user interactions for generating masks for object images taken using the green box. Usage:

```sh
$ scripts/segment_background.py -h
usage: segment_background.py [-h] [--method {bg_sub}] --data-directory
                             DATA_DIRECTORY --class-annotations
                             CLASS_ANNOTATIONS

Script to segment objects from background for data augmentation using the green box in b-it-bots@Home.

Available methods:
1. bg_sub(default): simple background subtraction

optional arguments:
  -h, --help            show this help message and exit
  --method {bg_sub}, -m {bg_sub}
                        background segmentation method (default: bg_sub)
  --data-directory DATA_DIRECTORY, -d DATA_DIRECTORY
                        directory where the script will look for images,
                        backgrounds and save object masks (default: None)
  --class-annotations CLASS_ANNOTATIONS, -c CLASS_ANNOTATIONS
                        file containing mapping from class ID to class name
                        (default: None)
```

The script will look for object images under `<DATA_DIRECTORY>/green_box_images`, where images for each object
is stored in a separated folder. For background subtraction, the script will look for images of the empty green box
under `<DATA_DIRECTORY>/green_box_backgrounds`. Generated masks will be stored under `<DATA_DIRECTORY>/object_masks`,
where masks for each object will be stored in a separated folder. The directory may then look like this:

```sh
<DATA_DIRECTORY>
├── green_box_backgrounds
├── green_box_images
│   ├── <class_name_1>
│   ├── <class_name_2>
│   └── ...
└── object_masks
    ├── <class_name_1>
    ├── <class_name_2>
    └── ...
```

`<class_name_1>`, `<class_name_2>`,... will be loaded from `<CLASS_ANNOTATIONS>`, which is the YAML file containing
mapping from class ID to class names described in [`tensorflow_models.md`](tensorflow_models.md).

## Generating images and bounding box annotations from segmented masks

Using object images and masks, synthetic images can be created by projecting the segmented pixels
onto background images, computing bounding box annotations and masks of the syntetic images.

The [`generate_detection_data.py`](../scripts/generate_detection_data.py) script handles user interactions for
generating images, masks, and bounding box annotations from the segmented object masks.  This is done by randomly
projecting segemented object pixels onto backgrounds, then calculating the corresponding masks and bounding boxes.

Usage:

```sh
$ scripts/generate_detection_data.py -h
usage: generate_detection_data.py [-h] --data-directory DATA_DIRECTORY
                                  --background-directory BACKGROUND_DIRECTORY
                                  --class-annotations CLASS_ANNOTATIONS
                                  --num-objects-per-class NUM_OBJECTS_PER_CLASS
                                  --prob_rand_trans PROB_RAND_TRANS
                                  [--output-dir OUTPUT_DIR]
                                  [--output-annotation-dir OUTPUT_ANNOTATION_DIR]
                                  [--annotation-format ANNOTATION_FORMAT]
                                  [--invert_mask]

optional arguments:
  -h, --help            show this help message and exit
  --data-directory, -d
                        directory where the script will look for images and
                        object masks (default: None)
  --background-directory, -bg
                        directory where the script will look for background
                        images (default: None)
  --class-annotations, -c
                        file containing mappings from class ID to class name
                        (default: None)
  --num-objects-per-class -n
                        maximum number of images per class that are used for augmentation
                        (this means that if there are more available images from the class,
                        not all of them will be used during augmentation)
  --prob_rand_trans -pt
                        probability of applying a random rotation to each object
                        during augmentation (probability = 0 means that no rotation
                        will be applied)
  --output-dir, -o
                        (optional) directory to store generated images
                        (default: None)
  --output-annotation-dir, -a
                        (optional) directory to store the generated
                        annotations (default: None)
  --invert_mask - im
                        whether to invert the colours of the object segmentation mask
                        (necessary for black and white masks in case the object is black
                        and the background is white)
  --annotation-format -af
                        format in which the augmented images should be annotated
                        (allowed values 'custom' and 'voc'); in case of custom
                        annotations, only a YAML annotation file as described below
                        is created, while in case of VOC annotations, a VOC annotation
                        file for each image is created as well
```



The script will look for the objects' masks and images in the `<DATA_DIRECTORY>` and for background images in the
`<BACKGROUND_DIRECTORY>`. It will prompt for the data split name (e.g. `go_2019_train`). This will be used as the
annotation file name and the prefix for the image names. If `<OUTPUT_DIR>` is not specified, the images will be
created under `<DATA_DIRECTORY>/synthetic_images/<split_name>`. If `<OUTPUT_ANNOTATION_DIR>` is not specified,
the annotations will be appended to `<DATA_DIRECTORY>/annotations/<split_name>.yml`.

The `<DATA_DIRECTORY>` is expected to be organized in the following manner:

```sh
<DATA_DIRECTORY>
│  ├── <class_name_1>
│    ├── images
│    ├── masks
│  ├── <class_name_2>
│    ├── images
│    ├── masks
│  └── ...
```
A sample session may look like the following:

```sh
$ scripts/generate_detection_data.py -d ${DATA_DIRECTORY} -c ${CLASS_ANNOTATIONS}
Data directory: ${DATA_DIRECTORY}
will look for object green box images in: ${DATA_DIRECTORY}/green_box_images
will look for object masks in: ${DATA_DIRECTORY}/object_masks
will look for backgrounds for image augmentation in: ${DATA_DIRECTORY}/augmentation_backgrounds
found '5' background images
Class annotation file: ${CLASS_ANNOTATIONS}
Loading object masks for '12' classes
loading images and masks for class 'class_name_1'
found '20' object images in '${DATA_DIRECTORY}/class_name_1'
loading images and masks for class 'class_name_2'
found '20' object images in '${DATA_DIRECTORY}/class_name_2'
...
begin generating images under '${DATA_DIRECTORY}/synthetic_images' and annotation files under '${DATA_DIRECTORY}/annotations'
please enter split name (e.g. 'go2019_train'): ${SPLIT_NAME}
please enter the number of images to be generated for each background: 200
please enter the maximum number of objects to be projected onto each background: 7
generating images for split 'go_train' under '${DATA_DIRECTORY}/synthetic_images/${SPLIT_NAME}'
generating annotations for split 'go_train' in '${DATA_DIRECTORY}/annotations/${SPLIT_NAME}.yml'
creating image 50/1000
creating image 100/1000
...
creating image 1000/1000
do you want to generate images for another dataset split? [(y)es/(n)o]: n
image and annotation generation complete
```

## Collect images from a ROS topic

To directly collect images from the robot, a simple script that read `sensor_msgs/Image` messages from a
[ROS](http://www.ros.org/) topic is also included. Usage:

```sh
$ scripts/collect_images_ros.py -h
usage: collect_images_ros.py [-h] --topic-name TOPIC_NAME --image-dir
                             IMAGE_DIR --class-file CLASS_FILE
                             [--sleep-time SLEEP_TIME]

helper script to collect images from a ROS topic given a class YAML annotation

optional arguments:
  -h, --help            show this help message and exit
  --topic-name TOPIC_NAME, -t TOPIC_NAME
                        'sensor_msgs/Image' topic to subscribe to (default:
                        None)
  --image-dir IMAGE_DIR, -d IMAGE_DIR
                        Location to store the taken images (default: None)
  --class-file CLASS_FILE, -c CLASS_FILE
                        YAML file which contains the mappings from class ID to
                        class names (default: None)
  --sleep-time SLEEP_TIME, -s SLEEP_TIME
                        Delay in seconds between taking each picture (default:
                        0.5)
```

For example, for collecting green box images for objects, you may execute:

```sh
scripts/collect_image_ros.py -t ${IMAGE_TOPIC} -d ${DATA_DIRECTORY}/green_box_images -c ${CLASS_ANNOTATIONS}
```

This will save images for each class under `${DATA_DIRECTORY}/green_box_images/<class_name>`. For each class, the user
will be prompted to input the number of images to take until done, allowing changing the object orientation and
camera perspective in between.
