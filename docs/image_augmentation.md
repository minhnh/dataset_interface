# Image augmentation

TODO(minhnh) add and link to Jupyter notebook example

* [Background segmentation](#background-segmentation)
* [Generating images and bounding box annotations from segmented masks](#generating-images-and-bounding-box-annotations-from-segmented-masks)

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

Using the object masks and green box images created above, synthetic images can be created by projecting the segmented
pixels onto new background images, while corresponding bounding box annotations can be calculated.

The [`generate_detection_data.py`](../scripts/generate_detection_data.py) script handles user interactions for
generating images and bounding box annotations from the segmented object masks. Usage:

```sh
$ scripts/generate_detection_data.py -h
usage: generate_detection_data.py [-h] --data-directory DATA_DIRECTORY
                                  --class-annotations CLASS_ANNOTATIONS
                                  [--output-dir OUTPUT_DIR]
                                  [--output-annotation-dir OUTPUT_ANNOTATION_DIR]

Script to generate training images and annotations for bounding box based
object detection. This is done by randomly projecting segemented object pixels
onto backgrounds, then calculating the corresponding bounding boxes.

optional arguments:
  -h, --help            show this help message and exit
  --data-directory DATA_DIRECTORY, -d DATA_DIRECTORY
                        directory where the script will look for images,
                        backgrounds and saved object masks (default: None)
  --class-annotations CLASS_ANNOTATIONS, -c CLASS_ANNOTATIONS
                        file containing mappings from class ID to class name
                        (default: None)
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        (optional) directory to store generated images
                        (default: None)
  --output-annotation-dir OUTPUT_ANNOTATION_DIR, -a OUTPUT_ANNOTATION_DIR
                        (optional) directory to store the generated YAML
                        annotations (default: None)
```

The script will look for the objects' masks and green box images in the directory as specified in the
[background segmentation section](#background-segmentation). The script will prompt for the data split name
(e.g. `go_2019_train`). This will be used as the annotation file name and prefix for the image names.
If `<OUTPUT_DIR>` is not specified, the images will be created under `<DATA_DIRECTORY>/synthetic_images/<split_name>`.
If `<OUTPUT_ANNOTATION_DIR>` is not specified, the annotations will be appended to
`<DATA_DIRECTORY>/annotations/<split_name>.yml`. A sample session may look like the following:

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
