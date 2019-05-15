# Image augmentation

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

The script will look for object images under `<DATA_DIRECTORY>/images`, where images for each object is stored in
a separated folder. For background subtraction, the script will look for images of the empty green box under
`<DATA_DIRECTORY>/backgrounds`. Generated masks will be stored under `<DATA_DIRECTORY>/object_masks`, where masks
for each object will be stored in a separated folder. The directory may then look like this:

```sh
<DATA_DIRECTORY>
├── backgrounds
├── images
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

## Generating bounding box annotations from segmented masks

TODO(minhnh)
