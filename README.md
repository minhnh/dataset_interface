# `dataset_interface`

This repository includes:

* A common interface for interacting with different datasets, in context of the
  [b-it-bots RoboCup teams](https://github.com/b-it-bots/).
* Tools/pipelines for automatically segmenting object masks using the green box (picture below), using these masks
  to synthesize data for training object detection models, and training on the synthesized data

Note: tested only with Python 3.

* [Installation](#installation)
* [Common interface for datasets](#common-interface-for-datasets)
* [Image augmentation](#image-augmentation)
* [Training](#training)

## Installation

Since the data API from `pycocotools` requires building Cython modules, `pip install -e .` and
`python setup.py develop` does not seem to work. Arch users may have to install the
[`tk` windowing toolkit](https://www.archlinux.org/packages/extra/x86_64/tk/) manually as a system dependency.

```sh
python setup.py install --user
```

## Common interface for datasets

A more detailed description of how this interface works can be found in the
[dataset interface documentation](docs/dataset_interface.md)

A sample config file for the COCO dataset: [`sample_coco_configs.yml`](./config/sample_coco_configs.yml)

```python
from dataset_interface.coco import COCODataAPI

data_dir = 'data/dir'
config_file = 'config/file.yml'
coco_api = COCODataAPI(data_dir, config_file)

# return a dictionary mapping image ID to ImageInfo objects
images = coco_api.get_images_in_category('indoor')
first_image = next(iter(images.values()))
print(first_image.image_path)   # image file location
print(first_image.url)          # image URL for download
print(first_image.id)           # image unique ID in the dataset

# return a dictionary containing all categories under the 'indoor' super category,
# mapping category ID's to Category objects
indoor_cats = coco_api.get_sub_categories('indoor')
# mapping category ID's to category names
indoor_cat_names = coco_api.get_sub_category_names('indoor')

```

## Object Detection and Classification Model Training

We primarily train detection models using the tools and model definitions from `torchvision` as described in the [Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Guidelines for training and using models can be found in [`docs/object_detection.md`](docs/object_detection.md)

## Image augmentation

We aim to ease the process of generating data for object detection. Using the green box in the picture below,
it's possible to automatically segment objects and transform them onto new backgrounds to create new training
data for an object detection model. A more detailed documentation of how we solve this problem can be found
in [`docs/image_augmentation.md`](docs/image_augmentation.md).

![Green Box](docs/green_box.png)
