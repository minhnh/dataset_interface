# `dataset_interface`

A common interface for interacting with dataset, in context of the
[b-it-bots teams](https://github.com/b-it-bots/).

Note: tested only on Python 3.7.

## Installation
```
python setup.py install --user
```

## Common interfaces

### [`image_data_api.py`](./common/image_data_api.py)

#### `ImageInfo`
Meant to hold image metadata.

#### `Category`
Meant to handle hierarchies of images categories.

#### `ImageDetectionDataAPI`
Meant to handle common queries to dataset.

#### Sample usage
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

## Handled datasets

* [COCO](http://cocodataset.org/): see specific [README](./coco/README.md) for more info
