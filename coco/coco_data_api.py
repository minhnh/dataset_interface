import os
from dataset_interface.common import ImageDetectionDataAPI, Category, ImageInfo

try:
    from pycocotools.coco import COCO
except ImportError:
    print('pycocotools not installed or installed incorrectly')
    raise


class COCODataAPI(ImageDetectionDataAPI):
    # data split name, i.e. 'val2017'
    _split_name = None              # type:str
    # COCO API object
    _coco = None                    # type: COCO

    def __init__(self, data_dir, config_file_path):
        super(COCODataAPI, self).__init__(data_dir, config_file_path)

    def _initialize(self):
        self._split_name = self._configurations.get('split_name', None)
        if not self._split_name:
            raise ValueError('required configuration "split_name" is not specified')

        annotation_file = os.path.join(self._data_dir, 'annotations', 'instances_{}.json'.format(self._split_name))
        self._coco = COCO(annotation_file)

        self._image_dir = os.path.join(self._data_dir, 'images', self._split_name)

    def _parse_categories(self):
        categories = self._coco.loadCats(self._coco.getCatIds())
        for category in categories:
            super_category = category['supercategory']
            if super_category not in self._category_hierarchy:
                # COCO has no ID for super categories, so use name as ID's
                self._categories[super_category] = Category(super_category, super_category)
                self._category_hierarchy[super_category] = self._categories[super_category]

            if category['id'] in self._categories:
                # avoiding redundant categories
                continue

            category_obj = Category(category['id'], category['name'])
            self._categories[category_obj.category_id] = category_obj
            self._category_hierarchy[super_category].add_sub_category(category_obj)

    def get_images_in_category(self, category_id):
        images = {}
        category_ids = None
        if category_id in self.category_hierarchy:
            sub_categories = self.get_sub_categories(category_id)
            category_ids = list(sub_categories.keys())
        elif category_id in self._categories:
            category_ids = [self._categories[category_id].category_id]
        else:
            print("category with ID '{}' not found".format(category_id))
            return images

        for category in category_ids:
            coco_images = self._coco.loadImgs(self._coco.catToImgs[category])
            for coco_image in coco_images:
                image_id = coco_image['id']
                if image_id in images:
                    continue

                if image_id in self._image_info_collection:
                    images[image_id] = self._image_info_collection[image_id]
                    continue

                image_info = ImageInfo(image_id, self._image_dir, coco_image['file_name'], coco_image['coco_url'])
                self._image_info_collection[image_id] = image_info
                images[image_id] = image_info

        return images
