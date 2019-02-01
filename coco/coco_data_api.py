import os
from ..common.image_data_api import ImageDetectionDataAPI, Category

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
