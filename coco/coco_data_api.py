import os
from ..common.image_data_api import ImageDetectionDataAPI, ObjectClass

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

    def _parse_classes(self):
        categories = self._coco.loadCats(self._coco.getCatIds())
        for category in categories:
            super_category = category['supercategory']
            if super_category not in self._class_hierarchy:
                # COCO has no ID for super categories
                self._class_hierarchy[super_category] = ObjectClass(super_category, super_category)
            self._class_hierarchy[super_category].add_subclass(ObjectClass(category['id'], category['name']))
