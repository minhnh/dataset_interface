import os
from dataset_interface.common import ImageDetectionDataAPI, Category, ImageInfo, BoundingBox
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
                self._category_names[super_category] = super_category
                self._categories[super_category] = Category(super_category, super_category)
                self._category_hierarchy[super_category] = self._categories[super_category]

            if category['id'] in self._categories:
                # avoiding redundant categories
                continue

            category_obj = Category(category['id'], category['name'], parent_category=super_category)
            self._category_names[category_obj.name] = category_obj.category_id
            self._categories[category_obj.category_id] = category_obj
            self._category_hierarchy[super_category].add_sub_category(category_obj)

    def _parse_image_info(self):
        for image_id, image_dict in self._coco.imgs.items():
            self._image_info_dict[image_id] = ImageInfo(image_id, self._image_dir, image_dict['file_name'],
                                                        image_dict['coco_url'])

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
            for image_id in self._coco.catToImgs[category]:
                # skip overlapping annotations
                if image_id in images:
                    continue
                # image should have been loaded during initialization
                if image_id not in self._image_info_dict:
                    raise RuntimeError("image of id '{}' not found".format(image_id))

                images[image_id] = self._image_info_dict[image_id]

        return images

    def get_bounding_boxes_by_ids(self, image_id, category_ids):
        if image_id not in self._image_info_dict:
            raise RuntimeError("image of id '{}' not found".format(image_id))

        annotations = self._coco.imgToAnns[image_id]

        # create dictionary { category_id: [ list of annotation indices ] }
        cat_to_ann_indices = {}
        for i, ann in enumerate(annotations):
            cat_id = ann['category_id']
            if cat_id in cat_to_ann_indices:
                cat_to_ann_indices[cat_id].append(i)
            else:
                cat_to_ann_indices[cat_id] = [i]

        # fill bounding boxes
        boxes = {}
        for cat_id in category_ids:
            if cat_id not in self._categories:
                raise ValueError("category ID '{}' is not in collection".format(cat_id))

            category = self._categories[cat_id]
            if category.parent_category in category_ids:
                raise RuntimeError("category '{}' (id '{}') is a child of super category '{}'"
                                   .format(category.name, category.category_id, category.parent_category))

            if cat_id not in boxes:
                boxes[cat_id] = []

            # get all unique annotation indices of a category and its sub-categories
            ann_indices = set(self._get_all_ann_indices(cat_to_ann_indices, cat_id))

            for ann_index in ann_indices:
                box = annotations[ann_index]['bbox']
                boxes[cat_id].append(BoundingBox(round(box[0]), round(box[1]),
                                                 width=round(box[2]), height=round(box[3]),
                                                 class_id=cat_id))

        return boxes

    def _get_all_ann_indices(self, cat_to_ann_indices, category_id):
        """
        recursively get annotation indices for all sub categories (if exist) of a category

        :param cat_to_ann_indices: { category_id: [ list of annotation indices ] }
        :param category_id:
        :return: list of annotation indices
        """
        if category_id in cat_to_ann_indices:
            return cat_to_ann_indices[category_id]

        if category_id not in self.category_hierarchy:
            return []

        sub_categories = self.category_hierarchy[category_id].get_sub_categories_recursive()
        indices = []
        for sub_cat_id in sub_categories:
            indices.extend(self._get_all_ann_indices(cat_to_ann_indices, sub_cat_id))

        return indices
