import os
import yaml
import abc
import six


class ImageInfo(object):
    _id = None
    _url = None
    _image_path = None
    _file_name = None

    def __init__(self, image_id, directory, filename, url=None):
        self._id = image_id
        self._file_name = filename
        self._image_path = os.path.join(directory, self._file_name)
        self._url = url

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._url

    @property
    def image_path(self):
        return self._image_path


class Category(object):
    _category_id = None         # type: str
    _name = None                # type: str
    _parent_category = None     # type: str
    # dictionary mapping sub category id to the corresponding Category object
    _sub_categories = None      # type: dict

    def __init__(self, category_id, name, parent_category=None):
        """
        :param category_id: unique ID of class
        :param name: human readable class name
        :type name: str
        :param sub_categories: dictionary of sub categories
        :type sub_categories: dict
        """
        self._category_id = category_id
        self._name = name
        self._parent_category = parent_category
        self._sub_categories = {}

    @property
    def category_id(self):
        return self._category_id

    @property
    def name(self):
        return self._name

    @property
    def parent_category(self):
        return self._parent_category

    @property
    def sub_categories(self):
        return self._sub_categories

    def __str__(self):
        return 'Category: id={}, name={}'.format(self.category_id, self.name)

    def add_sub_categories(self, sub_categories):
        """
        :param sub_categories: list of Category instances
        :type sub_categories: list
        :rtype: None
        """
        for sub_category in sub_categories:
            self.add_sub_category(sub_category)

    def add_sub_category(self, sub_category):
        """
        :type sub_category: Category
        :rtype: None
        """
        if sub_category.category_id in self._sub_categories:
            return
        self._sub_categories[sub_category.category_id] = sub_category

    def get_sub_categories_recursive(self):
        """
        Recursively add sub categories into a dictionary of 'Category' objects
        :return: dictionary of all sub categories
        :rtype: dict
        """
        sub_categories = {}
        for key, value in self._sub_categories.items():
            # break if key exist
            if key in sub_categories:
                continue
            # break if current child has no sub_category
            if len(value.sub_categories) == 0:
                sub_categories[key] = value
                continue
            # recursive call to add child's sub-categories
            sub_categories.update(value.get_sub_categories_recursive())

        return sub_categories

    def is_sub_category(self, category_id):
        raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class ImageDetectionDataAPI(object):
    # directory containing dataset info, i.e annotations, category descriptions,...
    _data_dir = None            # type: str
    # directory containing images
    _image_dir = None            # type: str
    # contains configurations for the specific dataset API
    _configurations = None      # type: dict
    # contains category hierarchy of the dataset, map category ID to top level 'Category' objects
    _category_hierarchy = None  # type: dict
    # contain categories of all levels, map category ID to 'Category' object
    _categories = None          # type: dict
    # contain category names of all levels, map category name to category ID
    _category_names = None      # type: dict
    # dictionary contains images' metadata, maps images' unique ID's to ImageInfo objects
    _image_info_dict = None     # type: dict

    def __init__(self, data_dir, config_file_path=None):
        """
        :param data_dir: directory of the dataset
        :param config_file_path: YAML file containing the specific API configurations
        """
        if not os.path.exists(data_dir):
            raise IOError('data directory does not exist: ' + data_dir)

        self._data_dir = data_dir
        self._categories = {}
        self._category_names = {}
        self._category_hierarchy = {}
        self._image_info_dict = {}

        if config_file_path:
            if not os.path.exists(config_file_path):
                raise IOError('config file does not exist: {}'.format(config_file_path))
            with open(config_file_path) as config_file:
                self._configurations = yaml.load(config_file, Loader=yaml.FullLoader)
        else:
            self._configurations = {}

        if not self._configurations:
            raise ValueError('loading of configuration file "{}" failed'.format(config_file_path))

        self._initialize()
        self._parse_categories()
        self._parse_image_info()

    @property
    def category_hierarchy(self):
        return self._category_hierarchy

    @property
    def categories(self):
        return self._categories

    @property
    def category_names(self):
        """
        :return: dictionary mapping category names to ID's
        """
        return self._category_names

    @abc.abstractmethod
    def _initialize(self):
        """
        Contains specific initializations for the extension class, i.e. pointing to correct subdirectories

        :rtype: None
        """
        raise NotImplementedError("abstract method '_initialize' not implemented")

    @abc.abstractmethod
    def _parse_categories(self):
        """
        Handles indexing the category hierarchy, '_category_hierarchy' should be filled here

        :rtype: None
        """
        raise NotImplementedError("abstract method '_parse_categories' not implemented")

    @abc.abstractmethod
    def _parse_image_info(self):
        """
        Handles indexing image metadata, '_image_info_dict' should be filled here

        :rtype: None
        """
        raise NotImplementedError("abstract method '_parse_image_info' not implemented")

    @abc.abstractmethod
    def get_images_in_category(self, category_id):
        """
        Gets all images which contain a certain category. Category can be of any level.

        :param category_id:
        :return:
        """
        raise NotImplementedError("abstract method 'get_images_in_category' not implemented")

    def get_images_and_boxes_in_categories(self, category_ids):
        """
        get all ImageInfo objects of images that have the given categories, with their relevant
        bounding box annotations
        @param category_ids: list of unique category ID's
        @return: dictionary containing mapping
                 { <image_id>: { 'info': ImageInfo, 'bounding_boxes': { category_id: [ BoundingBox ] } } }
        """
        image_dict = {}
        for category_id in category_ids:
            for image_id, image_inf in self.get_images_in_category(category_id).items():
                if image_id in image_dict:
                    # skip if current image is already added
                    continue
                boxes = self.get_bounding_boxes_by_ids(image_id, category_ids)
                image_dict[image_id] = {'info': image_inf, 'bounding_boxes': boxes}

        return image_dict

    @abc.abstractmethod
    def get_bounding_boxes_by_ids(self, image_id, category_ids):
        """
        Get all bounding boxes for the specified category ID's in a given image, should work for all levels.
        Querying for parent category should return boxes of all child categories.

        :type image_id: str
        :type category_ids: list
        :return: dictionary of boxes, maps { category_id: [ BoundingBox ]}
        :rtype: dict
        """
        raise NotImplementedError("abstract method 'get_bounding_boxes' not implemented")

    def get_bounding_boxes_by_names(self, image_id, category_names):
        """
        Get all bounding boxes for the specified category names in a given image.
        Call 'get_bounding_boxes_by_ids' after parsing category names

        :type category_names: list
        """
        cat_ids = []
        for cat_name in category_names:
            if cat_name not in self._category_names:
                raise ValueError("'{}' is not a recognized category name".format(cat_name))
            cat_ids.append(self._category_names[cat_name])

        return self.get_bounding_boxes_by_ids(image_id, cat_ids)

    def get_category_id(self, category_name):
        """
        look up category ID by name
        """
        if category_name in self._category_names:
            return self._category_names[category_name]
        return None

    def get_sub_categories(self, category_id):
        """
        :rtype: dict
        """
        if category_id not in self._categories:
            raise ValueError('did not find category id "{}" in dataset'.format(category_id))
        return self._categories[category_id].get_sub_categories_recursive()

    def get_sub_category_names(self, category_id):
        sub_categories = self.get_sub_categories(category_id)
        return {k: v.name for k, v in sub_categories.items()}
