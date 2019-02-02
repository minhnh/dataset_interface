import os
import yaml
from abc import ABC, abstractmethod


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
    # dictionary mapping sub category id to the corresponding Category object
    _sub_categories = None      # type: dict

    def __init__(self, category_id, name):
        """
        :param category_id: unique ID of class
        :type category_id: str
        :param name: human readable class name
        :type name: str
        :param sub_categories: dictionary of sub categories
        :type sub_categories: dict
        """
        self._category_id = category_id
        self._name = name
        self._sub_categories = {}

    @property
    def category_id(self):
        return self._category_id

    @property
    def name(self):
        return self._name

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


class ImageDetectionDataAPI(ABC):
    # directory containing dataset info, i.e annotations, category descriptions,...
    _data_dir = None            # type: str
    # directory containing images
    _image_dir = None            # type: str
    # contains configurations for the specific dataset API
    _configurations = None      # type: dict
    # contains category hierarchy of the dataset
    _category_hierarchy = None  # type: dict
    # contain category names of all levels
    _categories = None          # type:dict
    # dictionary of ImageInfo objects
    _image_info_collection = None

    def __init__(self, data_dir, config_file_path=None):
        """
        :param data_dir: directory of the dataset
        :param config_file_path: YAML file containing the specific API configurations
        """
        if not os.path.exists(data_dir):
            raise IOError('data directory does not exist: ' + data_dir)

        self._data_dir = data_dir
        self._categories = {}
        self._category_hierarchy = {}
        self._image_info_collection = {}

        if config_file_path:
            with open(config_file_path) as config_file:
                self._configurations = yaml.load(config_file)
        else:
            self._configurations = {}

        if not self._configurations:
            raise ValueError('loading of configuration file "{}" failed'.format(config_file_path))

        self._initialize()
        self._parse_categories()

    @property
    def category_hierarchy(self):
        return self._category_hierarchy

    @abstractmethod
    def _initialize(self):
        raise NotImplementedError('abstract method "_initialize" not implemented')

    @abstractmethod
    def _parse_categories(self):
        raise NotImplementedError('abstract method "_parse_categories" not implemented')

    @abstractmethod
    def get_images_in_category(self, category_id):
        raise NotImplementedError('abstract method "get_images_in_category" not implemented')

    def get_sub_categories(self, category_id):
        """
        :type category_id: str
        :rtype: dict
        """
        if category_id not in self._categories:
            raise ValueError('did not find category id "{}" in dataset'.format(category_id))
        return self._categories[category_id].get_sub_categories_recursive()

    def get_sub_category_names(self, category_id):
        sub_categories = self.get_sub_categories(category_id)
        return {k: v.name for k, v in sub_categories.items()}
