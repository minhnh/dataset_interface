from abc import ABC, abstractmethod
import yaml


class Category(object):
    _category_id = None     # type: str
    _name = None            # type: str
    _sub_categories = None  # type: dict

    def __init__(self, category_id, name, sub_categories=None):
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
        if sub_categories is not None:
            self._sub_categories = sub_categories
        else:
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

    def add_sub_categories(self, sub_categories):
        """
        :param sub_categories: list of ObjectClass instances
        :type sub_categories: list
        :return:
        """
        for sub_category in sub_categories:
            self.add_sub_category(sub_category)

    def add_sub_category(self, sub_category):
        if sub_category.category_id in self._sub_categories:
            return
        self._sub_categories[sub_category.category_id] = sub_category

    def get_sub_categories_recursive(self):
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
    # directory containing images, annotations, category descriptions,...
    _data_dir = None            # type: str
    # contains configurations for the specific dataset API
    _configurations = None      # type: dict
    # contains category hierarchy of the dataset
    _category_hierarchy = None     # type: dict

    def __init__(self, data_dir, config_file_path):
        """
        :param data_dir: directory of the dataset
        :param config_file_path: YAML file containing the specific API configurations
        """
        self._data_dir = data_dir
        self._category_hierarchy = {}

        with open(config_file_path) as config_file:
            self._configurations = yaml.load(config_file)

        if not self._configurations:
            raise ValueError('loading of configuration file "{}" failed'.format(config_file_path))

        self._initialize()
        self._parse_categories()

    @property
    def category_hierarchy(self):
        return self._category_hierarchy

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _parse_categories(self):
        pass

    def get_sub_category_names(self, category_id):
        raise NotImplementedError()
