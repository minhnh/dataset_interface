from abc import ABC, abstractmethod
import yaml


class ObjectClass(object):
    _class_id = None        # type: str
    _name = None            # type: str
    _sub_classes = None      # type: dict

    def __init__(self, class_id, name, sub_classes=None):
        """
        :param class_id: unique ID of class
        :type class_id: str
        :param name: human readable class name
        :type name: str
        :param sub_classes: dictionary of sub classes
        :type sub_classes: dict
        """
        self._class_id = class_id
        self._name = name
        if sub_classes is not None:
            self._sub_classes = sub_classes
        else:
            self._sub_classes = {}

    @property
    def class_id(self):
        return self._class_id

    @property
    def name(self):
        return self._name

    @property
    def sub_classes(self):
        return self._sub_classes

    def add_sub_classes(self, subclasses):
        """
        :param subclasses: list of ObjectClass instances
        :type subclasses: list
        :return:
        """
        for subclass in subclasses:
            self.add_subclass(subclass)

    def add_subclass(self, subclass):
        if subclass.class_id in self._sub_classes:
            return
        self._sub_classes[subclass.class_id] = subclass

    def get_subclasses_recursive(self):
        subclasses = {}
        for key, value in self._sub_classes.items():
            # break if key exist
            if key in subclasses:
                continue
            # break if current child has no subclass
            if len(value.sub_classes) == 0:
                subclasses[key] = value
                continue
            # recursive call to add child's subclasses
            subclasses.update(value.get_subclasses_recursive())

        return subclasses

    def is_subclass(self, class_id):
        raise NotImplementedError()


class ImageDetectionDataAPI(ABC):
    # directory containing images, annotations, class descriptions,...
    _data_dir = None            # type: str
    # contains configurations for the specific dataset API
    _configurations = None      # type: dict
    # contains class hierarchy of the dataset
    _class_hierarchy = None     # type: dict

    def __init__(self, data_dir, config_file_path):
        """
        :param data_dir: directory of the dataset
        :param config_file_path: YAML file containing the specific API configurations
        """
        self._data_dir = data_dir
        self._class_hierarchy = {}

        with open(config_file_path) as config_file:
            self._configurations = yaml.load(config_file)

        if not self._configurations:
            raise ValueError('loading of configuration file "{}" failed'.format(config_file_path))

        self._initialize()
        self._parse_classes()

    @property
    def class_hierarchy(self):
        return self._class_hierarchy

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _parse_classes(self):
        pass

    def get_subclass_names(self, class_id):
        raise NotImplementedError()
