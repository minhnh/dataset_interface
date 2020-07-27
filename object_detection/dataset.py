from typing import Sequence, Dict

import os
import yaml
import xmltodict

from PIL import Image
import torch

class DatasetCustom(object):
    def __init__(self, root, transforms, split_name):
        self.root = root
        self.transforms = transforms
        self.split_name = split_name
        self.data_list = []
        if os.path.isdir(self.root):
            self.data_list = self.__get_image_data(os.path.join(root, 'images', self.split_name),
                                                   os.path.join(root, self.split_name + '.yaml'))

    def __get_image_data(self, image_dir: str,
                         annotations_file_path: str) -> Sequence[Dict[str, str]]:
        '''Returns a list of dictionaries containing image paths and image
        annotations for all images in the given directory. Each dictionary
        in the resulting list is of the following format:
        {
            'img': path to an image (starting from image_dir),
            'annotations': a dictionary containing class labels and bounding
                           box annotations for all objects in the image
        }

        Keyword arguments:
        image_dir: str -- name of a directory with RGB images
        annotations_file_path: str -- path to an image annotation file

        '''
        image_list = []
        annotations = {}
        with open(annotations_file_path, 'r') as annotations_file:
            annotations = yaml.load(annotations_file)

        for x in os.listdir(image_dir):
            name, _ = x.split('.')
            image_data = {}
            img_name = name + '.jpg'

            image_data['img'] = os.path.join(image_dir, img_name)
            image_data['annotations'] = annotations[img_name]
            image_list.append(image_data)
        return image_list

    def __getitem__(self, idx: int):
        img_path = self.data_list[idx]['img']
        annotations = self.data_list[idx]['annotations']

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for annotation in annotations:
            xmin = annotation['xmin']
            xmax = annotation['xmax']
            ymin = annotation['ymin']
            ymax = annotation['ymax']
            boxes.append([xmin, ymin, xmax, ymax])

            label = annotation['class_id']
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data_list)


class DatasetVOC(object):
    def __init__(self, root, transforms, split_name, class_metadata):
        self.root = root
        self.transforms = transforms
        self.split_name = split_name

        self.data_list = []
        if os.path.isdir(self.root):
            self.data_list = self.__get_image_data(os.path.join(root, 'images', self.split_name),
                                                   os.path.join(root, 'annotations', self.split_name))

        self.class_metadata = class_metadata

    def __get_image_data(self, image_dir: str,
                         annotations_dir: str) -> Sequence[Dict[str, str]]:
        '''Returns a list of dictionaries containing image paths and image
        annotations for all images in the given directory. Each dictionary
        in the resulting list is of the following format:
        {
            'img': path to an image (starting from image_dir),
            'annotations': a dictionary containing class labels and bounding
                           box annotations for all objects in the image
        }

        Keyword arguments:
        image_dir: str -- name of a directory with RGB images
        annotations_file_path: str -- path to an image annotation file

        '''
        image_list = []

        for x in os.listdir(image_dir):
            name, _ = x.split('.')
            image_data = {}
            img_name = name + '.jpg'
            annotation_name = name + '.xml'

            image_data['img'] = os.path.join(image_dir, img_name)

            annotation = None
            with open(os.path.join(annotations_dir, annotation_name), 'r') as annotations_file:
                image_data['annotations'] = xmltodict.parse(annotations_file.read())

            image_list.append(image_data)
        return image_list

    def __getitem__(self, idx: int):
        img_path = self.data_list[idx]['img']
        annotations = self.data_list[idx]['annotations']

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        if type(annotations['annotation']['object']) == list:
            for annotation in annotations['annotation']['object']:
                xmin = int(annotation['bndbox']['xmin'])
                xmax = int(annotation['bndbox']['xmax'])
                ymin = int(annotation['bndbox']['ymin'])
                ymax = int(annotation['bndbox']['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])

                label = self.class_metadata[annotation['name']]
                labels.append(label)
        else:
            annotation = annotations['annotation']['object']
            xmin = int(annotation['bndbox']['xmin'])
            xmax = int(annotation['bndbox']['xmax'])
            ymin = int(annotation['bndbox']['ymin'])
            ymax = int(annotation['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])

            label = self.class_metadata[annotation['name']]
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data_list)
