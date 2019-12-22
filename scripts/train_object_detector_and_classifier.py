#!/usr/bin/env python3
from typing import Sequence, Dict

import os
import random
import yaml
import argparse

import math
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import dataset_interface.object_detection.dataset as dataset
from dataset_interface.object_detection.engine import train_one_epoch, validate_one_epoch
import dataset_interface.object_detection.utils as utils
import dataset_interface.object_detection.transforms as T
from dataset_interface.object_detection.dataset import DatasetCustom, DatasetVOC

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes: int):
    '''Returns a Faster R-CNN model pretrained on COCO and
    with a final layer matching the given number number of classes.

    num_classes: int -- number of classes in the model (including a background class)

    '''
    # we load Faster R-CNN pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # we get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # we finally replace the head of the pretrained model
    # so that it matches our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_class_metadata(class_metadata_file_path: str) -> Dict[int, str]:
    '''Returns a dictionary in which the keys are

    Keyword arguments:
    class_metadata_file_path: str -- path to a file with category metadata

    '''
    with open(class_metadata_file_path) as file:
        class_metadata = yaml.load(file, Loader=yaml.FullLoader)
    class_metadata[0] = '__background'
    return class_metadata

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data_path', type=str,
                           help='Directory containing training/validation data and data annotations',
                           default='/home/lucy/data/')
    argparser.add_argument('-c', '--class_metadata', type=str,
                           help='Path to a file with category metadata',
                           default='/home/lucy/data/class_metadata.yaml')
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a directory where the trained models (one per epoch) should be saved',
                           default='/home/lucy/data/model.pt')
    argparser.add_argument('-e', '--num_epochs', type=int,
                           help='Number of training epochs',
                           default=10)
    argparser.add_argument('-lr', '--learning_rate', type=float,
                           help='Initial learning rate',
                           default=1e-4)
    argparser.add_argument('-b', '--training_batch_size', type=int,
                           help='Training batch size',
                           default=1)
    argparser.add_argument('-l', '--train_loss_file_path', type=str,
                           help='Path to a file in which training losses will be saved',
                           default='/home/lucy/data/train_loss.log')
    argparser.add_argument('-v', '--val_loss_file_path', type=str,
                           help='Path to a file in which validation losses will be saved',
                           default='/home/lucy/data/val_loss.log')
    argparser.add_argument('-at', '--annotation_type', type=str,
                           choices=['voc', 'custom'],
                           help='Data annotation type (voc or custom)',
                           default='voc')

    # we read all arguments
    args = argparser.parse_args()
    data_path = args.data_path
    class_metadata_file_path = args.class_metadata
    model_path = args.model_path
    train_loss_file_path = args.train_loss_file_path
    val_loss_file_path = args.val_loss_file_path
    num_epochs = args.num_epochs
    training_batch_size = args.training_batch_size
    learning_rate = args.learning_rate
    annotation_type = args.annotation_type

    print('\nThe following arguments were read:')
    print('------------------------------------')
    print('data_path:               {0}'.format(data_path))
    print('class_metadata:          {0}'.format(class_metadata_file_path))
    print('model_path:              {0}'.format(model_path))
    print('train_loss_file_path:    {0}'.format(train_loss_file_path))
    print('val_loss_file_path:      {0}'.format(val_loss_file_path))
    print('num_epochs:              {0}'.format(num_epochs))
    print('training_batch_size:     {0}'.format(training_batch_size))
    print('learning_rate:           {0}'.format(learning_rate))
    print('annotation_type:         {0}'.format(annotation_type))
    print('------------------------------------')
    print('Proceed with training (y/n)')
    proceed = input()
    if proceed != 'y':
        print('Aborting training')
        sys.exit(1)

    # we read the class metadata
    class_metadata = get_class_metadata(class_metadata_file_path)
    num_classes = len(class_metadata.keys())

    # we create a data loader by instantiating an appropriate
    # dataset class depending on the annotation type
    dataset = None
    if annotation_type.lower() == 'voc':
        dataset = DatasetVOC(data_path, get_transform(train=True), 'train')
    else:
        dataset = DatasetCustom(data_path, get_transform(train=True), 'train')

    # we split the dataset into train and validation sets
    indices = torch.randperm(len(dataset)).tolist()
    train_split = math.ceil(0.7*len(indices))

    dataset_train = torch.utils.data.Subset(dataset, indices[0:train_split])
    dataset_val = torch.utils.data.Subset(dataset, indices[train_split:])

    # we define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=training_batch_size,
                                                    shuffle=True, num_workers=4,
                                                    collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=training_batch_size,
                                                  shuffle=True, num_workers=4,
                                                  collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes)

    # we now define an optimiser, and train
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # we move the model to the correct device before training
    model.to(device)

    # we clear the files in which the training and validation losses are saved
    open(train_loss_file_path, 'w').close()
    open(val_loss_file_path, 'w').close()

    # we create the model path directory if it doesn't exist
    if not os.path.isdir(model_path):
        print('Creating model directory {0}'.format(model_path))
        os.mkdir(model_path)

    print('Training model for {0} epochs'.format(num_epochs))
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch,
                        print_freq=10, loss_file_name=train_loss_file_path)
        lr_scheduler.step()
        validate_one_epoch(model, data_loader_val, device, epoch,
                           print_freq=10, loss_file_name=val_loss_file_path)

        torch.save(model.state_dict(), os.path.join(model_path, 'model_{0}.pt'.format(epoch)))
    print('Training done')
