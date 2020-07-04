#!/usr/bin/env python3
import os
import argparse

import numpy as np
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset_interface.siamese_net.model import SiameseNetwork
from dataset_interface.siamese_net.dataset import SiameseNetworkDataset
from dataset_interface.siamese_net.loss import ContrastiveLoss
from dataset_interface.siamese_net.utils import get_transforms

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data_path', type=str,
                           help='Directory containing training',
                           default='/home/lucy/data/')
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a directory where the trained models (one per epoch) should be saved',
                           default='/home/lucy/models')
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

    # we read all arguments
    args = argparser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    train_loss_file_path = args.train_loss_file_path
    num_epochs = args.num_epochs
    training_batch_size = args.training_batch_size
    learning_rate = args.learning_rate

    print('\nThe following arguments were read:')
    print('------------------------------------')
    print('data_path:               {0}'.format(data_path))
    print('model_path:              {0}'.format(model_path))
    print('train_loss_file_path:    {0}'.format(train_loss_file_path))
    print('num_epochs:              {0}'.format(num_epochs))
    print('training_batch_size:     {0}'.format(training_batch_size))
    print('learning_rate:           {0}'.format(learning_rate))
    print('------------------------------------')
    print('Proceed with training (y/n)')
    proceed = input()
    if proceed != 'y':
        print('Aborting training')
        sys.exit(1)

    # we create a data loader by instantiating an appropriate
    # dataset class depending on the annotation type
    folder_dataset = torchvision.datasets.ImageFolder(root=data_path)
    siamese_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset,
                                            transform=get_transforms(),
                                            should_invert=False)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=training_batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # we move the model to the correct device before training
    model.to(device)

    # we create the model path directory if it doesn't exist
    if not os.path.isdir(model_path):
        print('Creating model directory {0}'.format(model_path))
        os.mkdir(model_path)

    # we clear the files in which the training and validation losses are saved
    open(train_loss_file_path, 'w').close()

    print('Training model for {0} epochs'.format(num_epochs))
    for epoch in range(num_epochs):
        losses = []
        for i, data in enumerate(train_dataloader):
            img0, img1, label = data
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch number {}\n Current loss {}\n'.format(epoch,
                                                                   loss_contrastive.item()))
            losses.append(loss_contrastive.item())
        avg_loss = np.mean(losses)

        if train_loss_file_path:
            with open(train_loss_file_path, 'a+') as loss_file:
                loss_file.write(str(avg_loss).split(' ')[0] + '\n')

        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(model_path, 'model_{0}.pt'.format(epoch)))
    print('Training done')
