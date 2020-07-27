#!/usr/bin/env python3
import argparse

import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import torch
import torch.nn as nn

from dataset_interface.siamese_net.model import SiameseNetwork
from dataset_interface.siamese_net.utils import get_grayscale_image_tensor

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i1', '--image1_path', type=str,
                           help='Path to an anchor image',
                           default='')
    argparser.add_argument('-i2', '--image2_path', type=str,
                           help='Path to a target image',
                           default='')
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a trained model',
                           default='/home/lucy/data/model.pt')

    args = argparser.parse_args()
    img1_path = args.image1_path
    img2_path = args.image2_path
    model_path = args.model_path

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    img0 = get_grayscale_image_tensor(img1_path)
    img1 = get_grayscale_image_tensor(img2_path)

    out1, out2 = model(img0, img1)
    distance = nn.functional.pairwise_distance(out1, out2)
    print('Distance: {0}'.format(distance.item()))
