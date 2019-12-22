#!/usr/bin/env python3
import argparse

import torch
import torchvision

import dataset_interface.object_detection.utils as utils

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--image_path', type=str,
                           help='Path to an image',
                           default='')
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a trained model',
                           default='/home/lucy/data/model.pt')
    argparser.add_argument('-c', '--class_metadata', type=str,
                           help='Path to a file with category metadata',
                           default='/home/lucy/data/class_metadata.yaml')
    argparser.add_argument('-t', '--detection_threshold', type=float,
                           help='Detections score threshold (default 0.8)',
                           default=0.8)

    args = argparser.parse_args()
    img_path = args.image_path
    model_path = args.model_path
    class_metadata_file_path = args.class_metadata
    detection_threshold = args.detection_threshold

    class_metadata = utils.get_class_metadata(class_metadata_file_path)
    num_classes = len(class_metadata.keys())
    model = utils.get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    utils.detect_objects(model, device, img_path, class_metadata, threshold=detection_threshold)
