# Object Detection: Dataset/Annotation Format Description, Training, and Inference

We use Faster R-CNN for object detection and classification. The model we use is defined in [`torchvision`][https://pytorch.org/docs/stable/torchvision/models.html] and is pretrained on the COCO data set, so we only finetune the class predictor on our data.

# Class Metadata

For mapping class indices to actual class labels, we use a simple YAML file with the following format:

```
1: class_1_label
...
n: class_n_label
```

This file is required both before training and inference since we instantiate a model with the number of classes provided there.

# Data Set Format

We support two different data annotation formats:
* PASCAL VOC
* custom YAML-based annotations as generated during our data augmentation process

## PASCAL VOC Annotations

If PASCAL VOC annotations are used, we expect a directory structure with both images and annotations, where the names of the image and annotation files are the same. In order to allow for different data splits (e.g. training and testing), both image and annotation directories are expected to have dedicated subdirectories for those splits.

This is best illustrated by an example. Let us suppose that we have two data splits - `train` and `test`. In this case, the expected directory structure would be given as follows:

```
    dataset_dir
    |    images
    |     |   train
    |     |   |    1.jpg
    |     |   |    ...
    |     |   |____n.jpg
    |     |___test
    |         |    1.jpg
    |         |    ...
    |         |____n.jpg
    |____annotations
         |   train
         |   |    1.jpg
         |   |    ...
         |   |____n.jpg
         |___test
             |    1.jpg
             |    ...
             |____n.jpg

```

## Custom Annotations

In the case of custom annotations, we expect a directory structure with an annotation file and an image subdirectory.

The annotation file is a YAML file which contains object annotations (labels and object bounding boxes) for each image. The file has the following format:
```
img_1_name:
    - class_id: <int>
      xmin: <int>
      xmax: <int>
      ymin: <int>
      ymax: <int>
    - class_id: <int>
      xmin: <int>
      xmax: <int>
      ymin: <int>
      ymax: <int>
    ...
...
img_n_name:
    - class_id: <int>
      xmin: <int>
      xmax: <int>
      ymin: <int>
      ymax: <int>
    - class_id: <int>
      xmin: <int>
      xmax: <int>
      ymin: <int>
      ymax: <int>
    ...
```

In order to allow for different data splits (e.g. training and testing), the image directory is expected to have dedicated subdirectories for those splits; in this case, separate annotations files are expected.

This is best illustrated by an example. Let us suppose that we have two data splits - `train` and `test`. In this case, the expected directory structure would be given as follows:

```
    dataset_dir
    |    images
    |     |   train
    |     |   |    1.jpg
    |     |   |    ...
    |     |   |____n.jpg
    |     |___test
    |         |    1.jpg
    |         |    ...
    |         |____n.jpg
    |    train.yaml
    |____test.yaml
```

# Training an Object Detector

A detailed description of the training workflow is given in the [faster_rcnn_torch.ipynb](../notebooks/faster_rcnn_torch.ipynb) notebook.

In general however, the `train_object_detector_and_classifier.py` should be used for training since it automates the complete training process. The script accepts the following arguments:

* `-d --data_path`: Directory containing training/validation data and data annotations (default `/home/lucy/data`)
* `-c --class_metadata`: Path to a file with category metadata (default `/home/lucy/data/class_metadata.yaml`)
* `-m --model_path`: Path to a directory where the trained models (one per epoch) should be saved (default `/home/lucy/models`)
* `-e --num_epochs`: Number of training epochs (default `10`)
* `-lr --learning_rate`: Initial learning rate (default `1e-4`)
* `-b --training_batch_size`: Training batch size (default `1`)
* `-l --train_loss_file_path`: Path to a file in which training losses will be saved (default `/home/lucy/data/train_loss.log`)
* `-v --val_loss_file_path`: Path to a file in which validation losses will be saved (default `/home/lucy/data/val_loss.log`)
* `-at --annotation_type`: Data annotation type (voc or custom) (default `voc`)

An example call is given below:

```
./train_object_detector_and_classifier.py \
-d /home/lucy/images/ \
-c /home/lucy/classes.yaml \
-e 10 \
-b 1 \
-lr 0.0001 \
-l /home/lucy/training_loss.log \
-v /home/lucy/val_loss.log \
-m /home/lucy/models \
-at custom
```

# Using an Object Detector

A description of how a trained model can be used for inference is also given in the [faster_rcnn_torch.ipynb](../notebooks/faster_rcnn_torch.ipynb) notebook.

For convenience, we also include a script `test_object_detector_and_classifier.py` under `scripts`, which allows detecting and classifying objects in a given image as well as plotting the detections. This script takes the following arguments:

* `-i --image_path`: Path to an image (default `''`)
* `-m --model_path`: Path to a trained model (default `/home/lucy/data/model.pt`)
* `-c --class_metadata`: Path to a file with category metadata (default `/home/lucy/data/class_metadata.yaml`)
* `-t --detection_threshold`: Detections score threshold (default `0.8`)

An example call is given below:

```
./test_object_detector_and_classifier.py \
-i /home/lucy/img.jpg \
-m /home/lucy/models/model.pt \
-c /home/lucy/classes.yaml
```
