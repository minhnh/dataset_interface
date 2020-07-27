# Image Comparison: Dataset Format, Training, and Inference

We use a Siamese network for comparing images (e.g. objects or faces). As in the case of object detection, we use a PyTorch-based model for this purpose.

# Dataset Format

The Siamese network training script expects a data directory with multiple subdirectories - one per object - where each such directory contains images corresponding to the same object. The expected directory structure is shown below:

```
    train_dataset_dir
    |    obj_1
    |    |    1.jpg
    |    |    ...
    |    |____n.jpg
    |    obj_2
    |    |    1.jpg
    |    |    ...
    |    |____n.jpg
    |    ...
    |____obj_n
         |    1.jpg
         |    ...
         |____n.jpg
```

# Training a Comparison Model

For training a comparison model, the `train_siamese_net.py` script, which can be found under `scripts`, should be used. The script accepts the following arguments:

* `-d --data_path`: Directory containing training (default `/home/lucy/data`)
* `-m --model_path`: Path to a directory where the trained models (one per epoch) should be saved (default `/home/lucy/models`)
* `-e --num_epochs`: Number of training epochs (default `10`)
* `-lr --learning_rate`: Initial learning rate (default `1e-4`)
* `-b --training_batch_size`: Training batch size (default `1`)
* `-l --train_loss_file_path`: Path to a file in which training losses will be saved (default `/home/lucy/data/train_loss.log`)

An example call is given below:

```
./train_siamese_net.py \
-d /home/lucy/images/ \
-m /home/lucy/models \
-e 10 \
-lr 0.0001 \
-b 64 \
-l /home/lucy/training_loss.log
```

# Using a Comparison Model

The script `test_siamese_net.py`, also included under `scripts`, illustrates the use of a trained model for calculating the difference between two images. This script takes the following arguments:

* `-i1 --image1_path`: Path to an image (default `''`)
* `-i2 --image2_path`: Path to an image to be compared (default `''`)
* `-m --model_path`: Path to a trained model (default `/home/lucy/data/model.pt`)

An example call is given below:

```
./test_siamese_net.py \
-i1 /home/lucy/img1.jpg \
-i2 /home/lucy/img2.jpg \
-m /home/lucy/models/model.pt
```
