# Using `tensorflow/models`

This document describes how the tools and model definitions in the
[`tensorflow/models` repository](http://github.com/tensorflow/models) are used for training detection models for the
b-it-bots@Home team.

## Generate protobuf label map file

For b-it-bots@Home, a YAML file is used for annotating class names in the following format:

```YAML
1: cup
2: bowl
...
<class_id>: <class_name>
```

Note that `<class_id>` is the numeric output of the detection algorithm, and `<class_name>` is the descriptive string
which annotates the class. `<class_id>` starts at 1 to take into account the background (ID 0) class built into many
Single-Shot Multibox Detector (SSD) implementations. For example,
[`go_2019_classes.yml`](../configs/go_2019_classes.yml) contains the classes used in the
[RoboCup@Home 2019 German Open](https://www.robocupgermanopen.de/en/major/athome).

This YAML file needs to be converted to the `pbtxt` format expected by `tensorflow/models` scripts as label map file.
The format of `pbtxt` file is as follow:

```json
item {
  id: 1
  name: 'cup'
}

item {
  id: 2
  name: 'bowl'
}

...
item {
  id: <class_id>
  name: <class_name>
}
```

The script [`yaml_to_pbtxt_converter.py`](../scripts/yaml_to_pbtxt_converter.py) is provided to handle this conversion:

```sh
$ scripts/yaml_to_pbtxt_converter.py -h
usage: yaml_to_pbtxt_converter.py [-h] yaml_file pbtxt_file

Tool to convert YAML class annotations to pbtxt format

positional arguments:
  yaml_file   input YAML file containing class annotations
  pbtxt_file  output pbtxt file to write to

optional arguments:
  -h, --help  show this help message and exit
```

## Generate `TFRecord`'s from images and annotations

`TFRecord` is the data format expected by `tensorflow/models` scripts. It store images and annotations together as
binary files and can be broken into smaller chunks (i.e. shards) for easier management (see
[this tutorial](https://www.tensorflow.org/tutorials/load_data/tf_records) for more details).

The script [`create_robocup_tf_record.py`](../scripts/create_robocup_tf_record.py) handle the TFRecord generation:

* Command

    ```sh
    python3 create_robocup_tf_record.py \
        --output_dir ${OUTPUT_DIR}\
        --train_annotations_file  ${TRAIN_ANNOTATIONS_FILE} \
        --train_image_dir ${TRAIN_IMAGE_DIR} \
        --val_annotations_file ${VAL_ANNOTATIONS_FILE} \
        --val_image_dir ${VAL_IMAGE_DIR} \
        --classes_filename ${CLASSES_FILENAME}
    ```

    Example

    ```sh
    python3 create_robocup_tf_record.py \
        --output_dir tf_records \
        --train_annotations_file robocup_images/train_annotations.yml\
        --train_image_dir robocup_images/ \
        --val_annotations_file robocup_images/val_annotations.yml \
        --val_image_dir robocup_images/ \
        --classes_filename classes.yml
    ```

## Training process

* Command

    ```sh
    python3 object_detection/model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --model_dir=${MODEL_DIR} \
        --num_train_steps=${NUM_TRAIN_STEPS} \
        --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
        --alsologtostderr
    ```

    Example

    ```sh
    python3 object_detection/model_main.py \
        --pipeline_config_path data/ssd_resnet/pipeline.config \
        --model_dir data/trained_model \
        --num_train_steps 50000 \
        --sample_1_of_n_eval_examples 1 \
        --alsologtostderr
    ```

Notes:
    - protobuf label map and tf_records required
    - Script has to run from models/research
    - Execute `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
    - PIPELINE_CONFIG_PATH: path to the pipeline.config file with the information about the detection model and training process
        - Verify that the paths inside the pipeline.config match where the tf_records are
    - MODEL_DIR: path to the directory where the models checkpoints will be stored
    - NUM_TRAIN_STEPS: number of training steps for the training process
    - SAMPLE_1_OF_N_EVAL_EXAMPLES: TODO

## Generate frozen graph from checkpoint

* Command

    ```sh
    python3 object_detection/export_inference_graph.py \
        --input_type=${INPUT_TYPE} \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_prefix=${TRAINED_CHECKPOINT_PREFIX} \
        --output_directory=${OUTPUT_DIRECTORY}
    ```

    Example

    ```sh
    python3 object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ./pipeline.config \
        --trained_checkpoint_prefix ./model.ckpt-4953 \
        --output_directory ./frozen_output
    ```

Notes:
    - Script has to run from models/research
    - Execute `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
    - input_type has to be image_tensor
    - pipeline_config_path: path for the pipeline.config file used to trained the model
    - trained_checkpoint_prefix: path to the model.ckpt
    - output_directory: path to the directory where the frozen graph will be stored
