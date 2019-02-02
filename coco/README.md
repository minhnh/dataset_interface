# `dataset_interface` for COCO

Expected data directory structure:

```text
.
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
└── images
    ├── train2017
    └── val2017

```

Note: `val2017` or `train2017` are the split names, specified by `split_name` field in the
[config file](../config/sample_coco_configs.yml).
