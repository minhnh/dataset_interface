try:
    from pycocotools.coco import COCO
except ImportError:
    print('pycocotools not installed or installed incorrectly, see cocoapi/README.md for installation instruction')
    raise

from ..common.image_detection_data_api import ImageDetectionDataAPI


class COCODataAPI(ImageDetectionDataAPI):
    def __init__(self, data_dir):
        super().__init__(data_dir)
