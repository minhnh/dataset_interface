from ..common.image_detection_data_api import ImageDetectionDataAPI


class COCODataAPI(ImageDetectionDataAPI):
    def __init__(self, data_dir):
        super().__init__(data_dir)
