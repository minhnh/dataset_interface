import numpy as np


class BoundingBox(object):
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    x_center = None
    y_center = None
    width = None
    height = None

    def __init__(self, x_min, y_min, x_max=None, y_max=None, width=None, height=None):
        self.x_min = x_min
        self.y_min = y_min
        if x_max is not None and y_max is not None:
            self.x_max = x_max
            self.y_max = y_max
            self.width = self.x_max - self.x_min
            self.height = self.y_max - self.y_min
        elif width is not None and height is not None:
            self.width = width
            self.height = height
            self.x_max = self.x_min + self.width
            self.y_max = self.y_min + self.height
        else:
            raise ValueError("either (width, height) or (x_max, y_max) has to be specified")
        self.x_center = self.x_min + self.width / 2
        self.y_center = self.y_min + self.height / 2


class NormalizedBox(BoundingBox):
    x_min_norm = None
    x_max_norm = None
    y_min_norm = None
    y_max_norm = None
    width_norm = None
    height_norm = None
    image_height = None
    image_width = None

    def __init__(self, image_size, x_min, y_min, x_max=None, y_max=None, width=None, height=None):
        super().__init__(x_min, y_min, x_max=x_max, y_max=y_max, width=width, height=height)

        self.image_height, self.image_width = image_size
        self.x_min_norm, self.x_max_norm = self.x_min / self.image_width, self.x_max / self.image_width
        self.y_min_norm, self.y_max_norm = self.y_min / self.image_height, self.y_max / self.image_height
        self.x_center_norm, self.y_center_norm = self.x_center / self.image_width, self.y_center / self.image_height
        self.width_norm, self.height_norm = self.width / self.image_width, self.height / self.image_height
        self.width_norm, self.height_norm = self.width / self.image_width, self.height / self.image_height


class SegmentedBox(NormalizedBox):
    """
    contains geometric info calculated from segmented 2D coordinates of an object
    all values are normalized to image dimensions
    """
    max_dimension_norm = None
    segmented_coords_homog_norm = None

    def __init__(self, x_coords, y_coords, image_size):
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        super().__init__(image_size, x_min, y_min, x_max=x_max, y_max=y_max)

        # create a homogeneous matrix from the normalized segmented coordinates for applying transformations
        self.segmented_coords_homog_norm = np.vstack((x_coords / self.image_width,
                                                      y_coords / self.image_height,
                                                      np.ones(len(x_coords))))
        self.segmented_coords_homog_norm = self.segmented_coords_homog_norm.transpose()

        # calculate normalized diagonal as maximum dimension
        self.max_dimension_norm = np.sqrt(self.width_norm**2 + self.height_norm**2)
