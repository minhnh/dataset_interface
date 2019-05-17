import os
import yaml
import numpy as np
import cv2
from skimage.transform import SimilarityTransform, matrix_transform
from dataset_interface.utils import TerminalColors, glob_extensions_in_directory, ALLOWED_IMAGE_EXTENSIONS, \
                                    display_image_and_wait, prompt_for_yes_or_no, print_progress
from dataset_interface.augmentation.background_segmentation import get_image_mask_path


def apply_random_transformation(background_size, segmented_box, margin=0.1, max_obj_size_in_bg=0.4):
    """create transformation for 2D coordinates nomalized to image size"""
    orig_coords_norm = segmented_box.segmented_coords_homog_norm[:, :2]
    # translate object coordinates to the object center's frame, i.e. whitens
    whitened_coords_norm = orig_coords_norm - (segmented_box.midpoint_x_norm, segmented_box.midpoint_y_norm)

    # then generate a random rotation around the z-axis (perpendicular to the image plane), and limit the object scale
    # to maximum (default) 50% of the background image, i.e. the normalized largest dimension of the object must be at
    # most 0.5. To put it simply, scale the objects down if they're too big.
    # TODO(minhnh) add shear
    max_scale = 1.
    if segmented_box.max_dimension_norm > max_obj_size_in_bg:
        max_scale = max_obj_size_in_bg / segmented_box.max_dimension_norm
    random_rot_angle = np.random.uniform(0, np.pi)
    rand_scale = np.random.uniform(0.5*max_scale, max_scale)

    # generate a random translation within the image boundaries for whitened, normalized coordinates, taking into
    # account the maximum allowed object dimension. After this translation, the normalized coordinates should
    # stay within [margin, 1-margin] for each dimension
    scaled_max_dimension = segmented_box.max_dimension_norm * max_scale
    low_norm_bound, high_norm_bound = ((scaled_max_dimension / 2) + margin, 1 - margin - (scaled_max_dimension / 2))
    random_translation_x = np.random.uniform(low_norm_bound, high_norm_bound)
    random_translation_y = np.random.uniform(low_norm_bound, high_norm_bound)

    # create the transformation matrix for the generated rotation, translation and scale
    tf_matrix = SimilarityTransform(rotation=random_rot_angle, scale=rand_scale,
                                    translation=(random_translation_x, random_translation_y)).params

    # apply transformation
    transformed_coords_norm = matrix_transform(whitened_coords_norm, tf_matrix)
    return transformed_coords_norm


class BoundingBox(object):
    min_x_norm = None
    max_x_norm = None
    min_y_norm = None
    max_y_norm = None
    width_norm = None
    height_norm = None
    midpoint_x_norm = None
    midpoint_y_norm = None
    max_dimension_norm = None
    orig_image_width = None
    orig_image_height = None
    segmented_coords_homog_norm = None

    def __init__(self, x_coords, y_coords, image_size):
        self.orig_image_height, self.orig_image_width = image_size

        # create a homogeneous matrix from the normalized segmented coordinates for applying transformations
        self.segmented_coords_homog_norm = np.vstack((x_coords / self.orig_image_width,
                                                      y_coords / self.orig_image_height,
                                                      np.ones(len(x_coords))))
        self.segmented_coords_homog_norm = self.segmented_coords_homog_norm.transpose()

        # calculate normalized pixel extrema from the normalized segmented coordinates
        self.min_x_norm, self.max_x_norm = (np.min(self.segmented_coords_homog_norm[:, 0]),
                                            np.max(self.segmented_coords_homog_norm[:, 0]))
        self.min_y_norm, self.max_y_norm = (np.min(self.segmented_coords_homog_norm[:, 1]),
                                            np.max(self.segmented_coords_homog_norm[:, 1]))

        # calculate normalized pixel dimensions
        self.width_norm = self.max_x_norm - self.min_x_norm
        self.height_norm = self.max_y_norm - self.min_y_norm
        self.midpoint_x_norm = self.min_x_norm + self.width_norm / 2
        self.midpoint_y_norm = self.min_y_norm + self.height_norm / 2
        self.max_dimension_norm = np.sqrt(self.width_norm**2 + self.height_norm**2)


class SegmentedObject(object):
    _max_dimension = None
    _bgr_image = None
    _mask_image = None
    _segmented_box = None

    def __init__(self, image_path, mask_path):
        # read color and mask images
        self._bgr_image = cv2.imread(image_path)
        self._mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        segmented_y_coords, segmented_x_coords = np.where(self._mask_image)

        # calculate maximum pixel dimension from the segmented coordinates
        self._segmented_box = BoundingBox(segmented_x_coords, segmented_y_coords, self._mask_image.shape)

    @property
    def segmented_box(self):
        return self._segmented_box

    @property
    def segmented_coords_homog(self):
        return self._segmented_coords_homog

    @property
    def bgr_image(self):
        return self._bgr_image

    def view_segmented_color_img(self):
        segmented_bgr = cv2.bitwise_and(self._bgr_image, self._bgr_image, mask=self._mask_image)
        display_image_and_wait(segmented_bgr, 'segmented_object')


class SegmentedObjectCollection(object):
    """Collection which instances of 'SegmentedObject''s, one for each image-mask pair"""
    _class_id = None
    _segmented_objects = None

    def __init__(self, class_id, obj_img_dir, obj_mask_dir):
        self._class_id = class_id

        # check directories
        if not os.path.isdir(obj_img_dir):
            raise RuntimeError("image folder '{}' is not an existing directory".format(obj_img_dir))
        if not os.path.isdir(obj_mask_dir):
            raise RuntimeError("mask folder '{}' is not an existing directory".format(obj_mask_dir))

        # glob object images
        obj_img_paths = glob_extensions_in_directory(obj_img_dir, ALLOWED_IMAGE_EXTENSIONS)
        if not obj_img_paths:
            raise RuntimeError("found no image of supported types in '{}'".format(obj_img_dir))

        # load images and masks
        print("found '{}' object images in '{}'".format(len(obj_img_paths), obj_img_dir))
        self._segmented_objects = []
        for img_path in obj_img_paths:
            # check expected path to mask
            mask_path = get_image_mask_path(img_path, obj_mask_dir)
            if not os.path.exists(mask_path):
                TerminalColors.formatted_print("skipping image '{}': mask '{}' does not exist"
                                               .format(img_path, mask_path), TerminalColors.WARNING)
                continue

            # add SegmentedObject instance for image-mask pair
            try:
                self._segmented_objects.append(SegmentedObject(img_path, mask_path))
            except Exception as e:
                TerminalColors.formatted_print("failed to process image '{}' and mask '{}': {}"
                                               .format(img_path, mask_path, e), TerminalColors.FAIL)
                raise

    @property
    def class_id(self):
        return self._class_id

    def project_segmentation_on_background(self, background_image):
        # choose random segmentation from collection
        rand_segmentation = np.random.choice(self._segmented_objects)

        # create a random transformation
        bg_height, bg_width = background_image.shape[:2]
        transformed_coords_norm = apply_random_transformation((bg_height, bg_width), rand_segmentation.segmented_box)

        # denormalize transformed coordinates, preserving aspect ratio, but making sure the object is within
        # background frame
        denorm_factor = min(bg_height, bg_width)
        transformed_coords = np.array(transformed_coords_norm * denorm_factor, dtype=int)
        background_image[transformed_coords[:, 1], transformed_coords[:, 0], :] = (255, 0, 0)
        new_box = BoundingBox(transformed_coords[:, 0], transformed_coords[:, 1], (bg_height, bg_width))
        return background_image, new_box


class ImageAugmenter(object):
    _data_dir = None
    _greenbox_image_dir = None
    _mask_dir = None
    _augment_backgrounds = None
    _class_dict = None
    _segmented_object_collections = None

    def __init__(self, data_dir, class_annotation_file):
        # check required directories
        self._data_dir = data_dir
        TerminalColors.formatted_print('Data directory: ' + self._data_dir, TerminalColors.OKBLUE)
        if not os.path.isdir(self._data_dir):
            raise RuntimeError("'{}' is not an existing directory".format(self._data_dir))

        self._greenbox_image_dir = os.path.join(self._data_dir, 'green_box_images')
        print('will look for object green box images in: ' + self._greenbox_image_dir)
        if not os.path.isdir(self._greenbox_image_dir):
            raise RuntimeError("'{}' is not an existing directory".format(self._greenbox_image_dir))

        self._mask_dir = os.path.join(self._data_dir, 'object_masks')
        print('will look for object masks in: ' + self._mask_dir)
        if not os.path.isdir(self._mask_dir):
            raise RuntimeError("'{}' is not an existing directory".format(self._mask_dir))

        augment_bg_dir = os.path.join(self._data_dir, 'augmentation_backgrounds')
        print('will look for backgrounds for image augmentation in: ' + augment_bg_dir)
        if not os.path.isdir(augment_bg_dir):
            raise RuntimeError("'{}' is not an existing directory".format(augment_bg_dir))

        # load backgrounds for image augmentation
        background_paths = glob_extensions_in_directory(augment_bg_dir, ALLOWED_IMAGE_EXTENSIONS)
        print("found '{}' background images".format(len(background_paths)))
        self._augment_backgrounds = []
        for bg_path in background_paths:
            bg_img = cv2.imread(bg_path)
            self._augment_backgrounds.append(bg_img)

        # load class annotation file
        TerminalColors.formatted_print('Class annotation file: {}'.format(class_annotation_file),
                                       TerminalColors.OKBLUE)
        if not os.path.exists(class_annotation_file):
            raise RuntimeError('class annotation file does not exist: ' + class_annotation_file)
        with open(class_annotation_file, 'r') as infile:
            self._class_dict = yaml.load(infile, Loader=yaml.FullLoader)

        # load segmented objects
        TerminalColors.formatted_print("Loading object masks for '{}' classes".format(len(self._class_dict)),
                                       TerminalColors.OKBLUE)
        self._segmented_object_collections = {}
        for cls_id, cls_name in self._class_dict.items():
            obj_img_dir = os.path.join(self._greenbox_image_dir, cls_name)
            obj_mask_dir = os.path.join(self._mask_dir, cls_name)

            try:
                TerminalColors.formatted_print("loading images and masks for class '{}'".format(cls_name),
                                               TerminalColors.BOLD)
                segmented_obj = SegmentedObjectCollection(cls_id, obj_img_dir, obj_mask_dir)
                self._segmented_object_collections[cls_id] = segmented_obj
            except RuntimeError as e:
                TerminalColors.formatted_print("skipping class '{}': {}".format(cls_name, e), TerminalColors.WARNING)
                continue

            # test_obj = self._segmented_object_collections[cls_id]._segmented_objects[0]
            # print(test_obj._segmented_coords_homog.shape)
            # test_obj.view_segmented_color_img()

    def _sample_classes(self, max_obj_num_per_bg):
        # TODO(minhnh) ensure balance sampling
        num_obj = np.random.randint(1, max_obj_num_per_bg + 1)
        sampled_class_ids = np.random.choice(list(self._segmented_object_collections.keys()), num_obj)
        return [self._segmented_object_collections[cls_id] for cls_id in sampled_class_ids]

    def _generate_single_image(self, background_image, max_obj_num_per_bg):
        sampled_collections = self._sample_classes(max_obj_num_per_bg)
        annotations = []
        bg_img_copy = background_image.copy()
        for obj_collection in sampled_collections:
            generated_ann = {'class_id': obj_collection.class_id}
            bg_img_copy, box = obj_collection.project_segmentation_on_background(bg_img_copy)
            generated_ann = {'bounding_box': box}
            annotations.append(generated_ann)
        return bg_img_copy, annotations

    def generate_detection_data(self, split_name, output_dir, output_annotation_dir,
                                num_image_per_bg, max_obj_num_per_bg):
        split_output_dir = os.path.join(output_dir, split_name)
        TerminalColors.formatted_print("generating images for split '{}' under '{}'"
                                       .format(split_name, split_output_dir), TerminalColors.BOLD)
        # check output image directory
        if not os.path.isdir(split_output_dir):
            print("creating directory: " + split_output_dir)
            os.mkdir(split_output_dir)
        elif os.listdir(split_output_dir):
            if not prompt_for_yes_or_no("directory '{}' not empty. Overwrite?".format(split_output_dir)):
                raise RuntimeError("not overwriting '{}'".format(split_output_dir))

        # check output annotation file
        annotation_path = os.path.join(output_annotation_dir, split_name + '.yml')
        TerminalColors.formatted_print("generating annotations for split '{}' in '{}'"
                                       .format(split_name, annotation_path), TerminalColors.BOLD)
        if os.path.isfile(annotation_path):
            if not prompt_for_yes_or_no("file '{}' exists. Overwrite?".format(annotation_path)):
                raise RuntimeError("not overwriting '{}'".format(annotation_path))

        # store a reasonable value for the maximum number of objects projected onto each background
        if max_obj_num_per_bg <= 0 or max_obj_num_per_bg > len(self._class_dict):
            max_obj_num_per_bg = len(self._class_dict)

        # generate images and annotations
        img_cnt = 0
        total_img_cnt = num_image_per_bg * len(self._augment_backgrounds)
        for bg_img in self._augment_backgrounds:
            for _ in range(num_image_per_bg):
                print_progress(img_cnt + 1, total_img_cnt, prefix="creating image ", fraction=0.05)
                generated_image, annotations = self._generate_single_image(bg_img, max_obj_num_per_bg)
                display_image_and_wait(generated_image, 'synthetic image')
                img_cnt += 1
