import os
import yaml
import numpy as np
import cv2
from skimage.transform import SimilarityTransform, matrix_transform
from dataset_interface.common import SegmentedBox
from dataset_interface.utils import TerminalColors, glob_extensions_in_directory, ALLOWED_IMAGE_EXTENSIONS, \
                                    display_image_and_wait, prompt_for_yes_or_no, print_progress, cleanup_mask, \
                                    draw_labeled_boxes
from dataset_interface.augmentation.background_segmentation import get_image_mask_path


def apply_random_transformation(background_size, segmented_box, margin=0.1, max_obj_size_in_bg=0.4):
    """apply a random transformation to 2D coordinates nomalized to image size"""
    orig_coords_norm = segmented_box.segmented_coords_homog_norm[:, :2]
    # translate object coordinates to the object center's frame, i.e. whitens
    whitened_coords_norm = orig_coords_norm - (segmented_box.x_center_norm, segmented_box.y_center_norm)

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


def apply_image_filters(bgr_image, prob_rand_bright=1.0, bright_shift_range=(-5, 5)):
    """
    apply image filters to image
    TODO(minhnh) add color and contrast shifts
    """
    if np.random.uniform() < prob_rand_bright:
        # randomly change brightness at a certain probability
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        rand_bright_ship = np.random.randint(*bright_shift_range)
        h, s, v = cv2.split(hsv)
        lim = 255 - rand_bright_ship
        v[v > lim] = 255
        np.add(v[v <= lim], rand_bright_ship, out=v[v <= lim], casting="unsafe")
        final_hsv = cv2.merge((h, s, v))
        bgr_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return bgr_image


class SegmentedObject(object):
    """"contains information processed from a single image-mask pair"""
    max_dimension = None
    bgr_image = None
    mask_image = None
    segmented_box = None
    segmented_x_coords = None
    segmented_y_coords = None

    def __init__(self, image_path, mask_path):
        # read color and mask images
        self.bgr_image = cv2.imread(image_path)
        self.mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        self.segmented_y_coords, self.segmented_x_coords = np.where(self.mask_image)

        # calculate maximum pixel dimension from the segmented coordinates
        self.segmented_box = SegmentedBox(self.segmented_x_coords, self.segmented_y_coords, self.mask_image.shape)

    def view_segmented_color_img(self):
        segmented_bgr = cv2.bitwise_and(self.bgr_image, self.bgr_image, mask=self.mask_image)
        display_image_and_wait(segmented_bgr, 'segmented_object')


class SegmentedObjectCollection(object):
    """Collection of instances of 'SegmentedObject''s, one for each image-mask pair"""
    class_id = None
    _segmented_objects = None

    def __init__(self, class_id, obj_img_dir, obj_mask_dir):
        self.class_id = class_id

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
                continue

    def project_segmentation_on_background(self, background_image):
        # choose random segmentation from collection
        rand_segmentation = np.random.choice(self._segmented_objects)

        # create a random transformation
        bg_height, bg_width = background_image.shape[:2]
        transformed_coords_norm = apply_random_transformation((bg_height, bg_width), rand_segmentation.segmented_box)

        # denormalize transformed coordinates, preserving aspect ratio, but making sure the object is within
        # background image
        denorm_factor = min(bg_height, bg_width)
        transformed_coords = np.array(transformed_coords_norm * denorm_factor, dtype=int)

        # create and clean a new mask for the projected pixels
        morph_kernel_size = 2
        morph_iter_num = 1
        projected_obj_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
        projected_obj_mask[transformed_coords[:, 1], transformed_coords[:, 0]] = 255
        projected_obj_mask = cleanup_mask(projected_obj_mask, morph_kernel_size, morph_iter_num)

        # denoise projected RGB values
        projected_bgr = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        projected_bgr[transformed_coords[:, 1], transformed_coords[:, 0], :] = \
            rand_segmentation.bgr_image[rand_segmentation.segmented_y_coords,
                                        rand_segmentation.segmented_x_coords]
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        projected_bgr = cv2.morphologyEx(projected_bgr, cv2.MORPH_CLOSE, kernel, iterations=morph_iter_num)
        projected_bgr = apply_image_filters(projected_bgr)

        # write to background image
        cleaned_y_coords, clean_x_coords = np.where(projected_obj_mask)
        background_image[cleaned_y_coords, clean_x_coords] = projected_bgr[cleaned_y_coords, clean_x_coords]
        new_box = SegmentedBox(clean_x_coords, cleaned_y_coords, (bg_height, bg_width), class_id=self.class_id)
        return background_image, new_box


class ImageAugmenter(object):
    class_dict = None
    _data_dir = None
    _greenbox_image_dir = None
    _mask_dir = None
    _augment_backgrounds = None
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
            self.class_dict = yaml.load(infile, Loader=yaml.FullLoader)

        # load segmented objects
        TerminalColors.formatted_print("Loading object masks for '{}' classes".format(len(self.class_dict)),
                                       TerminalColors.OKBLUE)
        self._segmented_object_collections = {}
        for cls_id, cls_name in self.class_dict.items():
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

    def _sample_classes(self, max_obj_num_per_bg):
        # TODO(minhnh) ensure balance sampling
        num_obj = np.random.randint(1, max_obj_num_per_bg + 1)
        sampled_class_ids = np.random.choice(list(self._segmented_object_collections.keys()), num_obj)
        return [self._segmented_object_collections[cls_id] for cls_id in sampled_class_ids]

    def generate_single_image(self, background_image, max_obj_num_per_bg):
        """generate a single image and its bounding box annotations"""
        sampled_collections = self._sample_classes(max_obj_num_per_bg)
        annotations = []
        bg_img_copy = background_image.copy()
        for obj_collection in sampled_collections:
            bg_img_copy, box = obj_collection.project_segmentation_on_background(bg_img_copy)
            generated_ann = box.to_dict()
            annotations.append(generated_ann)
        return bg_img_copy, annotations

    def generate_detection_data(self, split_name, output_dir, output_annotation_dir, num_image_per_bg,
                                max_obj_num_per_bg, display_boxes=False, write_chunk_ratio=0.05):
        """
        The main function which generate
        - generate synthetic images under <outpu_dir>/<split_name>
        - generate
        """
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
        if max_obj_num_per_bg <= 0 or max_obj_num_per_bg > len(self.class_dict):
            max_obj_num_per_bg = len(self.class_dict)

        # generate images and annotations
        img_cnt = 0
        total_img_cnt = num_image_per_bg * len(self._augment_backgrounds)
        zero_pad_num = len(str(total_img_cnt))
        annotations = []
        for bg_img in self._augment_backgrounds:
            for _ in range(num_image_per_bg):
                if print_progress(img_cnt + 1, total_img_cnt, prefix="creating image ", fraction=write_chunk_ratio):
                    # periodically dump annotations
                    with open(annotation_path, 'a') as infile:
                        yaml.dump(annotations, infile, default_flow_style=False)
                        annotations = []

                # generate new image
                generated_image, box_annotations = self.generate_single_image(bg_img, max_obj_num_per_bg)
                if display_boxes:
                    drawn_img = draw_labeled_boxes(generated_image, box_annotations, self.class_dict)
                    display_image_and_wait(drawn_img, 'box image')

                # write image and annotations
                img_file_name = '{}_{}.jpg'.format(split_name, str(img_cnt).zfill(zero_pad_num))
                img_file_path = os.path.join(split_output_dir, img_file_name)
                annotations.append({'image_name': img_file_path, 'objects': box_annotations})
                cv2.imwrite(img_file_path, generated_image)
                img_cnt += 1
