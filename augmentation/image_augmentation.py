import copy
import os
import yaml
import numpy as np
import cv2
from skimage.transform import SimilarityTransform, matrix_transform
from skimage.util import random_noise
from dataset_interface.common import SegmentedBox
from dataset_interface.utils import TerminalColors, glob_extensions_in_directory, ALLOWED_IMAGE_EXTENSIONS, \
                                    display_image_and_wait, prompt_for_yes_or_no, print_progress, cleanup_mask, \
                                    draw_labeled_boxes, split_path
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


def apply_image_filters(bgr_image, prob_rand_color=0.2, prob_rand_noise=0.2,
                        prob_rand_bright=0.2, bright_shift_range=(-5, 5)):
    """
    apply image filters to image
    TODO(minhnh) add contrast shifts
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

    # randomly select from the allowed color maps below to apply to the image
    # see more at: https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
    ALLOWED_COLOR_MAPS = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_HOT, cv2.COLORMAP_OCEAN,
                          cv2.COLORMAP_PARULA, cv2.COLORMAP_PINK, cv2.COLORMAP_SUMMER, cv2.COLORMAP_WINTER]
    if np.random.uniform() < prob_rand_color:
        bgr_image = cv2.applyColorMap(bgr_image, np.random.choice(ALLOWED_COLOR_MAPS))

    # randomly add gaussian noise, since random_noise normalize the image, we need to convert it back to pixel value
    if np.random.uniform() < prob_rand_noise:
        noise_img = random_noise(bgr_image, mode='gaussian')
        bgr_image = (255 * noise_img).astype(np.uint8)

    # randomly add salt & pepper noise, convert it back to pixel value
    if np.random.uniform() < prob_rand_noise:
        noise_img = random_noise(bgr_image, mode='s&p')
        bgr_image = (255 * noise_img).astype(np.uint8)

    return bgr_image


class SegmentedObject(object):
    """"contains information processed from a single image-mask pair"""
    max_dimension = None
    bgr_image = None
    mask_image = None
    class_id = None
    segmented_box = None
    segmented_x_coords = None
    segmented_y_coords = None

    def __init__(self, image_path, mask_path, class_id, invert_mask):
        # read color and mask images
        self.bgr_image = cv2.imread(image_path)
        self.mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if invert_mask:
            self.mask_image = cv2.bitwise_not(self.mask_image)
        self.class_id = class_id
        self.segmented_y_coords, self.segmented_x_coords = np.where(self.mask_image)

        # calculate maximum pixel dimension from the segmented coordinates
        self.segmented_box = SegmentedBox(self.segmented_x_coords, self.segmented_y_coords, self.mask_image.shape)

    def view_segmented_color_img(self):
        segmented_bgr = cv2.bitwise_and(self.bgr_image, self.bgr_image, mask=self.mask_image)
        display_image_and_wait(segmented_bgr, 'segmented_object')




class ImageAugmenter(object):
    _class_dict = None
    _background_paths = None
    _images_paths = None
    _masks_paths = None
    _object_collections = dict()
    _num_objects_per_class = None

    def __init__(self, data_dir, background_dir, class_annotation_file, num_objects_per_class):
        # check required directories
        TerminalColors.formatted_print('Data directory: ' + data_dir, TerminalColors.OKBLUE)
        if not os.path.isdir(data_dir):
            raise RuntimeError("'{}' is not an existing directory".format(data_dir))

        TerminalColors.formatted_print('Background directory: ' + background_dir, TerminalColors.OKBLUE)
        if not os.path.isdir(background_dir):
            raise RuntimeError("'{}' is not an existing directory".format(background_dir))

        # load backgrounds for image augmentation
        self._background_paths = glob_extensions_in_directory(background_dir, ALLOWED_IMAGE_EXTENSIONS)
        TerminalColors.formatted_print('Found {} background images '.format(len(self._background_paths)),
                                        TerminalColors.OKBLUE)
        # self._augment_backgrounds = []
        # for bg_path in background_paths:
        #     bg_img = cv2.imread(bg_path)
        #     self._augment_backgrounds.append(bg_img)

        # Saving number of objects per class
        self._num_objects_per_class = num_objects_per_class

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
        
        for cls_id, (cls_name, _) in self._class_dict.items():
            obj_dir = os.path.join(data_dir, cls_name)
            obj_img_dir = os.path.join(obj_dir, 'images')
            obj_mask_dir = os.path.join(obj_dir, 'masks')

            try:
                TerminalColors.formatted_print("loading images and masks for class '{}'".format(cls_name),
                                               TerminalColors.BOLD)
                # segmented_obj = SegmentedObjectCollection(cls_id, obj_img_dir, obj_mask_dir)
                # self._segmented_object_collections[cls_id] = segmented_obj
                self._load_objects_per_class(cls_id, obj_img_dir, obj_mask_dir)
            except RuntimeError as e:
                TerminalColors.formatted_print("skipping class '{}': {}".format(cls_name, e), TerminalColors.WARNING)
                continue

    def project_segmentation_on_background(self, background_image, segmented_obj_data, augmented_mask):
        # create a random transformation
        bg_height, bg_width = background_image.shape[:2]
        transformed_coords_norm = apply_random_transformation((bg_height, bg_width), segmented_obj_data.segmented_box)

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
            segmented_obj_data.bgr_image[segmented_obj_data.segmented_y_coords,
                                        segmented_obj_data.segmented_x_coords]
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        projected_bgr = cv2.morphologyEx(projected_bgr, cv2.MORPH_CLOSE, kernel, iterations=morph_iter_num)
        projected_bgr = apply_image_filters(projected_bgr, prob_rand_color=0.2)

        # write to background image
        cleaned_y_coords, clean_x_coords = np.where(projected_obj_mask)
        background_image[cleaned_y_coords, clean_x_coords] = projected_bgr[cleaned_y_coords, clean_x_coords]

        # Add object mask 
        augmented_mask[cleaned_y_coords, clean_x_coords] = self._class_dict[segmented_obj_data.class_id][1][::-1]
        # display_image_and_wait(augmented_mask, 'object onto the background') # NOTE: remove
        new_box = SegmentedBox(clean_x_coords, cleaned_y_coords, (bg_height, bg_width), class_id=segmented_obj_data.class_id)
        return background_image, new_box


    def _load_objects_per_class(self, class_id, obj_img_dir, obj_mask_dir):
        # check directories
        if not os.path.isdir(obj_img_dir):
            raise RuntimeError("image folder '{}' is not an existing directory".format(obj_img_dir))
        if not os.path.isdir(obj_mask_dir):
            raise RuntimeError("mask folder '{}' is not an existing directory".format(obj_mask_dir))

        # glob object images
        obj_img_paths = glob_extensions_in_directory(obj_img_dir, ALLOWED_IMAGE_EXTENSIONS)
        if not obj_img_paths:
            raise RuntimeError("found no image of supported types in '{}'".format(obj_img_dir))

        # glob mask images
        obj_mask_paths = glob_extensions_in_directory(obj_mask_dir, ALLOWED_IMAGE_EXTENSIONS)
        if not obj_img_paths:
            raise RuntimeError("found no mask of supported types in '{}'".format(obj_img_dir))

        # load images and masks
        print("found '{}' object images in '{}'".format(len(obj_img_paths), obj_img_dir))
        print("found '{}' object masks in '{}'".format(len(obj_mask_paths), obj_mask_dir))

        img_indices = list(range(len(obj_img_paths)))
        np.random.shuffle(img_indices)

        img_count = 0
        current_img_idx = 0
        img_paths = []
        mask_paths = []
        while img_count < self._num_objects_per_class and current_img_idx < len(img_indices):
            try:
                # we check if the current image has a corresponding mask
                img_path = obj_img_paths[img_indices[current_img_idx]]
                mask_path = get_image_mask_path(img_path, obj_mask_paths)

                # even if both the image and mask exist, we still need to load them
                # in order to check that they are valid images
                cv2.imread(img_path)
                img_paths.append(img_path)
                mask_paths.append(mask_path)
                img_count += 1
            except Exception as exc:
                TerminalColors.formatted_print("[load_objects_per_class] Error: {0}. Skipping image".format(exc), TerminalColors.WARNING)
            current_img_idx += 1

        print("{} images saved per class {}".format(len(img_paths), class_id))
        self._object_collections[class_id] = {'images': img_paths, 'masks' : mask_paths}

    def _sample_classes(self, max_obj_num_per_bg, invert_mask=False):
        sample_count = 0
        sampled_objects = []
        while self._object_collections and sample_count < max_obj_num_per_bg:
            # we sample an object class and then an image from that class
            sampled_class_id = np.random.choice(list(self._object_collections.keys()))
            sampled_object_id = np.random.choice(list(range(len(self._object_collections[sampled_class_id]['images']))))

            # we remove the image and mask paths from the image path dictionary
            image_path = self._object_collections[sampled_class_id]['images'].pop(sampled_object_id)
            mask_path = self._object_collections[sampled_class_id]['masks'].pop(sampled_object_id)
            # if there are no more images from the sampled class, we remove the
            # class entry from the image path dictionary
            if not self._object_collections[sampled_class_id]['images']:
                del self._object_collections[sampled_class_id]

            sampled_objects.append(SegmentedObject(image_path, mask_path, sampled_class_id, invert_mask))
            sample_count += 1
        return sampled_objects

    def generate_single_image(self, background_image, max_obj_num_per_bg, invert_mask=False):
        """generate a single image and its bounding box annotations"""
        sampled_objects = self._sample_classes(max_obj_num_per_bg, invert_mask)
        bg_img_copy = background_image.copy()
        bg_img_copy = apply_image_filters(bg_img_copy)

        augmented_mask = bg_img_copy.copy()
        augmented_mask[:] = (255,255,255)
        
        annotations = []
        for obj in sampled_objects:
            bg_img_copy, box = self.project_segmentation_on_background(bg_img_copy, obj,augmented_mask)
            generated_ann = box.to_dict()
            annotations.append(generated_ann)
        return bg_img_copy, annotations, augmented_mask

    def generate_detection_data(self, split_name, output_dir_images, output_dir_masks, output_annotation_dir, max_obj_num_per_bg,
        num_images_per_bg=10, display_boxes=False, write_chunk_ratio=0.05, invert_mask=False):
        """
        The main function which generate
        - generate synthetic images under <outpu_dir>/<split_name>
        - generate
        """
        split_output_dir_images = os.path.join(output_dir_images, split_name)
        split_output_dir_masks = os.path.join(output_dir_masks, split_name)
        TerminalColors.formatted_print("generating images for split '{}' under '{}'"
                                       .format(split_name, split_output_dir_images), TerminalColors.BOLD)
        TerminalColors.formatted_print("generating masks for split '{}' under '{}'"
                                       .format(split_name, split_output_dir_masks), TerminalColors.BOLD)
        # check output image directory
        if not os.path.isdir(split_output_dir_images):
            print("creating directory: " + split_output_dir_images)
            os.mkdir(split_output_dir_images)
        elif os.listdir(split_output_dir_images):
            if not prompt_for_yes_or_no("directory '{}' not empty. Overwrite?".format(split_output_dir_images)):
                raise RuntimeError("not overwriting '{}'".format(split_output_dir_images))

        if not os.path.isdir(split_output_dir_masks):
            print("creating directory: " + split_output_dir_masks)
            os.mkdir(split_output_dir_masks)
        elif os.listdir(split_output_dir_masks):
            if not prompt_for_yes_or_no("directory '{}' not empty. Overwrite?".format(split_output_dir_masks)):
                raise RuntimeError("not overwriting '{}'".format(split_output_dir_masks))

        # check output annotation file
        annotation_path = os.path.join(output_annotation_dir, split_name + '.yml')
        TerminalColors.formatted_print("generating annotations for split '{}' in '{}'"
                                       .format(split_name, annotation_path), TerminalColors.BOLD)
        if os.path.isfile(annotation_path):
            if not prompt_for_yes_or_no("file '{}' exists. Overwrite?".format(annotation_path)):
                raise RuntimeError("not overwriting '{}'".format(annotation_path))

        # # store a reasonable value for the maximum number of objects projected onto each background
        # if max_obj_num_per_bg <= 0 or max_obj_num_per_bg > len(self.class_dict):
        #     max_obj_num_per_bg = len(self.class_dict)

        # generate images and annotations
        img_cnt = 0

        # Total number of images = classes * objects per background * number of backgrounds
        total_img_cnt = len(self._background_paths) * num_images_per_bg
        zero_pad_num = len(str(total_img_cnt))
        annotations = {}
        for bg_path in self._background_paths:
            # generate new image
            try:
                bg_img = cv2.imread(bg_path)
            except RuntimeError as e:
                TerminalColors.formatted_print("Ignoring background {} because {}".format(bg_path, e), TerminalColors.WARNING)
                continue
            # we store the current object path dictionary since we will sample images without replacement
            img_path_dictionary = copy.deepcopy(self._object_collections)

            for _ in range(num_images_per_bg):
                generated_image, box_annotations, augmented_mask = self.generate_single_image(bg_img, max_obj_num_per_bg, invert_mask)
                if display_boxes:
                    drawn_img = draw_labeled_boxes(generated_image, box_annotations, self.class_dict)
                    display_image_and_wait(drawn_img, 'box image')

                # write image and annotations
                img_file_name = '{}_{}.jpg'.format(split_name, str(img_cnt).zfill(zero_pad_num))
                img_file_path = os.path.join(split_output_dir_images, img_file_name)
                
                mask_file_name = '{}_{}.png'.format(split_name, str(img_cnt).zfill(zero_pad_num))
                mask_file_path = os.path.join(split_output_dir_masks, mask_file_name)

                # Cast box_annotations_class_id 
                for box in box_annotations:
                    box['class_id'] = int(box['class_id'])
                annotations[img_file_name] =  box_annotations
                cv2.imwrite(img_file_path, generated_image)
                cv2.imwrite(mask_file_path, augmented_mask)
                img_cnt += 1

                # Writing annotations
                if print_progress(img_cnt + 1, total_img_cnt, prefix="creating image ", fraction=write_chunk_ratio):
                    # periodically dump annotations
                    with open(annotation_path, 'a') as infile:
                        yaml.dump(annotations, infile, default_flow_style=False)
                        annotations = {}

                # we restore the object path dictionary after the image augmentation with the current background
                if not self._object_collections:
                    self._object_collections = copy.deepcopy(img_path_dictionary)
        with open(annotation_path, 'a') as infile:
            yaml.dump(annotations, infile, default_flow_style=False)
