import copy
import os
import yaml
from collections import OrderedDict
import time
import multiprocessing as mp
from tqdm import tqdm
import xmltodict
import numpy as np
import cv2
from skimage.transform import SimilarityTransform, matrix_transform
from skimage.util import random_noise

from dataset_interface.common import SegmentedBox
from dataset_interface.utils import TerminalColors, glob_extensions_in_directory, ALLOWED_IMAGE_EXTENSIONS, \
                                    display_image_and_wait, prompt_for_yes_or_no, print_progress, cleanup_mask, \
                                    draw_labeled_boxes, split_path
from dataset_interface.augmentation.background_segmentation import get_image_mask_path

def apply_random_transformation(background_size, segmented_box, margin=0.03, max_obj_size_in_bg=0.4, prob_rand_transformation=1.0):
    """apply a random transformation to 2D coordinates nomalized to image size"""
    # translate object coordinates to the object center's frame, i.e. whitens
    whitened_coords_norm = segmented_box.segmented_coords_norm - (segmented_box.x_center_norm, segmented_box.y_center_norm)

    # then generate a random rotation around the z-axis (perpendicular to the image plane), and limit the object scale
    # to maximum (default) 50% of the background image, i.e. the normalized largest dimension of the object must be at
    # most 0.5. To put it simply, scale the objects down if they're too big.
    # TODO(minhnh) add shear
    max_scale = 1.75
    if segmented_box.max_dimension_norm > max_obj_size_in_bg:
        max_scale = max_obj_size_in_bg / segmented_box.max_dimension_norm
    random_rot_angle = np.random.uniform(0, np.pi)
    rand_scale = np.random.uniform(0.5, max_scale)

    # generate a random translation within the image boundaries for whitened, normalized coordinates, taking into
    # account the maximum allowed object dimension. After this translation, the normalized coordinates should
    # stay within [margin, 1-margin] for each dimension
    scaled_max_dimension = segmented_box.max_dimension_norm * max_scale
    low_norm_bound, high_norm_bound = ((scaled_max_dimension / 2) + margin, 1 - margin - (scaled_max_dimension / 2))
    random_translation_x = np.random.uniform(low_norm_bound, high_norm_bound) * background_size[1]
    random_translation_y = np.random.uniform(low_norm_bound, high_norm_bound) * background_size[0]

    # create the transformation matrix for the generated rotation, translation and scale
    if np.random.uniform() > prob_rand_transformation:
        tf_matrix = SimilarityTransform(rotation=random_rot_angle, scale=min(background_size),
                                        translation=(random_translation_x, random_translation_y)).params
    else:
        tf_matrix = SimilarityTransform(scale=rand_scale * min(background_size),
                                        translation=(random_translation_x, random_translation_y)).params

    # apply transformation
    transformed_coords = matrix_transform(whitened_coords_norm, tf_matrix)
    return transformed_coords


def apply_image_filters(bgr_image, prob_rand_color=0.2, prob_rand_noise=0.01,
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

def get_voc_annotation_dict(image_name, image_dir, annotation_data, class_dict):
    '''Returns a dictionary representing an image annotation in VOC format.
    The "folder" and "size" tags are not included in the annotation.

    Keyword arguments:
    image_name: str -- name of the image for which an annotation is generated
    image_dir: str -- directory in which the image is saved
    annotation_data: Dict[Sequence[str, int]]: list of object annotation in the image;
                                               the annotation for each object is expected
                                               to contain the following keys:
                                               {class_id, xmin, xmax, ymin, ymax}
    class_dict: Dict[int, str] -- a dictionary mapping class IDs to class labels

    '''
    annotation_dict = OrderedDict()
    annotation_dict['filename'] = image_name
    annotation_dict['path'] = os.path.join(image_dir, image_name)
    annotation_dict['source'] = {'database': 'unknown'}
    annotation_dict['segmented'] = 0
    annotation_dict['object'] = []
    for object_data in annotation_data:
        object_annotation = {'name': class_dict[object_data['class_id']],
                             'pose': 'Unspecified',
                             'truncated': 0,
                             'difficult': 0,
                             'bndbox': {'xmin': object_data['xmin'],
                                        'ymin': object_data['ymin'],
                                        'xmax': object_data['xmax'],
                                        'ymax': object_data['ymax']
                                       }
                            }
        annotation_dict['object'].append(object_annotation)
    return {'annotation': annotation_dict}

class AnnotationFormats(object):
    CUSTOM = 'custom'
    VOC = 'voc'

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
    _num_objects_per_class = None

    # dictionary in which each key is an object class, while
    # each value is a dictionary containing two keys:
    # * images: absolute paths to images from the class
    # * masks: absolute paths to segmentation masks for
    #          the images from the class (i.e. masks[i] is the
    #          segmentation mask of images[i])
    _object_collections = dict()

    # we will sample images without replacement - this is done by removing sampled
    # images from self._object_collections - so we keep a copy of the object path
    # dictionary so that we can restore it later
    _object_collections_copy = dict()

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

        self._object_collections_copy = copy.deepcopy(self._object_collections)

    def project_segmentation_on_background(self, background_image, segmented_obj_data, augmented_mask, prob_rand_transformation=1.0):
        # create a random transformation
        bg_height, bg_width = background_image.shape[:2]
        transformed_coords = apply_random_transformation((bg_height, bg_width), segmented_obj_data.segmented_box,
                                                          prob_rand_transformation=prob_rand_transformation)
        transformed_coords = transformed_coords.astype(np.int)

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
        projected_bgr = apply_image_filters(projected_bgr, prob_rand_color=0.5)

        # write to background image
        cleaned_y_coords, clean_x_coords = np.where(projected_obj_mask)
        background_image[cleaned_y_coords, clean_x_coords] = projected_bgr[cleaned_y_coords, clean_x_coords]

        # Add object mask
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

    def generate_single_image(self, background_image, max_obj_num_per_bg, invert_mask=False, prob_rand_trans=1.0):
        """generate a single image and its bounding box annotations"""
        sampled_objects = self._sample_classes(max_obj_num_per_bg, invert_mask)
        bg_img_copy = background_image.copy()
        # bg_img_copy = apply_image_filters(bg_img_copy)

        augmented_mask = bg_img_copy.copy()
        augmented_mask[:] = (255,255,255)

        annotations = []
        for obj in sampled_objects:
            bg_img_copy, box = self.project_segmentation_on_background(bg_img_copy,
                                                                       obj,
                                                                       augmented_mask,
                                                                       prob_rand_transformation=prob_rand_trans)
            generated_ann = box.to_dict()
            annotations.append(generated_ann)
            time.sleep(0.1)
        return bg_img_copy, annotations, augmented_mask

    def create_image(self, params):
        bg_img, max_obj_num_per_bg, invert_mask, split_name, zero_pad_num, \
                              split_output_dir_images, split_output_dir_masks, prob_rand_trans, seed = params

        np.random.seed(seed + int(time.time()))

        # we restore the object path dictionary if there are no more objects to be sampled at this point
        if not self._object_collections:
            self._object_collections = copy.deepcopy(self._object_collections_copy)

        generated_image, box_annotations, augmented_mask = self.generate_single_image(bg_img,
                                                                                      max_obj_num_per_bg,
                                                                                      invert_mask,
                                                                                      prob_rand_trans)

        # write image and annotations
        with lock:
            img_file_name = '{}_{}.jpg'.format(split_name, str(img_cnt.value).zfill(zero_pad_num))
            img_file_path = os.path.join(split_output_dir_images, img_file_name)

            mask_file_name = '{}_{}.png'.format(split_name, str(img_cnt.value).zfill(zero_pad_num))
            mask_file_path = os.path.join(split_output_dir_masks, mask_file_name)
            cv2.imwrite(img_file_path, generated_image)
            cv2.imwrite(mask_file_path, augmented_mask)

            img_cnt.value += 1

        # Cast box_annotations_class_id
        for box in box_annotations:
            box['class_id'] = int(box['class_id'])
        # annotations[img_file_name] =  box_annotations

        return (img_file_name, box_annotations)

    def setup(self, t, l):
        global img_cnt, lock
        img_cnt = t
        lock = l

    def save_annotations(self, annotations, split_output_image_dir,
                         output_annotation_dir, split_name,
                         annotation_file_path, annotation_format):
        with open(annotation_file_path, 'a') as infile:
            yaml.dump(annotations, infile, default_flow_style=False)

        if annotation_format == AnnotationFormats.VOC:
            annotation_dir = os.path.join(output_annotation_dir, split_name)
            for img_file_name, annotation_data in annotations.items():
                annotation_dict = get_voc_annotation_dict(img_file_name,
                                                          split_output_image_dir,
                                                          annotation_data,
                                                          self._class_dict)
                img_name = img_file_name.split('.')[0]
                img_annotation_file_path = os.path.join(annotation_dir, img_name + '.xml')
                with open(img_annotation_file_path, 'w') as annotation_file:
                    xmltodict.unparse(annotation_dict, output=annotation_file, pretty=True)

    def generate_detection_data(self, split_name, output_dir_images, output_dir_masks,
                                output_annotation_dir, max_obj_num_per_bg,
                                num_images_per_bg=10, write_chunk_ratio=0.05,
                                invert_mask=False, prob_rand_trans=1.0,
                                annotation_format=AnnotationFormats.CUSTOM):
        """Generates:
        * synthetic images under <output_dir>/synthetic_images/<split_name>
        * image annotations under <output_dir>/annotations

        If annotation_format is equal to AnnotationFormats.VOC, an annotation
        file is generated for each image under <output_dir>/annotations/split_name,
        such that the name of the annotation is the same as the name of the
        image file (e.g. if the image name is train_01.jpg, the name of the
        annotation file will be train_01.xml).
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
        annotation_file_path = os.path.join(output_annotation_dir, split_name + '.yaml')
        TerminalColors.formatted_print("generating annotations for split '{}' in '{}'"
                                       .format(split_name, annotation_file_path), TerminalColors.BOLD)
        if os.path.isfile(annotation_file_path):
            if not prompt_for_yes_or_no("file '{}' exists. Overwrite?".format(annotation_file_path)):
                raise RuntimeError("not overwriting '{}'".format(annotation_file_path))

        if annotation_format == AnnotationFormats.VOC:
            annotation_dir = os.path.join(output_annotation_dir, split_name)
            if not os.path.isdir(annotation_dir):
                os.mkdir(annotation_dir)
            else:
                if not prompt_for_yes_or_no("directory '{}' exists. Overwrite?".format(annotation_dir)):
                    raise RuntimeError("not overwriting '{}'".format(annotation_dir))

        # Total number of images = classes * objects per background * number of backgrounds
        total_img_cnt = len(self._background_paths) * num_images_per_bg
        zero_pad_num = len(str(total_img_cnt))
        annotations = {}

        # Prepare multiprocessing
        img_cnt = mp.Value('i', 0)
        lock = mp.Lock()
        pool = mp.Pool(initializer=self.setup, initargs=[img_cnt, lock])

        for bg_idx in tqdm(range(len(self._background_paths))):
            bg_path = self._background_paths[bg_idx]
        # for bg_path in self._background_paths:
            # generate new image
            try:
                bg_img = cv2.imread(bg_path)
            except RuntimeError as e:
                TerminalColors.formatted_print("Ignoring background {} because {}".format(bg_path, e), TerminalColors.WARNING)
                continue

            bg_img_params = [(bg_img, max_obj_num_per_bg, invert_mask, split_name, zero_pad_num, \
                              split_output_dir_images, split_output_dir_masks, prob_rand_trans, seed ) \
                                  for seed in range(num_images_per_bg)]

            annotations_per_bg = pool.map(self.create_image, bg_img_params)

            for img_file_name, box_annotations in annotations_per_bg:
                annotations[img_file_name] = box_annotations

            # Writing annotations
            if print_progress(img_cnt.value + 1, total_img_cnt,
                              prefix="creating image ", fraction=write_chunk_ratio):
                # periodically dump annotations
                self.save_annotations(annotations, split_output_dir_images,
                                      output_annotation_dir, split_name,
                                      annotation_file_path, annotation_format)
                annotations = {}

        self.save_annotations(annotations, split_output_dir_images,
                              output_annotation_dir, split_name,
                              annotation_file_path, annotation_format)
