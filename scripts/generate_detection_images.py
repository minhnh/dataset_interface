#!/usr/bin/env python3
import argparse
from dataset_interface.utils import TerminalColors

import os
import numpy as np
from imageio import imread, imwrite
import yaml
import cv2
import glob


class Vec2D(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class BoundingBox(object):
    def __init__(self):
        self.nonzero_rows = None
        self.nonzero_cols = None
        self.min_coords = Vec2D()
        self.max_coords = Vec2D()


def get_bb_from_mask(segmentation_mask: np.array) -> BoundingBox:
    '''Returns a BoundingBox object with information extracted
    from the given segmentation mask.

    Keyword arguments:
    segmentation_mask: np.array -- a 2D numpy array representing a grayscale image
                                   with a single object in it, where the assumption
                                   is that non-zero pixels represent the object
    '''
    bb = BoundingBox()
    bb.nonzero_rows, bb.nonzero_cols = np.where(segmentation_mask)
    bb.min_coords.x, bb.max_coords.x = (np.min(bb.nonzero_cols), np.max(bb.nonzero_cols))
    bb.min_coords.y, bb.max_coords.y = (np.min(bb.nonzero_rows), np.max(bb.nonzero_rows))
    return bb


def get_bb(coords):
    bb = BoundingBox()
    bb.min_coords.x, bb.max_coords.x = (np.min(coords[0]), np.max(coords[0]))
    bb.min_coords.y, bb.max_coords.y = (np.min(coords[1]), np.max(coords[1]))
    return bb


def generate_transformation(bb: BoundingBox, boundaries: tuple) -> np.array:
    '''Generates a homogeneous transformation matrix of type int that translates,
    rotates, and scales the given bounding box, ensuring that the
    transformed points are within the given boundaries.

    Keyword arguments:
    bb: BoundingBox -- a BoundingBox object
    boundaries: tuple -- coordinate boundaries; assumed to represent
                         the (row, column) sizes of an image

    '''
    use_transformation = False
    t = None
    while not use_transformation:
        use_transformation = True
        rectangle_points = np.array([[bb.min_coords.x, bb.min_coords.x, bb.max_coords.x, bb.max_coords.x],
                                     [bb.min_coords.y, bb.max_coords.y, bb.min_coords.y, bb.max_coords.y]])
        rectangle_points = np.vstack((rectangle_points, [1., 1., 1., 1.]))

        # we generate a random rotation angle
        random_rot_angle = np.random.uniform(0, 2*np.pi)
        random_rot_matrix = np.array([[np.cos(random_rot_angle), -np.sin(random_rot_angle)],
                                      [np.sin(random_rot_angle), np.cos(random_rot_angle)]])

        # we generate a random translation within the image boundaries
        random_translation_x = np.random.uniform(-bb.min_coords.x, boundaries[1]-bb.max_coords.x)
        random_translation_y = np.random.uniform(-bb.min_coords.y, boundaries[0]-bb.max_coords.y)
        translation_vector = np.array([[random_translation_x], [random_translation_y]])

        # we generate a random scaling factor between 0.5 and 1.5
        # of the original object size
        # random_scaling_factor = np.random.uniform(1, 1.5)
        random_scaling_factor = 1
        s = np.array([[random_scaling_factor, 0., 0.],
                      [0., random_scaling_factor, 0.],
                      [0., 0., 1.]])

        t = np.hstack((random_rot_matrix, translation_vector))
        t = np.vstack((t, np.array([0., 0., 1.])))
        t = t.dot(s)

        transformed_bb = t.dot(rectangle_points)
        transformed_bb = np.array(transformed_bb, dtype=int)
        for point in transformed_bb.T:
            if point[0] < 0 or point[0] >= boundaries[1] or \
               point[1] < 0 or point[1] >= boundaries[0]:
                # print('Invalid transformation')
                use_transformation = False
                break
    return t


def augment_data(img_dir_name: str,
                 background_img_dir: str,
                 images_per_background: int,
                 annotations_file: str,
                 output_dir: str,
                 classes_to_id: dict) -> None:
    '''Given the images in "img_dir_name", each of which is assumed to have a
    single object, generates a new set of images in which the objects are put
    on the backgrounds in "background_img_dir" and are transformed (translated,
    rotated, scaled) in random fashion. For each background and image combination,
    "images_per_background" images are generated.

    Args:
    * img_dir_name: str -- path to a directory with image files and object
                         segmentation masks (img_dir_name is expected to have
                         a directory "object_masks" with the segmentation masks,
                         such that if an image is called "test.jpg", its segmentation
                         mask will have the name "test_mask.jpg")
    * background_img_dir: str -- path to a directory with background images for augmentation
    * images_per_background: int -- number of images to generate per given background
    * annotations_file: str -- yaml file for the annotations of the training images
    * output_dir:str -- directory to store the genenated images
    '''

    max_objects_per_image = 6

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    if 'train' in annotations_file:
        if not os.path.isdir('training_images'):
            os.mkdir('training_images')
    elif 'val' in annotations_file:
        if not os.path.isdir('validation_images'):
            os.mkdir('validation_images')

    backgrounds = os.listdir(background_img_dir)
    print('Number of backgrounds images ', len(backgrounds))

    perspectives = None

    if 'perspectives' in img_dir_name:
        perspectives = os.listdir(img_dir_name)
        print('Number of perspectives ', len(perspectives))

    # images = os.listdir(os.path.join(img_dir_name,objects[0]))

    # Generating images paths
    background_paths = []
    for background in backgrounds:
        background_path = os.path.join(background_img_dir, background)
        background_paths.append(background_path)

    objects_paths = []
    if perspectives is not None:
        for perspective in perspectives:
            objects = os.listdir(os.path.join(img_dir_name, perspective))
            for object in objects:
                object_path = os.path.join(img_dir_name, perspective, object)
                objects_paths.append(object_path)
    else:
        objects = os.listdir(img_dir_name)
        for object in objects:
            object_path = os.path.join(img_dir_name, perspective, object)
            objects_paths.append(object_path)

    augmented_img_counter = 0

    training_images = []

    for background_path in background_paths:
        background_img_original = np.array(imread(background_path), dtype=np.uint8)
        # print(background_path)
        for _ in range(images_per_background):  # Number of images generated using that backgrounds
            background_img = np.array(background_img_original, dtype=np.uint8)
            augmented_objects = []
            for _ in range(np.random.randint(1, max_objects_per_image)):  # Number of objects in the image
                object_path = objects_paths[np.random.randint(len(objects_paths))]

                # Collect object class
                if perspectives is not None:
                    object_class = object_path.split('/')[2]
                else:
                    object_class = object_path.split('/')[1]

                # List of images in the directory
                images = glob.glob(object_path+'/*.jpg')

                if len(images) == 0:
                    # print('Inspecting subdirectories...')
                    # print(object_path)
                    images = glob.glob(object_path+'/**/*.jpg')

                if len(images) == 0:
                    print('Images not found')
                    continue

                # print('Len of images ', len(images))
                image_full_name = images[np.random.randint(0, len(images))]
                # print(image_full_name)
                while 'background' in image_full_name:  # Verify that we do not obtain a background image ={_}
                    image_full_name = images[np.random.randint(0, len(images))]
                    # print(image_full_name)
                # Load image
                img = np.array(imread(image_full_name), dtype=np.uint8)

                # Obtain image name to find the mask
                image_name = os.path.basename(image_full_name).split('.')[0]

                # Obtain the path to the mask required
                segmentation_dir = os.path.dirname(image_full_name)
                segmentation_mask_path = os.path.join(segmentation_dir, 'object_masks',
                                                      image_name + '_mask'+'.jpg')

                # Load the segmentation mask
                segmentation_mask = cv2.imread(segmentation_mask_path, 0)
                # print(segmentation_mask_path)
                # Remove noise im the image mask
                kernel = np.ones((3, 3), np.uint8)
                smoothed = cv2.GaussianBlur(segmentation_mask, (7, 7), 0)
                segmentation_mask = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=10)

                # we get the bounding box of the object and generate a transformation matrix
                try:
                    bb = get_bb_from_mask(segmentation_mask)
                except Exception:
                    print("\033[1;31mInvalid mask  \n {} \n {}\033[0;37m"
                          .format(image_full_name, segmentation_mask_path))
                    continue

                t = generate_transformation(bb, background_img.shape)

                # the object points are transformed with the given transformation matrix
                obj_coords = np.vstack((bb.nonzero_cols[np.newaxis],
                                        bb.nonzero_rows[np.newaxis],
                                        np.ones(len(bb.nonzero_rows), dtype=int)))
                transformed_obj_coords = t.dot(obj_coords)
                transformed_obj_coords = np.array(transformed_obj_coords, dtype=int)

                transformed_bb = get_bb(transformed_obj_coords)

                # the object is added to the background image
                for i, point in enumerate(transformed_obj_coords.T):
                    x = point[0]
                    y = point[1]
                    background_img[y, x] = img[obj_coords[1, i], obj_coords[0, i]]

                id = classes_to_id[object_class]
                augmented_object = {'class_id': id,
                                    'xmin': int(transformed_bb.min_coords.x),
                                    'xmax': int(transformed_bb.max_coords.x),
                                    'ymin': int(transformed_bb.min_coords.y),
                                    'ymax': int(transformed_bb.max_coords.y)}

                augmented_objects.append(augmented_object)

            if 'train' in annotations_file:
                output_path = os.path.join('training_images', str(augmented_img_counter) + '.jpg')
            elif 'val' in annotations_file:
                output_path = os.path.join('validation_images', str(augmented_img_counter) + '.jpg')

            # print(output_path)
            imwrite(output_path, background_img)
            training_images.append({'image_name': output_path, 'objects': augmented_objects})

            # image_name = output_path
            augmented_img_counter += 1
            print('Augmented {} out of {} images '
                  .format(augmented_img_counter, images_per_background * len(backgrounds)))

            if not augmented_img_counter % 1000:
                generate_annotation_file(annotations_file, training_images)

    generate_annotation_file(annotations_file, training_images)
    # print(training_images)


def generate_annotation_file(annotations_file, training_images):
    print('--- Annotations checkpoint --- ')
    with open(annotations_file, 'w') as annotation_file:
        yaml.safe_dump(training_images, annotation_file, default_flow_style=False, encoding='utf-8')


def generate_masks_and_annotations(data_dir, annotation_file):
    # check required directories
    TerminalColors.formatted_print('Data directory: ' + data_dir, TerminalColors.OKBLUE)
    if not os.path.exists(data_dir):
        raise RuntimeError('data directory does not exist: ' + data_dir)

    greenbox_image_dir = os.path.join(data_dir, 'green_box_images')
    print('looking for object green box images in: ' + greenbox_image_dir)
    if not os.path.exists(greenbox_image_dir):
        raise RuntimeError('directory for object green box images does not exist: ' + greenbox_image_dir)

    mask_dir = os.path.join(data_dir, 'object_masks')
    print('looking for object masks in: ' + mask_dir)
    if not os.path.exists(mask_dir):
        raise RuntimeError('directory for object masks does not exist: ' + mask_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to generate training images and annotations for bounding box based object detection."
                    " This is done by randomly projecting segemented object pixels onto backgrounds, then"
                    " calculating the corresponding bounding boxes.")
    parser.add_argument('--data-directory', '-d', required=True,
                        help='directory where the script will look for images, backgrounds and saved object masks')
    parser.add_argument('--class-annotations', '-c', required=True,
                        help='file containing mappings from class ID to class name')
    args = parser.parse_args()

    try:
        generate_masks_and_annotations(args.data_directory, args.class_annotations)
        TerminalColors.formatted_print('image and annotation generation complete', TerminalColors.OKGREEN)
    except KeyboardInterrupt:
        TerminalColors.formatted_print('\nscript interrupted', TerminalColors.WARNING)
    except Exception as e:
        TerminalColors.formatted_print(e, TerminalColors.FAIL)
