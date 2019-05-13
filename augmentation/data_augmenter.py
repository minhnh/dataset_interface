#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import pandas as pd
import yaml

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
    bb.nonzero_rows, bb.nonzero_cols  = np.where(segmentation_mask)
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
        random_scaling_factor = np.random.uniform(0.5, 1.0)
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
                use_transformation = False
                break
    return t

def augment_data(img_dir_name: str,
                 background_img_dir: str,
                 images_per_background: int,
                 class_id: int,
                 train_annotations_file: str,
                 val_annotations_file: str) -> None:
    '''Given the images in "img_dir_name", each of which is assumed to have a
    single object, generates a new set of images in which the objects are put
    on the backgrounds in "background_img_dir" and are transformed (translated,
    rotated, scaled) in random fashion. For each background and image combination,
    "images_per_background" images are generated.

    Keyword arguments:
    img_dir_name: str -- path to a directory with image files and object
                         segmentation masks (img_dir_name is expected to have
                         a directory "object_masks" with the segmentation masks,
                         such that if an image is called "test.jpg", its segmentation
                         mask will have the name "test_mask.jpg")
    background_img_dir: str -- path to a directory with background images for augmentation
    images_per_background: int -- number of images to generate per given background

    '''
    # we create a directory for the augmented data
    if not os.path.isdir('augmented_data_train'):
        os.mkdir(os.path.join(img_dir_name, 'augmented_data'))

    if not os.path.isdir('augmented_data_val'):
        os.mkdir(os.path.join(img_dir_name, 'augmented_data_val'))

    images = os.listdir(img_dir_name)
    backgrounds = os.listdir(background_img_dir)

    # we count the number of images
    total_img_counter = 0
    for img_counter, img_name in enumerate(images):
        img_path = os.path.join(img_dir_name, img_name)
        if os.path.isfile(img_path):
            total_img_counter += 1

    # we get the full paths of the background images so that
    # we don't have to recreate them every time we iterate
    # through the background images
    background_paths = []
    for background in backgrounds:
        background_path = os.path.join(background_img_dir, background)
        background_paths.append(background_path)


    for background_path in background_paths:
        background_img_original = np.array(imread(background_path), dtype=int)


        for _ in range(images_per_background): # Number of images generated using that backgrounds
            for _ in range(np.random.randint(6)): # Number of objects in the image
                img_name = images[np.random.randint(total_img_counter)] # Choose a random image
                img_path

    img_counter = 0
    for img_file_name in images:
        img_path = os.path.join(img_dir_name, img_file_name)
        if not os.path.isfile(img_path):
            continue
        img_counter += 1
        print('Augmenting image {0} of {1}'.format(img_counter, total_img_counter))

        # we read the image and its object segmentation mask
        img = np.array(imread(img_path), dtype=np.uint8)
        img_name, img_extension = img_file_name.split('.')
        segmentation_mask_name = os.path.join(img_dir_name, 'object_masks',
                                              img_name + '_mask.' + img_extension)
        segmentation_mask = np.array(imread(segmentation_mask_name), dtype=int)

        # we get the bounding box of the object and generate a transformation matrix
        bb = get_bb_from_mask(segmentation_mask)

        augmented_img_counter = 0
        for background_path in background_paths:
            background_img_original = np.array(imread(background_path), dtype=int)
            for _ in range(images_per_background):
                background_img = np.array(background_img_original)
                t = generate_transformation(bb, background_img.shape)

                # the object points are transformed with the given transformation matrix
                obj_coords = np.vstack((bb.nonzero_cols[np.newaxis],
                                        bb.nonzero_rows[np.newaxis],
                                        np.ones(len(bb.nonzero_rows), dtype=int)))
                transformed_obj_coords = t.dot(obj_coords)
                transformed_obj_coords = np.array(transformed_obj_coords, dtype=int)

                transformed_bb = get_bb(transformed_obj_coords)

                # the object is added to the background image
                augmented_img = np.array(background_img, dtype=np.uint8)
                for i, point in enumerate(transformed_obj_coords.T):
                    x = point[0]
                    y = point[1]
                    augmented_img[y, x] = img[obj_coords[1, i], obj_coords[0, i]]


                if img_counter <= 15:
                    # we finally save the augmented image
                    augmented_img_path = os.path.join(img_dir_name, 'augmented_data',
                                                      img_name + '_' + str(augmented_img_counter) + \
                                                      '_augmented.' + img_extension)
                    imwrite(augmented_img_path, augmented_img)
                    save_bounding_box(transformed_bb, train_annotations_file, augmented_img_path, class_id)
                else:
                    # we finally save the augmented image
                    augmented_img_path = os.path.join(img_dir_name, 'augmented_data_val',
                                                      img_name + '_' + str(augmented_img_counter) + \
                                                      '_augmented.' + img_extension)
                    imwrite(augmented_img_path, augmented_img)
                    save_bounding_box(transformed_bb, val_annotations_file, augmented_img_path, class_id)


                augmented_img_counter += 1

def generate_annotation_file(train_annotations_file, train_images):
    # dummy_dict.append({'image_name': 'demo2', 'objects': [{'class_id':1, 'xmin':1, 'xmax':2, 'ymin':2, 'ymax':2},{'class_id':1, 'xmin':1, 'xmax':2, 'ymin':2, 'ymax':2}]})
    pass

def save_bounding_box(bb, train_annotations_file, img_file_name, class_id):
    if os.path.isfile(train_annotations_file):
        df = pd.read_csv(train_annotations_file, sep=',')
        # print('1 ', df)
        df = df.append(pd.Series([img_file_name, bb.min_coords.x, bb.max_coords.x, bb.min_coords.y, bb.max_coords.y, class_id], \
                            index=df.columns), \
                  ignore_index=True)
        # print('2 ', df)
        # print()
    else:
        # print([img_file_name, bb.min_coords.x, bb.max_coords.x, bb.min_coords.y, bb.max_coords.y, class_id])
        df = pd.DataFrame([[img_file_name, bb.min_coords.x, bb.max_coords.x, bb.min_coords.y, bb.max_coords.y, class_id]],\
            columns = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
    df.to_csv(train_annotations_file, sep=',', index=False)


if __name__ == '__main__':
    img_dir_name = sys.argv[1]
    background_img_dir = sys.argv[2]
    images_per_background = int(sys.argv[3])
    class_name = sys.argv[4]
    class_id = int(sys.argv[5])
    train_annotations_file = sys.argv[6]
    val_annotations_file = sys.argv[7]

    with open('classes.yml', 'r') as class_file:
        classes = yaml.load(class_file)
        if classes is None:
            classes = dict()
        classes[class_id] = class_name

    with open('classes.yml', 'w') as class_file:
        yaml.dump(classes, class_file,default_flow_style=False)

    print('Augmenting data...')
    augment_data(img_dir_name, background_img_dir, images_per_background, class_id, train_annotations_file, val_annotations_file)
    print('Data augmentation complete')
