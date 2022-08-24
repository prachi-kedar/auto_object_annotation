import os
import random
import cv2
import config
from tqdm import tqdm
from scipy import ndimage
import numpy as np


# rotation angle in degree


def read_image(image_path):
    images_list = []
    image_name = []
    for file in os.listdir(image_path):
        image_path_ = f"{image_path}/{file}"

        images_list.append(cv2.imread(image_path_, cv2.IMREAD_UNCHANGED))
        image_name.append(file)
    return images_list, image_name, image_path_


def get_yolo_annotations(xmin, ymin, image_width, image_height, frame_width, frame_height):
    cx = xmin + ((image_width) / 2)
    cy = ymin + ((image_height) / 2)
    x = round((cx / frame_width), 4)
    y = round((cy / frame_height), 4)
    w = round((image_width / frame_width), 4)
    h = round((image_height / frame_height), 4)
    annotation_string_ = ("{} {:.4f} {:.4f} {:.4f} {:.4f}".format(4, x, y, w, h))
    return annotation_string_


def image_preprocessing(img):
    num = random.choice(range(1, 21))
    height, width = img.shape[:2]
    if num == 12:
        a = random.choice(range(0, 2))
        if a:
            img = img[:, 0:width // 2, :]
        else:
            img = img[:, 0:width * 2 // 3, :]
    elif num == 8:
        a = random.choice(range(0, 2))
        if a:
            img = img[:, width // 2:width, :]
        else:
            img = img[:, width * 2 // 3:width, :]
    else:
        img = img

    num = random.choice(range(1, 5))

    if num == 1:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    elif num == 2:
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    elif num == 3:
        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    elif num == 4:
        img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
    else:
        img = img

    num = random.choice(range(1, 5))

    if num < 3:
        img = cv2.flip(img, 1)
    elif num == 4:
        img = cv2.flip(img, 0)
    else:
        img = img
        # memberexperience@mahindraholidays.com

    a = random.choice(range(1, 8))
    if a <= 3:
        img = ndimage.rotate(img, random.choice(range(0, 90)))
    elif a > 3 & a <= 6:
        img = ndimage.rotate(img, random.choice(range(270, 360)))
    else:
        img = ndimage.rotate(img, random.choice(range(90, 270)))
    b = random.choice(range(1, 20))
    if b == 11:
        z = 0.5
        y = 0
    elif b == 15:
        z = 0
        y = 0.5
    else:
        z = 0
        y = 0
    M = np.float32([[1, z, 0],
                    [y, 1, 0],
                    [0, 0, 1]])
    rows, cols, dim = img.shape
    img = cv2.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))

    return img


def overlay_image(background_image, object_image, loc_0, loc_1):
    background_image = background_image.copy()

    if object_image.shape[2] == 4:
        alpha_object_image = object_image[:, :, 3] / 255.0
    else:
        alpha_object_image = 1

    alpha_background_image = 1.0 - alpha_object_image
    y1, y2 = loc_0, loc_0 + object_image.shape[0]
    x1, x2 = loc_1, loc_1 + object_image.shape[1]

    for c in range(0, 3):
        background_image[y1:y2, x1:x2, c] = (alpha_object_image * object_image[:, :, c] +
                                             alpha_background_image * background_image[y1:y2, x1:x2, c])

    return background_image


def place_objects(image, background, objects_per_image,output):
    bg_img_original, bg_name, bg_path = read_image(background)
    img, img_name, img_path = read_image(image)
    folder_name = img_path.split('/')[-3]


    for b in tqdm(range(len(img)), desc="Generating Annotations..."):

        for i, k in zip(bg_img_original, bg_name):
            j = image_preprocessing(img[b])
            k = k.split('.')[0]

            bg_shape = i.shape
            img_shape = j.shape
            if bg_shape[0] > img_shape[0] and bg_shape[1] > img_shape[1]:
                dim_0_range = bg_shape[0] - img_shape[0]
                dim_1_range = bg_shape[1] - img_shape[1]

                name = img_name[b].split('.')[0]

                for repeat in objects_per_image:
                    bg_img = i.copy()
                    object_vocs = []

                    filename_this = f"{repeat}-{name}-{k}--{folder_name}"

                    for itr in range(repeat):
                        loc_0 = random.choice(range(0, dim_0_range))
                        loc_1 = random.choice(range(0, dim_1_range))
                        bg_img = overlay_image(bg_img, j, loc_0, loc_1)
                        object_vocs.append(
                            get_yolo_annotations(loc_1, loc_0, img_shape[1], img_shape[0], bg_shape[1], bg_shape[0]))

                    complete_voc_objects = '\n'.join(object_vocs)

                    # Directory
                    annotation_directory = "annotations"

                    img_directory = 'images'

                    # Parent Directory path
                    parent_dir = 'output'

                    # Path
                    annotation_path = os.path.join(parent_dir, annotation_directory)
                    img_path = os.path.join(parent_dir, img_directory)

                    # Create the directory

                    isAnnotaionsExist = os.path.exists(annotation_path)
                    isImgExist = os.path.exists(img_path)

                    if not isAnnotaionsExist:
                        os.mkdir(annotation_path)

                    if not isImgExist:
                        os.mkdir(img_path)

                    with open(f"{output}/annotations/{filename_this}.txt", "w+") as fvoc:
                        fvoc.write(complete_voc_objects)
                    cv2.imwrite(f"{output}/images/{filename_this}.jpg", bg_img)


place_objects(config.OBJECT_IMG_DIRECTORY, config.BAGROUND_IMG_DIRECTORY, config.NO_OF_OBJECTS, config.OUTPUT_DIRECTORY)
