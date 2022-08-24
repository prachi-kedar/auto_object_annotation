import os
import random
import cv2
import config


def read_image(image_path):
    images_list = []
    image_name = []
    for file in os.listdir(image_path):
        image_path_ = f"{image_path}/{file}"

        images_list.append(cv2.imread(image_path_, cv2.IMREAD_UNCHANGED))
        image_name.append(file)
    return images_list, image_name


def get_yolo_annotations(xmin,ymin,image_width,image_height,frame_width,frame_height):
    cx = xmin + ((image_width) / 2)
    cy = ymin + ((image_height) / 2)
    x = round((cx / frame_width), 4)
    y = round((cy / frame_height), 4)
    w = round((image_width / frame_width), 4)
    h = round((image_height / frame_height), 4)
    annotation_string_ = ("{} {:.4f} {:.4f} {:.4f} {:.4f}".format(0, x,y,w,h))
    return annotation_string_


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


def place_objects(image, background, objects_per_image, output):
    bg_img_original, bg_name = read_image(background)
    img, img_name = read_image(image)
    # combinations = len(img) * len(bg_img_original)

    for i, j, k, l in zip(bg_img_original, img, bg_name, img_name):
        l = l.split('.')[0]
        k = k.split('.')[0]

        bg_shape = i.shape
        img_shape = j.shape

        if bg_shape[0] > img_shape[0] and bg_shape[1] > img_shape[1]:
            dim_0_range = bg_shape[0] - img_shape[0]
            dim_1_range = bg_shape[1] - img_shape[1]

            name = os.path.dirname(image).split('/')[-1]

            for repeat in objects_per_image:
                bg_img = i.copy()
                object_vocs = []

                filename_this = f"{name}-{k}--{l}"

                for itr in range(repeat):
                    loc_0 = random.choice(range(0, dim_0_range))
                    loc_1 = random.choice(range(0, dim_1_range))
                    if len(bg_img.shape) == 3:
                        bg_img = overlay_image(bg_img, j, loc_0, loc_1)

                        object_vocs.append(get_yolo_annotations(loc_1, loc_0, img_shape[1],img_shape[0], bg_shape[1],bg_shape[0] ))


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
