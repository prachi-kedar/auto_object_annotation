import os
import random
from tqdm import tqdm
import cv2
import concurrent.futures


def read_image(image_path):
    for file in os.listdir(image_path):
        image_path = f"{image_path}/{file}"

        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


def get_voc_xml(folder, filename, width, height, depth, objects):
    string = open(os.path.dirname(os.path.realpath(__file__)) + "/voc.xml").read()

    return string.format(
        folder=folder,
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        objects=objects,
        path=""
    )


def get_voc_object(name, xmin, ymin, xmax, ymax):
    string = open(os.path.dirname(os.path.realpath(__file__)) + "/voc_object.xml").read()

    return string.format(
        name=name,
        pose='unspecified',
        truncated=0,
        difficult=0,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax
    )


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


def place_objects(image, background, objects_per_image, output, n):
    bg_img_original = read_image(background)
    img = read_image(image)
    bg_shape = bg_img_original.shape
    img_shape = img.shape

    dim_0_range = bg_shape[0] - img_shape[0]
    dim_1_range = bg_shape[1] - img_shape[1]

    name = os.path.dirname(image).split("/")[-1]
    folder = "name"
    filename = os.path.basename(os.path.splitext(image)[0])

    for repeat in objects_per_image:
        bg_img = bg_img_original.copy()
        object_vocs = []
        filename_this = f"{filename}-{repeat}-{name}-{n}"

        for itr in range(repeat):
            loc_0 = random.choice(range(0, dim_0_range))
            loc_1 = random.choice(range(0, dim_1_range))
            bg_img = overlay_image(bg_img, img, loc_0, loc_1)
            object_vocs.append(get_voc_object(name, loc_1, loc_0, loc_1 + img_shape[1], loc_0 + img_shape[0]))

        complete_voc_objects = "".join(object_vocs)
        complete_voc = get_voc_xml(folder, filename_this, bg_shape[1], bg_shape[0], bg_shape[2], complete_voc_objects)

        # Directory
        annotation_directory = "annotations"

        img_directory = 'images'

        # Parent Directory path
        parent_dir = output

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

        with open(f"{output}/annotations/{filename_this}.xml", "w+") as fvoc:
            fvoc.write(complete_voc)
        cv2.imwrite(f"{output}/images/{filename_this}.jpg", bg_img)


def generate_annotation_fixed_size(images, backgrounds, images_per_object, objects_per_image, random_seed, threads,
                                   output):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

    if random_seed is not None:
        random.seed(random_seed)

    exec_args = []

    for n, image in enumerate(images):
        for itr in range(images_per_object):
            # randomly pick a background
            background = random.choice(backgrounds)
            exec_args.append([image, background, objects_per_image, output, n])

    for itr in tqdm(executor.map(
            lambda p: place_objects('/home/neosoft/Desktop/projects/automatic_object_annotation/img',
                                    '/home/neosoft/Desktop/projects/automatic_object_annotation/bag', range(4),
                                    '/home/neosoft/Desktop/projects/automatic_object_annotation/out', 1), exec_args),
            total=len(exec_args),
            desc="Generating images."):
        pass

    executor.shutdown(wait=True)


generate_annotation_fixed_size('/home/neosoft/Desktop/projects/automatic_object_annotation/img',
                               '/home/neosoft/Desktop/projects/automatic_object_annotation/bag', 3, range(4), 1, 1,
                               '/home/neosoft/Desktop/projects/automatic_object_annotation/out')