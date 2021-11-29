import albumentations as A
import numpy as np
import tensorflow as tf


# All augmentation functions assume the following format:
# images are float32 with range 0 - 255
# boxes are normalized and in (min_x, min_y, max_x, max_y) format


def flip_horizontal(image, boxes):
    image = tf.image.flip_left_right(image)
    boxes = tf.stack(
        [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
    )
    return image, boxes


def flip_horizontal_rand(image, boxes):
    if tf.random.uniform(()) < 0.5:
        image, boxes = flip_horizontal(image, boxes)
    return image, boxes


def flip_vertical(image, boxes):
    image = tf.image.flip_up_down(image)
    boxes = tf.stack(
        [boxes[:, 0], 1 - boxes[:, 3], boxes[:, 2], 1 - boxes[:, 1]], axis=-1
    )
    return image, boxes


def flip_vertical_rand(image, boxes):
    if tf.random.uniform(()) < 0.5:
        image, boxes = flip_vertical(image, boxes)
    return image, boxes


def rotate_90(image, boxes, k):
    """
        k: number of 90 degree rotations
    """
    image = tf.image.rot90(image, k=k)

    for i in range(k):
        xs = tf.stack([boxes[:, 1], boxes[:, 3]], axis=-1)
        ys = tf.stack([1 - boxes[:, 0], 1 - boxes[:, 2]], axis=-1)
        boxes = tf.stack(
            [tf.math.reduce_min(xs, axis=-1),
             tf.math.reduce_min(ys, axis=-1),
             tf.math.reduce_max(xs, axis=-1),
             tf.math.reduce_max(ys, axis=-1)], axis=-1)
        #boxes = tf.stack(
        #    [boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3], 1 - boxes[:, 2]], axis=-1
        #)
    return image, boxes


def rotate_90_rand(image, boxes):
    k = tf.random.uniform(shape=(), minval=0, maxval=3, dtype=tf.int32)
    return rotate_90(image, boxes, k)


def adjust_brightness(image, factor):
    image /= 255.
    image = tf.image.adjust_brightness(image, factor)
    image *= 255.
    return image


def adjust_brightness_rand(image, min_factor, max_factor):
    factor = tf.random.uniform((), minval=min_factor, maxval=max_factor)
    image = adjust_brightness(image, factor)
    return image


def adjust_saturation(image, factor):
    image /= 255.
    image = tf.image.adjust_saturation(image, factor)
    image *= 255.
    return image  


def adjust_saturation_rand(image, min_factor, max_factor):
    factor = tf.random.uniform((), minval=min_factor, maxval=max_factor)
    image = adjust_saturation(image, factor)
    return image




def apply_augmentations(augmentations, image, boxes, classes):


    aug_methods = []

    for augmentation in augmentations:
        aug_type = augmentation["type"]
        aug_params = augmentation["parameters"] if "parameters" in augmentation else None

        if aug_type == "flip_horizontal":
            aug_methods.append(A.HorizontalFlip(p=aug_params["probability"]))#0.5))
        elif aug_type == "flip_vertical":
            aug_methods.append(A.VerticalFlip(p=aug_params["probability"]))#0.5))
        elif aug_type == "rotate":
            aug_methods.append(A.Rotate(p=aug_params["probability"], limit=aug_params["limit"]))
        elif aug_type == "rotate_90":
            aug_methods.append(A.RandomRotate90(p=aug_params["probability"]))
        elif aug_type == "affine":
            aug_methods.append(A.Affine(p=aug_params["probability"],
                                        scale=aug_params["scale"], 
                                        translate_percent=aug_params["translate_percent"],
                                        rotate=aug_params["rotate"],
                                        shear=aug_params["shear"]))
        elif aug_type == "brightness_contrast":
            aug_methods.append(A.RandomBrightnessContrast(p=aug_params["probability"],
                                                          brightness_limit=aug_params["brightness_limit"],
                                                          contrast_limit=aug_params["contrast_limit"]))
        elif aug_type == "rgb_shift":
            aug_methods.append(A.RGBShift(p=aug_params["probability"],
                                          r_shift_limit=aug_params["r_shift_limit"],
                                          g_shift_limit=aug_params["g_shift_limit"],
                                          b_shift_limit=aug_params["b_shift_limit"]))



    transform = A.Compose(aug_methods, bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    transformed = transform(image=image, bboxes=boxes, class_labels=classes)
    tf_img = transformed["image"]
    tf_boxes = np.array(transformed["bboxes"]).reshape(-1, 4)
    tf_classes = np.array(transformed["class_labels"])

    return tf_img, tf_boxes, tf_classes




def apply_augmentations_custom(augmentations, image, boxes):

    for augmentation in augmentations:
        aug_type = augmentation["type"]
        aug_params = augmentation["parameters"] if "parameters" in augmentation else None

        if aug_type == "flip_horizontal":
            image, boxes = flip_horizontal_rand(image, boxes)
        elif aug_type == "flip_vertical":
            image, boxes = flip_vertical_rand(image, boxes)
        elif aug_type == "rotate_90":
            image, boxes = rotate_90_rand(image, boxes)
        elif aug_type == "adjust_brightness":
            image = adjust_brightness_rand(image, aug_params["min_factor"], aug_params["max_factor"])
        elif aug_type == "adjust_saturation":
            image = adjust_saturation_rand(image, aug_params["min_factor"], aug_params["max_factor"])
        else:
            raise RuntimeError("Unrecognized augmentation type: '{}'".format(aug_type))

    return image, boxes
