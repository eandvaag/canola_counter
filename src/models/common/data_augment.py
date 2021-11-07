import tensorflow as tf

def flip_horizontal(image, boxes, probability):
    """
        boxes: must be normalized and in (min_x, min_y, max_x, max_y) format
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def flip_vertical(image, boxes, probability):
    """
        boxes: must be normalized and in (min_x, min_y, max_x, max_y) format
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        boxes = tf.stack(
            [boxes[:, 0], 1 - boxes[:, 3], boxes[:, 2], 1 - boxes[:, 1]], axis=-1
        )
    return image, boxes


def adjust_brightness(image, min_factor, max_factor):

    factor = tf.random.uniform((), minval=min_factor, maxval=max_factor)
    image = tf.image.adjust_brightness(image, factor)
    return image


def adjust_saturation(image, min_factor, max_factor):

    factor = tf.random.uniform((), minval=min_factor, maxval=max_factor)
    image = tf.image.adjust_brightness(image, factor)
    return image  


def apply_augmentations(augmentations, image, boxes):

    for augmentation in augmentations:
        aug_type = augmentation["type"]
        aug_params = augmentation["parameters"]

        if aug_type == "flip_horizontal":
            image, boxes = flip_horizontal(image, boxes, aug_params["probability"])
        elif aug_type == "flip_vertical":
            image, boxes = flip_vertical(image, boxes, aug_params["probability"])
        elif aug_type == "adjust_brightness":
            image = adjust_brightness(image, aug_params["min_factor"], aug_params["max_factor"])
        elif aug_type == "adjust_saturation":
            image = adjust_saturation(image, aug_params["min_factor"], aug_params["max_factor"])
        else:
            raise RuntimeError("Unrecognized augmentation type: '{}'".format(aug_type))

    return image, boxes
