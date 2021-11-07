import numpy as np
import tensorflow as tf


def swap_xy_np(boxes):

    return np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def swap_xy_tf(boxes):
    """Swaps the order of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh_tf(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners_tf(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def convert_to_openseadragon_format(boxes, img_width, img_height):
    """
        input boxes are in min_y, min_x, max_y, max_x format
    """

    min_y = boxes[..., 0] / img_height
    min_x = boxes[..., 1] / img_width
    h = (boxes[..., 2] - boxes[..., 0]) / img_height
    w = (boxes[..., 3] - boxes[..., 1]) / img_width

    return np.stack([min_x, min_y, w, h], axis=-1)


def non_max_suppression(boxes, classes, scores, iou_thresh):

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)
    sel_boxes = tf.gather(boxes, sel_indices).numpy()
    sel_classes = tf.gather(classes, sel_indices).numpy()
    sel_scores = tf.gather(scores, sel_indices).numpy()
    
    return sel_boxes, sel_classes, sel_scores






