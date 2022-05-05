import math as m
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


def box_areas_np(boxes):
    """
        boxes: min_y, min_x, max_y, max_x format
    """

    return (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])

def box_visibilities_np(boxes, clipped_boxes):

    box_areas = box_areas_np(boxes)
    clipped_box_areas = box_areas_np(clipped_boxes)
    visibilities = np.divide(clipped_box_areas, box_areas, out=np.zeros_like(clipped_box_areas, dtype="float64"), where=box_areas!=0)
    return visibilities

def clip_boxes_and_get_small_visibility_mask(boxes, patch_coords, min_visibility):

    clipped_boxes = clip_boxes_np(boxes, patch_coords)
    box_visibilities = box_visibilities_np(boxes, clipped_boxes)
    mask = box_visibilities >= min_visibility
    return clipped_boxes, mask


def get_edge_boxes_mask(boxes, patch_shape):
    #if box_fmt == "xywh":
    #    boxes = convert_to_corners_tf(boxes)
    mask = np.logical_or(np.logical_or(boxes[:, 0] <= 0, boxes[:, 1] <= 0), 
                  np.logical_or(boxes[:, 2] >= patch_shape[0]-1, boxes[:, 3] >= patch_shape[1]-1))
    #edge_boxes = boxes[mask]
    #non_edge_boxes = boxes[np.logical_not(mask)]
    return mask #edge_boxes, non_edge_boxes



# def get_normalized_patch_wh(training_dataset):

#     BOX_PATCH_RATIO = 0.02

#     mean_box_area = training_dataset.get_mean_box_area()
#     patch_area = mean_box_area / BOX_PATCH_RATIO

#     patch_wh = round(m.sqrt(patch_area))

#     return patch_wh


# def clip_boxes_np(boxes, img_width, img_height):
#     """
#         boxes: min_y, min_x, max_y, max_x format
#     """

#     boxes = np.concatenate([np.maximum(boxes[:, :2], [0, 0]),
#                             np.minimum(boxes[:, 2:], [img_height-1, img_width-1])], axis=-1)
#     return boxes


def clip_boxes_np(boxes, patch_coords):
    """
        boxes: min_y, min_x, max_y, max_x format
    """

    #boxes = np.stack([np.maximum(boxes[:,0], patch_coords[0]),
    #                           np.maximum(boxes[:,1], patch_coords[1]),
    #                           np.minimum(boxes[:,2], patch_coords[2]-1),
    #                           np.minimum(boxes[:,3], patch_coords[3]-1)], axis=-1)

    boxes = np.concatenate([np.maximum(boxes[:, :2], [patch_coords[0], patch_coords[1]]),
                            np.minimum(boxes[:, 2:], [patch_coords[2], patch_coords[3]])], axis=-1)
    return boxes

def non_max_suppression_with_classes(boxes, classes, scores, iou_thresh):

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)
    sel_boxes = tf.gather(boxes, sel_indices).numpy()
    sel_classes = tf.gather(classes, sel_indices).numpy()
    sel_scores = tf.gather(scores, sel_indices).numpy()
    
    return sel_boxes, sel_classes, sel_scores

def non_max_suppression(boxes, scores, iou_thresh):

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)
    sel_boxes = tf.gather(boxes, sel_indices).numpy()
    sel_scores = tf.gather(scores, sel_indices).numpy()

    return sel_boxes, sel_scores


def compute_iou(boxes1, boxes2, box_format="xywh"):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

      box_format: "xywh" or "corners_xy" or "corner_yx"
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """



    if box_format == "xywh":
        boxes1_corners = convert_to_corners_tf(boxes1)
        boxes2_corners = convert_to_corners_tf(boxes2)
        #boxes1_area = boxes1[:, 2] * boxes1[:, 3]
        #boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    elif box_format == "corners_yx":
        boxes1_corners = swap_xy_tf(boxes1) #tf.convert_to_tensor(boxes1, dtype=tf.float32))
        boxes2_corners = swap_xy_tf(boxes2) #tf.convert_to_tensor(boxes2, dtype=tf.float32))
    elif box_format == "corners_xy":
        boxes1_corners = boxes1
        boxes2_corners = boxes2
    else:
        raise RuntimeError("Unrecognized box format")

    boxes1_area = (boxes1_corners[:,2] - boxes1_corners[:,0]) * (boxes1_corners[:,3] - boxes1_corners[:,1])
    boxes2_area = (boxes2_corners[:,2] - boxes2_corners[:,0]) * (boxes2_corners[:,3] - boxes2_corners[:,1])

    print("boxes1_corners", boxes1_corners)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    res = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


    return res




def compute_iou_np(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

      box_format: [min_y, min_x, max_y, max_x]
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """

    boxes1_corners = boxes1
    boxes2_corners = boxes2

    boxes1_area = (boxes1_corners[:,2] - boxes1_corners[:,0]) * (boxes1_corners[:,3] - boxes1_corners[:,1])
    boxes2_area = (boxes2_corners[:,2] - boxes2_corners[:,0]) * (boxes2_corners[:,3] - boxes2_corners[:,1])

    lu = np.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = np.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    union_area = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    res = np.clip(intersection_area / union_area, 0.0, 1.0)

    return res

