import os
import cv2
import numpy as np

from models.common.box_utils import non_max_suppression
from image_set import Img
from io_utils import json_io, xml_io


def decode_patch_predictions(predictions):

    patch_data = {}

    is_annotated = predictions["is_annotated"]
    for pred in predictions["predictions"]:

        patch_path = pred["patch_path"]
        patch_data[patch_path] = {}

        patch_data[patch_path]["img_path"] = pred["img_path"]
        patch_data[patch_path]["pred_patch_abs_boxes"] = np.rint(pred["pred_patch_abs_boxes"]).astype(np.int32)
        patch_data[patch_path]["pred_classes"] = np.array(pred["pred_classes"]).astype(np.int32)
        patch_data[patch_path]["patch_coords"] = np.array(pred["patch_coords"]).astype(np.int32)
        patch_data[patch_path]["pred_scores"] = np.array(pred["pred_scores"]).astype(np.float32)
        if patch_data[patch_path]["pred_patch_abs_boxes"].size == 0:
            patch_data[patch_path]["pred_img_abs_boxes"] = np.copy(patch_data[patch_path]["pred_patch_abs_boxes"])
        else:
            patch_data[patch_path]["pred_img_abs_boxes"] = np.rint(np.array(pred["pred_patch_abs_boxes"]) + \
                                                            np.tile(np.array(pred["patch_coords"])[:2], 2)).astype(np.int32)

        if is_annotated:
            patch_data[patch_path]["patch_normalized_boxes"] = np.array(pred["patch_normalized_boxes"]).astype(np.float32)
            patch_data[patch_path]["patch_abs_boxes"] = np.array(pred["patch_abs_boxes"]).astype(np.int32)
            patch_data[patch_path]["img_abs_boxes"] = np.array(pred["img_abs_boxes"]).astype(np.int32)
            patch_data[patch_path]["patch_classes"] = np.array(pred["patch_classes"]).astype(np.int32)

    return patch_data




def decode_img_predictions(predictions, class_map, img_nms_iou_thresh):

    is_annotated = predictions["is_annotated"]
    decoded_predictions = decode_patch_predictions(predictions)

    img_data = {}

    for pred in decoded_predictions.values():

        img_path = pred["img_path"]

        if img_path not in img_data:

            img_data[img_path] = {
                "pred_img_abs_boxes": np.array([]).reshape(0, 4).astype(np.int32),
                "pred_classes": np.array([]).astype(np.int32),
                "pred_scores": np.array([]).astype(np.float32),
                "patch_coords": np.array([]).reshape(0, 4).astype(np.int32)
            }


        img_data[img_path]["pred_img_abs_boxes"] = np.vstack(
            [img_data[img_path]["pred_img_abs_boxes"], pred["pred_img_abs_boxes"].reshape(-1, 4)]).astype(np.int32)

        img_data[img_path]["pred_classes"] = np.concatenate(
            [img_data[img_path]["pred_classes"], pred["pred_classes"]]).astype(np.int32)

        img_data[img_path]["pred_scores"] = np.concatenate(
            [img_data[img_path]["pred_scores"], pred["pred_scores"]]).astype(np.float32)

        img_data[img_path]["patch_coords"] = np.vstack(
            [img_data[img_path]["patch_coords"], pred["patch_coords"]]).astype(np.int32)



    for img_path in img_data.keys():

        nms_boxes, nms_classes, nms_scores = non_max_suppression(
                                                np.array(img_data[img_path]["pred_img_abs_boxes"]),
                                                np.array(img_data[img_path]["pred_classes"]),
                                                np.array(img_data[img_path]["pred_scores"]),
                                                iou_thresh=img_nms_iou_thresh)
        img_data[img_path]["nms_pred_img_abs_boxes"] = nms_boxes
        img_data[img_path]["nms_pred_classes"] = nms_classes
        img_data[img_path]["nms_pred_scores"] = nms_scores

        if is_annotated:
            gt_boxes, gt_classes = xml_io.load_boxes_and_classes(Img(img_path).xml_path, class_map)
            img_data[img_path]["img_abs_boxes"] = gt_boxes
            img_data[img_path]["img_classes"] = gt_classes

    return img_data
