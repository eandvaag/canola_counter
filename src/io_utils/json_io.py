import json
import os
import numpy as np
import cv2
import tensorflow as tf

import extract_patches as ep
from models.detectors.common.box_utils import non_max_suppression
from io_utils import xml_io
from dataset import Img

def save_json(path, data):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data



def read_patch_predictions(model, patch_dir, read_patches=False):

    pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
    pred_path = os.path.join(pred_dir, "predictions.json")

    pred_data = load_json(pred_path)
    patch_info = ep.parse_patch_dir(pred_dir)
    is_annotated = patch_info["is_annotated"]

    ret_data = []

    for pred in pred_data["predictions"]:

        ret_pred = {}

        if read_patches:
            ret_pred["patch"] = cv2.imread(pred["patch_path"])
        ret_pred["patch_path"] = pred["patch_path"]
        ret_pred["img_path"] = pred["img_path"]
        ret_pred["pred_patch_abs_boxes"] = np.rint(pred["pred_patch_abs_boxes"]).astype(np.int32)
        ret_pred["pred_classes"] = np.array(pred["pred_classes"]).astype(np.int32)
        ret_pred["patch_coords"] = np.array(pred["patch_coords"]).astype(np.int32)
        ret_pred["pred_scores"] = np.array(pred["pred_scores"]).astype(np.float32)
        if ret_pred["pred_patch_abs_boxes"].size == 0:
            ret_pred["pred_img_abs_boxes"] = np.copy(ret_pred["pred_patch_abs_boxes"])
        else:
            ret_pred["pred_img_abs_boxes"] = np.rint(np.array(pred["pred_patch_abs_boxes"]) + \
                                         np.tile(np.array(pred["patch_coords"])[:2], 2)).astype(np.int32)

        #nms_boxes, nms_classes, nms_scores = non_max_suppression(
        #                                        np.array(ret_pred["pred_patch_abs_boxes"]),
        #                                        np.array(ret_pred["pred_classes"]),
        #                                        np.array(ret_pred["pred_scores"]),
        #                                        iou_thresh=settings.img_nms_iou_thresh)

        #ret_pred["nms_pred_patch_abs_boxes"] = nms_boxes
        #ret_pred["nms_pred_classes"] = nms_classes
        #ret_pred["nms_pred_scores"] = nms_scores

        #ret_pred["sel_nms_patch_inds"] = np.where(ret_pred["nms_pred_scores"] > settings.score_thresh)[0]


        if is_annotated:
            ret_pred["patch_normalized_boxes"] = np.array(pred["patch_normalized_boxes"]).astype(np.float32)
            ret_pred["patch_abs_boxes"] = np.array(pred["patch_abs_boxes"]).astype(np.int32)
            ret_pred["img_abs_boxes"] = np.array(pred["img_abs_boxes"]).astype(np.int32)
            ret_pred["patch_classes"] = np.array(pred["patch_classes"]).astype(np.int32)


        ret_data.append(ret_pred)

    return ret_data, is_annotated




def read_img_predictions(model, patch_dir, settings):

    pred_data, is_annotated = read_patch_predictions(model, patch_dir, read_patches=False)

    img_data = {}

    for pred in pred_data:

        img_path = pred["img_path"]

        if img_path not in img_data:

            img_data[img_path] = {
                "pred_img_abs_boxes": np.array([]).reshape(0, 4),
                "pred_classes": np.array([]),
                "pred_scores": np.array([]),
                "patch_coords": np.array([]).reshape(0, 4)
            }


        img_data[img_path]["pred_img_abs_boxes"] = np.vstack(
            [img_data[img_path]["pred_img_abs_boxes"], pred["pred_img_abs_boxes"].reshape(-1, 4)])

        img_data[img_path]["pred_classes"] = np.concatenate(
            [img_data[img_path]["pred_classes"], pred["pred_classes"]])

        img_data[img_path]["pred_scores"] = np.concatenate(
            [img_data[img_path]["pred_scores"], pred["pred_scores"]])

        img_data[img_path]["patch_coords"] = np.vstack(
            [img_data[img_path]["patch_coords"], pred["patch_coords"]])



    for img_path in img_data.keys():

        nms_boxes, nms_classes, nms_scores = non_max_suppression(
                                                np.array(img_data[img_path]["pred_img_abs_boxes"]),
                                                np.array(img_data[img_path]["pred_classes"]),
                                                np.array(img_data[img_path]["pred_scores"]),
                                                iou_thresh=settings.img_nms_iou_thresh)
        img_data[img_path]["nms_pred_img_abs_boxes"] = nms_boxes
        img_data[img_path]["nms_pred_classes"] = nms_classes
        img_data[img_path]["nms_pred_scores"] = nms_scores

        if is_annotated:
            gt_boxes, gt_classes = xml_io.load_boxes_and_classes(Img(img_path).xml_path, settings.class_map)
            img_data[img_path]["img_abs_boxes"] = gt_boxes
            img_data[img_path]["img_classes"] = gt_classes

    return img_data, is_annotated
