import json
import os
import numpy as np
import cv2
import tensorflow as tf

def save_json(path, data):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data



def read_patch_predictions(pred_dir, read_patches=False):

    pred_data = load_json(os.path.join(pred_dir, "predictions.json"))

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
        ret_pred["pred_img_abs_boxes"] = np.rint(np.array(pred["pred_patch_abs_boxes"]) + \
                                         np.tile(np.array(pred["patch_coords"])[:2], 2)).astype(np.int32)


        ret_data.append(ret_pred)

    return ret_data


def non_max_suppression(boxes, classes, scores, iou_threshold):

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_threshold)
    sel_boxes = tf.gather(boxes, sel_indices).numpy()
    sel_classes = tf.gather(classes, sel_indices).numpy()
    sel_scores = tf.gather(scores, sel_indices).numpy()
    
    return sel_boxes, sel_classes, sel_scores




def read_img_predictions(pred_dir, nms_iou_threshold):

    pred_data = read_patch_predictions(pred_dir)

    img_data = {}

    for pred in pred_data:

        img_path = pred["img_path"]
        if img_path not in img_data:

            img_data[img_path] = {
                "pred_boxes": pred["pred_img_abs_boxes"],
                "pred_classes": pred["pred_classes"],
                "pred_scores": pred["pred_scores"],
                "patch_coords": pred["patch_coords"]
            }

        else:

            img_data[img_path]["pred_boxes"] = np.vstack(
                [img_data[img_path]["pred_boxes"], pred["pred_img_abs_boxes"]])
            img_data[img_path]["pred_classes"] = np.concatenate(
                [img_data[img_path]["pred_classes"], pred["pred_classes"]])
            img_data[img_path]["pred_scores"] = np.concatenate(
                [img_data[img_path]["pred_scores"], pred["pred_scores"]])
            img_data[img_path]["patch_coords"] = np.vstack(
                [img_data[img_path]["patch_coords"], pred["patch_coords"]])

    for img_path in img_data.keys():
        print("pred_boxes", img_data[img_path]["pred_boxes"])
        sel_boxes, sel_classes, sel_scores = non_max_suppression(
                                                np.array(img_data[img_path]["pred_boxes"]),
                                                np.array(img_data[img_path]["pred_classes"]),
                                                np.array(img_data[img_path]["pred_scores"]),
                                                iou_threshold=nms_iou_threshold)
        img_data[img_path]["sel_boxes"] = sel_boxes
        img_data[img_path]["sel_classes"] = sel_classes
        img_data[img_path]["sel_scores"] = sel_scores

    return img_data