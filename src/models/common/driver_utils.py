import os
import logging
import pandas as pd
import pandas.io.formats.excel
from natsort import index_natsorted
import numpy as np

from io_utils import xml_io
import extract_patches as ep
from image_set import Img
from models.common import box_utils



def create_patches(extraction_params, img_set, dataset_name):
    dataset = img_set.datasets[dataset_name]

    scenario_id = ep.extract_patches(extraction_params,
                                     img_set,
                                     dataset,
                                     annotate_patches=dataset.is_annotated)

    patch_dir = os.path.join(img_set.patch_dir, scenario_id)
    return patch_dir, dataset.is_annotated


# def constant_learning_rate_function(steps_taken, training_steps_per_epoch, config):
#     return config.training["active"]["learning_rate_schedule"]["learning_rate"]

# def get_learning_rate_function(config):

#     schedule = config.training["active"]["learning_rate_schedule"]
#     schedule_type = schedule["schedule_type"]

#     if schedule_type == "constant"
#         f = constant_learning_rate_function




def get_learning_rate(steps_taken, training_steps_per_epoch, config):

    schedule = config.training["active"]["learning_rate_schedule"]
    schedule_type = schedule["schedule_type"]


    if schedule_type == "constant":
        cur_lr = schedule["learning_rate"]

    elif schedule_type == "cosine_annealing":
        lr_init = schedule["learning_rate_init"]
        lr_end = schedule["learning_rate_end"]
        warm_up_epochs = schedule["warm_up_epochs"]
        num_epochs = config.training["active"]["num_epochs"]
        warm_up_steps = warm_up_epochs * training_steps_per_epoch
        total_steps = num_epochs * training_steps_per_epoch

        if steps_taken < warm_up_steps:
            cur_lr = (steps_taken / warm_up_steps) * lr_init
        else:
            cur_lr = lr_end + 0.5 * (lr_init - lr_end) * (
                  (1 + np.cos(((steps_taken - warm_up_steps) / (total_steps - warm_up_steps)) * np.pi)))

    #elif schedule_type == "piecewise_constant_decay"
    else:
        raise RuntimeError("Unknown schedule type: '{}'.".format(schedule_type))

    return cur_lr



def get_weight_names(model, input_shape):

    for layer in model:
        for weight in layer.weights:
            print(weight.name)




def output_excel(out_path, predictions, img_set, dataset_name):

    farm_name = img_set.farm_name
    field_name = img_set.field_name
    mission_date = img_set.mission_date

    dataset = img_set.datasets[dataset_name]

    d = {
        "farm_name": [],
        "field_name": [],
        "mission_date": [],
        "dataset_name": [],
        "image_id": [],
    }
    for class_name in img_set.class_map.keys(): #config.arch["class_map"].keys():
        d["annotated_" + class_name + "_count"] = []
        d["model_" + class_name + "_count"] = []

    for img in dataset.imgs:
        d["farm_name"].append(farm_name)
        d["field_name"].append(field_name)
        d["mission_date"].append(mission_date)
        d["dataset_name"].append(dataset_name)
        d["image_id"].append(img.img_name)

        if img.is_annotated:
            img_abs_boxes, img_classes = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map) #config.arch["class_map"])
            unique, counts = np.unique(img_classes, return_counts=True)
            class_num_to_count = dict(zip(unique, counts))
            cur_img_class_counts = {k: 0 for k in img_set.class_map.keys()} #config.arch["class_map"].keys()}
            for class_num in class_num_to_count.keys():
                cur_img_class_counts[img_set.reverse_class_map[class_num]] = class_num_to_count[class_num]
                #cur_img_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

        cur_img_pred_class_counts = predictions["image_predictions"][img.img_name]["pred_class_counts"]
        for class_name in img_set.class_map.keys(): #config.arch["class_map"].keys():
            if img.is_annotated:
                d["annotated_" + class_name + "_count"].append(cur_img_class_counts[class_name])
            else:
                d["annotated_" + class_name + "_count"].append(np.nan)
            d["model_" + class_name + "_count"].append(cur_img_pred_class_counts[class_name])

    
    pandas.io.formats.excel.ExcelFormatter.header_style = None
    df = pd.DataFrame(data=d)
    df.sort_values(by="image_id", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_id"])))
    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    #df.to_excel(writer, index=False, sheet_name="Sheet1")
    #for sheetname, df in dfs.items():  # loop through `dict` of dataframes
    df.to_excel(writer, index=False, sheet_name="Sheet1", na_rep='NA')  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save()



def update_loss_tracker_entry(loss_tracker, key, cur_loss, cur_epoch):

    loss_tracker[key]["values"].append(cur_loss)

    best = loss_tracker[key]["best"]["value"]
    if cur_loss < best:
        loss_tracker[key]["best"]["value"] = cur_loss
        loss_tracker[key]["best"]["epoch"] = cur_epoch
        loss_tracker[key]["epochs_since_improvement"] = 0
        return True
    else:
        loss_tracker[key]["epochs_since_improvement"] += 1
        return False






def stop_early(config, loss_tracker):
    if "min_num_epochs" in config.training["active"] and \
        len(loss_tracker["training_loss"]["values"]) < config.training["active"]["min_num_epochs"]:
        
        return False
    
    if config.training["active"]["early_stopping"]["apply"]:
        key = config.training["active"]["early_stopping"]["monitor"]
        if loss_tracker[key]["epochs_since_improvement"] >= config.training["active"]["early_stopping"]["num_epochs_tolerance"]:
            return True


    return False


def set_active_training_params(config, seq_num):

    config.training["active"] = {}

    for k in config.training["shared_default"]:
        config.training["active"][k] = config.training["shared_default"][k]

    for k in config.training["training_sequence"][seq_num]:
        config.training["active"][k] = config.training["training_sequence"][seq_num][k]


def set_active_inference_params(config, img_set_num):

    config.inference["active"] = {}

    for k in config.inference["shared_default"]:
        config.inference["active"][k] = config.inference["shared_default"][k]

    for k in config.inference["image_sets"][img_set_num]:
        config.inference["active"][k] = config.inference["image_sets"][img_set_num][k]






def get_img_detections(patch_abs_boxes, patch_scores, patch_classes, patch_coords, 
                      img_path, buffer_pct=None):

    if patch_abs_boxes.size == 0:
        img_abs_boxes = np.array([], dtype=np.int32)
        img_scores = np.array([], dtype=np.float32)
        img_classes = np.array([], dtype=np.int32)

    else:
        img_width, img_height = Img(img_path).get_wh()

        img_abs_boxes = (np.array(patch_abs_boxes) + \
                         np.tile(patch_coords[:2], 2)).astype(np.int32)

        if buffer_pct is not None:
            patch_wh = (patch_coords[2] - patch_coords[0])
            buffer_px = (buffer_pct / 100 ) * patch_wh

            mask = np.logical_and(
                    np.logical_and(
                     np.logical_or(img_abs_boxes[:, 0] > patch_coords[0] + buffer_px, img_abs_boxes[:, 0] <= buffer_px),
                     np.logical_or(img_abs_boxes[:, 1] > patch_coords[1] + buffer_px, img_abs_boxes[:, 1] <= buffer_px)),
                    np.logical_and(
                     np.logical_or(img_abs_boxes[:, 2] < patch_coords[2] - buffer_px, img_abs_boxes[:, 2] >= img_height - buffer_px),
                     np.logical_or(img_abs_boxes[:, 3] < patch_coords[3] - buffer_px, img_abs_boxes[:, 3] >= img_width - buffer_px))
                )

            img_abs_boxes = img_abs_boxes[mask]
            img_scores = patch_scores[mask]
            img_classes = patch_classes[mask]
        else:
            img_scores = patch_scores
            img_classes = patch_classes

        #print("img_abs_boxes", img_abs_boxes)
        #print("img_scores", img_scores)

    return img_abs_boxes, img_scores, img_classes





def clip_img_boxes(img_predictions):

    for img_name in img_predictions.keys():

        if len(img_predictions[img_name]["pred_img_abs_boxes"]) > 0:
            pred_img_abs_boxes = np.array(img_predictions[img_name]["pred_img_abs_boxes"])
            img_width, img_height = Img(img_predictions[img_name]["img_path"]).get_wh()
            pred_img_abs_boxes = box_utils.clip_boxes_np(pred_img_abs_boxes, [0, 0, img_height, img_width])
            img_predictions[img_name]["pred_img_abs_boxes"] = pred_img_abs_boxes.tolist()


def apply_nms_to_img_boxes(img_predictions, iou_thresh):

    for img_name in img_predictions.keys():
        if len(img_predictions[img_name]["pred_img_abs_boxes"]) > 0:
            pred_img_abs_boxes = np.array(img_predictions[img_name]["pred_img_abs_boxes"])
            pred_classes = np.array(img_predictions[img_name]["pred_classes"])
            pred_scores = np.array(img_predictions[img_name]["pred_scores"])

            nms_boxes, nms_classes, nms_scores = box_utils.non_max_suppression_with_classes(
                                                    pred_img_abs_boxes,
                                                    pred_classes,
                                                    pred_scores,
                                                    iou_thresh=iou_thresh)
        else:
            nms_boxes = np.array([])
            nms_classes = np.array([])
            nms_scores = np.array([])

        img_predictions[img_name]["pred_img_abs_boxes"] = nms_boxes.tolist()
        img_predictions[img_name]["pred_classes"] = nms_classes.tolist()
        img_predictions[img_name]["pred_scores"] = nms_scores.tolist()


def add_class_detections(img_predictions, img_set):
    for img_name in img_predictions.keys():
        pred_boxes = np.array(img_predictions[img_name]["pred_img_abs_boxes"])
        pred_classes = np.array(img_predictions[img_name]["pred_classes"])
        pred_scores = np.array(img_predictions[img_name]["pred_scores"])
        unique, counts = np.unique(pred_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        #pred_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
        pred_class_counts = {k: 0 for k in img_set.class_map.keys()}
        pred_class_boxes = {k: [] for k in img_set.class_map.keys()}
        pred_class_scores = {k: [] for k in img_set.class_map.keys()}
        for class_num in class_num_to_count.keys():
            class_name = img_set.reverse_class_map[class_num]
            #class_name = config.arch["reverse_class_map"][class_num]
            pred_class_counts[class_name] = int(class_num_to_count[class_num])
            pred_class_boxes[class_name] = (pred_boxes[class_num == pred_classes]).tolist()
            pred_class_scores[class_name] = (pred_scores[class_num == pred_classes]).tolist()


        img_predictions[img_name]["pred_class_counts"] = pred_class_counts
        img_predictions[img_name]["pred_class_boxes"] = pred_class_boxes
        img_predictions[img_name]["pred_class_scores"] = pred_class_scores



def create_predictions_skeleton(img_set, dataset):

    return {"farm_name": img_set.farm_name,
            "field_name": img_set.field_name,
            "mission_date": img_set.mission_date,
            "dataset_name": dataset.name,
            "image_predictions": {}, 
            "patch_predictions": {},
            "metrics": {"point": {},
                        "boxplot": {}}
            }