import os
import logging
import pandas as pd
import pandas.io.formats.excel
from natsort import index_natsorted
import numpy as np

from io_utils import w3c_io
#import extract_patches as ep
import extract_patches as ep
from image_set import Image
from models.common import box_utils



# def create_patches(extraction_params, image_set, dataset_name):
#     dataset = image_set.datasets[dataset_name]

#     scenario_id = ep.extract_patches(extraction_params,
#                                      image_set,
#                                      dataset,
#                                      annotate_patches=dataset.is_annotated)

#     patch_dir = os.path.join(image_set.patch_dir, scenario_id)
#     return patch_dir, dataset.is_annotated



# def extract_patches(dataset, config):
#     patch_dir = ep.extract_patches(dataset, config)
#     return patch_dir

# def constant_learning_rate_function(steps_taken, training_steps_per_epoch, config):
#     return config.training["active"]["learning_rate_schedule"]["learning_rate"]

# def get_learning_rate_function(config):

#     schedule = config.training["active"]["learning_rate_schedule"]
#     schedule_type = schedule["schedule_type"]

#     if schedule_type == "constant"
#         f = constant_learning_rate_function




def get_learning_rate(steps_taken, training_steps_per_epoch, config):

    schedule = config["training"]["active"]["learning_rate_schedule"]
    schedule_type = schedule["schedule_type"]


    if schedule_type == "constant":
        cur_lr = schedule["learning_rate"]

    elif schedule_type == "cosine_annealing":
        lr_init = schedule["learning_rate_init"]
        lr_end = schedule["learning_rate_end"]
        warm_up_epochs = schedule["warm_up_epochs"]
        num_epochs = config["training"]["active"]["num_epochs"]
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




def output_excel(out_path, predictions, dataset, config):

    class_map = config["arch"]["class_map"]
    reverse_class_map = {v: k for k, v in class_map.items()}

    farm_name = dataset.farm_name
    field_name = dataset.field_name
    mission_date = dataset.mission_date

    d = {
        "farm_name": [],
        "field_name": [],
        "mission_date": [],
        "used_for": [],
        "image_name": []
        #"COCO_mAP": [],
        #"PASCAL_VOC_mAP": []
    }
    for class_name in class_map.keys(): #config.arch["class_map"].keys():
        d["annotated_" + class_name + "_count"] = []
        d["model_" + class_name + "_count"] = []
        #d["model_" + class_name + "_count_at_optimal_score"] = []

    annotations = w3c_io.load_annotations(dataset.annotations_path, class_map)
    completed_image_names = w3c_io.get_completed_images(annotations)

    test_reserved_images = annotations.keys()
    for image_set_config in config["training"]["image_sets"]:
        if image_set_config["farm_name"] == farm_name and \
            image_set_config["field_name"] == field_name and \
            image_set_config["mission_date"] == mission_date:
            test_reserved_images = image_set_config["test_reserved_images"]
            break


    #for image in dataset.images: #image_set.all_dataset.images:
    for image_name in predictions["image_predictions"].keys():
        # if image.image_name in image_set.training_dataset.image_names:
        #     dataset_name = "training"
        # elif image.image_name in image_set.validation_dataset.image_names:
        #     dataset_name = "validation"
        # elif image.image_name in image_set.test_dataset.image_names:
        #     dataset_name = "test"
        # else:
        #     dataset_name = "NA"
        if image_name in completed_image_names:
            if image_name in test_reserved_images:
                dataset_name = "testing"
            else:
                dataset_name = "training/validation"
        else:
            dataset_name = "NA"

        d["farm_name"].append(farm_name)
        d["field_name"].append(field_name)
        d["mission_date"].append(mission_date)
        d["used_for"].append(dataset_name)
        d["image_name"].append(image_name)

        #if image.is_annotated:
        if annotations[image_name]["status"] == "completed":
            #image_abs_boxes, image_classes = xml_io.load_boxes_and_classes(image.xml_path, image_set.class_map) #config.arch["class_map"])
            image_abs_boxes = annotations[image_name]["boxes"]
            image_classes = annotations[image_name]["classes"]
            unique, counts = np.unique(image_classes, return_counts=True)
            class_num_to_count = dict(zip(unique, counts))
            cur_image_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
            for class_num in class_num_to_count.keys():
                cur_image_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
                #cur_image_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

        cur_image_pred_class_counts = predictions["image_predictions"][image_name]["pred_class_counts"]
        #cur_image_pred_opt_class_counts = predictions["image_predictions"][image_name]["pred_opt_class_counts"]
        for class_name in class_map.keys(): #config.arch["class_map"].keys():
            if annotations[image_name]["status"] == "completed":
                d["annotated_" + class_name + "_count"].append(cur_image_class_counts[class_name])
            else:
                d["annotated_" + class_name + "_count"].append(np.nan)
            d["model_" + class_name + "_count"].append(cur_image_pred_class_counts[class_name])
            #d["model_" + class_name + "_count_at_optimal_score"].append(cur_image_pred_opt_class_counts[class_name])

    
    pandas.io.formats.excel.ExcelFormatter.header_style = None
    df = pd.DataFrame(data=d)
    df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))
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
    if "min_num_epochs" in config["training"]["active"] and \
        len(loss_tracker["training_loss"]["values"]) < config["training"]["active"]["min_num_epochs"]:
        
        return False
    
    if config["training"]["active"]["early_stopping"]["apply"]:
        key = config["training"]["active"]["early_stopping"]["monitor"]
        if loss_tracker[key]["epochs_since_improvement"] >= config["training"]["active"]["early_stopping"]["num_epochs_tolerance"]:
            return True


    return False


def set_active_training_params(config, seq_num):

    config["training"]["active"] = {}
    for k in config["training"]["training_sequence"][seq_num]:
        config["training"]["active"][k] = config["training"]["training_sequence"][seq_num][k]

# def set_active_training_params(config, seq_num):

#     config.training["active"] = {}

#     for k in config.training["shared_default"]:
#         config.training["active"][k] = config.training["shared_default"][k]

#     for k in config.training["training_sequence"][seq_num]:
#         config.training["active"][k] = config.training["training_sequence"][seq_num][k]


# def set_active_inference_params(config):
#     config.inference["active"] = {}
#     for k in config.inference:
#         config.inference["active"][k] = config.inference[k]

# def set_active_inference_params(config, image_set_num):

#     config.inference["active"] = {}

#     for k in config.inference["shared_default"]:
#         config.inference["active"][k] = config.inference["shared_default"][k]

#     for k in config.inference["image_sets"][image_set_num]:
#         config.inference["active"][k] = config.inference["image_sets"][image_set_num][k]






def get_image_detections(patch_abs_boxes, patch_scores, patch_classes, patch_coords, 
                      image_path, trim=True): #, patch_border_buffer_percent=None): # buffer_pct=None):

    if patch_abs_boxes.size == 0:
        image_abs_boxes = np.array([], dtype=np.int32)
        image_scores = np.array([], dtype=np.float32)
        image_classes = np.array([], dtype=np.int32)

    else:
        patch_height = patch_coords[2] - patch_coords[0]
        patch_width = patch_coords[3] - patch_coords[1]

        image_width, image_height = Image(image_path).get_wh()


        

        image_abs_boxes = (np.array(patch_abs_boxes) + \
                           np.tile(patch_coords[:2], 2)).astype(np.int32)

        # if patch_border_buffer_percent is not None:
        #     patch_wh = (patch_coords[2] - patch_coords[0])
        #     buffer_px = (patch_border_buffer_percent / 100 ) * patch_wh

        #     mask = np.logical_and(
        #             np.logical_and(
        #              np.logical_or(image_abs_boxes[:, 0] > patch_coords[0] + buffer_px, image_abs_boxes[:, 0] <= buffer_px),
        #              np.logical_or(image_abs_boxes[:, 1] > patch_coords[1] + buffer_px, image_abs_boxes[:, 1] <= buffer_px)),
        #             np.logical_and(
        #              np.logical_or(image_abs_boxes[:, 2] < patch_coords[2] - buffer_px, image_abs_boxes[:, 2] >= image_height - buffer_px),
        #              np.logical_or(image_abs_boxes[:, 3] < patch_coords[3] - buffer_px, image_abs_boxes[:, 3] >= image_width - buffer_px))
        #         )

        #     image_abs_boxes = image_abs_boxes[mask]
        #     image_scores = patch_scores[mask]
        #     image_classes = patch_classes[mask]
        


        if trim:

            accept_bottom = 0 if patch_coords[0] == 0 else patch_coords[0] + round(patch_height / 4)
            accept_left = 0 if patch_coords[1] == 0 else patch_coords[1] + round(patch_width / 4)
            accept_top = image_height if patch_coords[2] == image_height else patch_coords[2] - round(patch_height / 4)
            accept_right = image_width if patch_coords[3] == image_width else patch_coords[3] - round(patch_width / 4)


            box_centres = (image_abs_boxes[..., :2] + image_abs_boxes[..., 2:]) / 2.0

            # print("box_centres", box_centres)
            # print("accept_bottom: {}, accept_left: {}, accept_top: {}, accept_right: {}".format(
            #     accept_bottom, accept_left, accept_top, accept_right
            # ))
            mask = np.logical_and(
                np.logical_and(box_centres[:,0] >= accept_bottom, box_centres[:,0] < accept_top),
                np.logical_and(box_centres[:,1] >= accept_left, box_centres[:,1] < accept_right)
            )

            # print("mask", mask)

            image_abs_boxes = image_abs_boxes[mask]
            image_scores = patch_scores[mask]
            image_classes = patch_classes[mask]
        else:
            image_scores = patch_scores
            image_classes = patch_classes





        

        # image_abs_boxes = (np.array(patch_abs_boxes) + \
        #                  np.tile(patch_coords[:2], 2)).astype(np.int32)

        # if buffer_pct is not None:
        #     patch_wh = (patch_coords[2] - patch_coords[0])
        #     buffer_px = (buffer_pct / 100 ) * patch_wh

        #     mask = np.logical_and(
        #             np.logical_and(
        #              np.logical_or(image_abs_boxes[:, 0] > patch_coords[0] + buffer_px, image_abs_boxes[:, 0] <= buffer_px),
        #              np.logical_or(image_abs_boxes[:, 1] > patch_coords[1] + buffer_px, image_abs_boxes[:, 1] <= buffer_px)),
        #             np.logical_and(
        #              np.logical_or(image_abs_boxes[:, 2] < patch_coords[2] - buffer_px, image_abs_boxes[:, 2] >= image_height - buffer_px),
        #              np.logical_or(image_abs_boxes[:, 3] < patch_coords[3] - buffer_px, image_abs_boxes[:, 3] >= image_width - buffer_px))
        #         )

        #     image_abs_boxes = image_abs_boxes[mask]
        #     image_scores = patch_scores[mask]
        #     image_classes = patch_classes[mask]
        # else:
        #     image_scores = patch_scores
        #     image_classes = patch_classes

        #print("image_abs_boxes", image_abs_boxes)
        #print("image_scores", image_scores)

    return image_abs_boxes, image_scores, image_classes





def clip_image_boxes(image_predictions):
    for image_name in image_predictions.keys():
        for transform_type in image_predictions[image_name].keys():
            if len(image_predictions[image_name][transform_type]["pred_image_abs_boxes"]) > 0:
                pred_image_abs_boxes = np.array(image_predictions[image_name][transform_type]["pred_image_abs_boxes"])
                image_width, image_height = Image(image_predictions[image_name][transform_type]["image_path"]).get_wh()
                pred_image_abs_boxes = box_utils.clip_boxes_np(pred_image_abs_boxes, [0, 0, image_height, image_width])
                image_predictions[image_name][transform_type]["pred_image_abs_boxes"] = pred_image_abs_boxes.tolist()


            # if len(image_predictions[image_name]["nt_pred_image_abs_boxes"]) > 0:
            #     pred_image_abs_boxes = np.array(image_predictions[image_name]["nt_pred_image_abs_boxes"])
            #     image_width, image_height = Image(image_predictions[image_name]["image_path"]).get_wh()
            #     pred_image_abs_boxes = box_utils.clip_boxes_np(pred_image_abs_boxes, [0, 0, image_height, image_width])
            #     image_predictions[image_name]["nt_pred_image_abs_boxes"] = pred_image_abs_boxes.tolist()




def apply_nms_to_image_boxes(image_predictions, iou_thresh):

        for image_name in image_predictions.keys():
            for transform_type in image_predictions[image_name].keys():
                if len(image_predictions[image_name][transform_type]["pred_image_abs_boxes"]) > 0:
                    pred_image_abs_boxes = np.array(image_predictions[image_name][transform_type]["pred_image_abs_boxes"])
                    pred_classes = np.array(image_predictions[image_name][transform_type]["pred_classes"])
                    pred_scores = np.array(image_predictions[image_name][transform_type]["pred_scores"])

                    nms_boxes, nms_classes, nms_scores = box_utils.non_max_suppression_with_classes(
                                                            pred_image_abs_boxes,
                                                            pred_classes,
                                                            pred_scores,
                                                            iou_thresh=iou_thresh)
                else:
                    nms_boxes = np.array([])
                    nms_classes = np.array([])
                    nms_scores = np.array([])

                image_predictions[image_name][transform_type]["pred_image_abs_boxes"] = nms_boxes.tolist()
                image_predictions[image_name][transform_type]["pred_classes"] = nms_classes.tolist()
                image_predictions[image_name][transform_type]["pred_scores"] = nms_scores.tolist()


def add_class_detections(image_predictions, config): #, opt_thresh_val):

    class_map = config["arch"]["class_map"]
    reverse_class_map = {v: k for k, v in class_map.items()}

    for image_name in image_predictions.keys():
        pred_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
        pred_classes = np.array(image_predictions[image_name]["pred_classes"])
        pred_scores = np.array(image_predictions[image_name]["pred_scores"])
        unique = np.unique(pred_classes)

        pred_class_counts = {k: 0 for k in class_map.keys()}
        pred_class_boxes = {k: [] for k in class_map.keys()}
        pred_class_scores = {k: [] for k in class_map.keys()}
        #pred_opt_class_counts = {k: 0 for k in class_map.keys()}

        for class_num in unique:
            class_name = reverse_class_map[class_num]
            inds = np.where(pred_classes == class_num)[0]

            pred_class_counts[class_name] = int(pred_scores[inds].size)
            pred_class_boxes[class_name] = (pred_boxes[inds]).tolist()
            pred_class_scores[class_name] = (pred_scores[inds]).tolist()
            #pred_opt_class_counts[class_name] = int((np.where(pred_scores[inds] >= opt_thresh_val)[0]).size)

        
        # class_num_to_count = dict(zip(unique, counts))
        # #pred_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
        # pred_class_counts = {k: 0 for k in class_map.keys()}
        # pred_class_boxes = {k: [] for k in class_map.keys()}
        # pred_class_scores = {k: [] for k in class_map.keys()}
        # for class_num in class_num_to_count.keys():
        #     class_name = reverse_class_map[class_num]
        #     #class_name = config.arch["reverse_class_map"][class_num]
        #     pred_class_counts[class_name] = int(class_num_to_count[class_num])
        #     pred_class_boxes[class_name] = (pred_boxes[class_num == pred_classes]).tolist()
        #     pred_class_scores[class_name] = (pred_scores[class_num == pred_classes]).tolist()


        image_predictions[image_name]["pred_class_counts"] = pred_class_counts
        image_predictions[image_name]["pred_class_boxes"] = pred_class_boxes
        image_predictions[image_name]["pred_class_scores"] = pred_class_scores
        #image_predictions[image_name]["pred_opt_class_counts"] = pred_opt_class_counts



# def create_metrics_skeleton(dataset):
#     metrics = {
#         "point": {},
#         "boxplot": {},
#         "image": {}
#     }
#     for image in dataset.images:
#         metrics["image"][image.image_name] = {}
#     return metrics


# def create_predictions_skeleton(dataset):

#     return {"farm_name": dataset.farm_name, #config["target_farm_name"],
#             "field_name": dataset.field_name, #config["target_field_name"],
#             "mission_date": dataset.mission_date, #config["target_mission_date"],
#             "image_predictions": {}, 
#             "patch_predictions": {}
#             # "metrics": 
#             #     {
#             #         # "training": 
#             #         # {
#             #         #     "point": {},
#             #         #     "boxplot": {}
#             #         # },
#             #         # "validation":
#             #         # {
#             #         #     "point": {},
#             #         #     "boxplot": {}
#             #         # },
#             #         # "test":
#             #         # {
#             #         #     "point": {},
#             #         #     "boxplot": {}
#             #         # },
#             #         "all":
#             #         {
#             #             "point": {},
#             #             "boxplot": {}
#             #         },                    
#             #     }
#             }