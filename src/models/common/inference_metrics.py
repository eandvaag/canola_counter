import logging
import tqdm
import os
import numpy as np
from mean_average_precision import MetricBuilder


import models.common.box_utils as box_utils

from io_utils import json_io, w3c_io



def DiC(actual, pred):
    return actual - pred

def abs_DiC(actual, pred):
    return np.abs(actual - pred)

def pct_DiC(actual, pred):
    return np.divide(abs_DiC(actual, pred), actual, out=np.zeros_like(actual, dtype=np.float64), where=actual!=0) * 100

def nonzero_abs_DiC(actual, pred):
    mask = actual != 0
    return abs_DiC(actual[mask], pred[mask])

def squared_DiC(actual, pred):
    return (actual - pred) ** 2

def r_squared(actual, pred):
    SS_res = np.sum((actual - pred) ** 2)
    SS_tot = np.sum((actual - np.mean(actual)) ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = 1 - (SS_res / SS_tot)
    if np.isnan(res):
        res = 0.0 
    return float(res)



def boxplot_data(data):

    data = np.array(data)
    if data.size == 0:
        res = {
            "range_min": 0,
            "whisker_min": 0,
            "q1": 0,
            "q2": 0,
            "q3": 0,
            "whisker_max": 0,
            "range_max": 0,
            "outliers": []
        }
    else:
        q1, q2, q3 = np.percentile(data, [25, 50, 75], interpolation="midpoint")
        iqr = q3 - q1

        res = {
            "range_min": float(np.min(data)),
            "whisker_min": q1 - 1.5 * iqr,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "whisker_max": q3 + 1.5 * iqr,
            "range_max": float(np.max(data))
        }

        res["outliers"] = data[np.logical_or(data < res["whisker_min"],
                                             data > res["whisker_max"])].tolist()

    return res




# def five_num_summary(data):
#     return np.percentile(np.array(data), [0, 25, 50, 75, 100], interpolation='midpoint').tolist()

# def five_num_summary_DiC(actual, pred):
#     return five_num_summary(actual - pred)

# def DiC_occurrences(actual, pred):
#     a = (actual - pred)
#     vals = np.arange(a.min(), a.max() + 1)
#     occurrences, _ = np.histogram(a, bins=(a.max() - a.min() + 1))
#     return {
#             "vals": vals.tolist(),
#             "occurrences": occurrences.tolist()
#            }


def collect_statistics(predictions, dataset, config, inference_times=None):


    # if "metrics" not in predictions:
    #     predictions["metrics"] = {}

    # if "point" not in predictions["metrics"]:
    #     predictions["metrics"]["point"] = {}

    # if "boxplot" not in predictions["metrics"]:
    #     predictions["metrics"]["boxplot"] = {}

    # datasets = {
    #     "training": image_set.training_dataset,
    #     "validation": image_set.validation_dataset,
    #     "test", image_set.test_dataset
    # }
    #dataset = image_set.all_dataset

    point_metrics = predictions["metrics"]["all"]["point"]
    boxplot_metrics = predictions["metrics"]["all"]["boxplot"]

    if inference_times is not None:
        total_inference_time = float(np.sum(inference_times))
        point_metrics["Total Inference Time (s)"] = {}
        point_metrics["Total Inference Time (s)"]["---"] = total_inference_time
        point_metrics["Per Patch Inference Time (s)"] = {}
        point_metrics["Per Patch Inference Time (s)"]["---"] = total_inference_time / len(inference_times)
        point_metrics["Per Image Inference Time (s)"] = {}
        point_metrics["Per Image Inference Time (s)"]["---"] = total_inference_time / len(predictions["image_predictions"])

    #boxplot_metrics["Inference Times (Per Patch)"] = {}
    #boxplot_metrics["Inference Times (Per Patch)"]["---"] = boxplot_data(inference_times)

    # point_metrics["Mean Confidence Score"] = {}
    # point_metrics["Max Confidence Score"] = {}
    # point_metrics["Max Box Area"] = {}
    # point_metrics["Min Box Area"] = {}
    # point_metrics["Mean Box Area"] = {}

    boxplot_metrics["Confidence"] = {}
    boxplot_metrics["Box Area"] = {}

    confidences = {k: [] for k in config.arch["class_map"].keys()}
    boxes = {k: [] for k in config.arch["class_map"].keys()}
    for image in dataset.images:
        for cls_name in config.arch["class_map"].keys():
            cls_confs = predictions["image_predictions"][image.image_name]["pred_class_scores"][cls_name] 
            confidences[cls_name].extend(cls_confs)

            cls_boxes = predictions["image_predictions"][image.image_name]["pred_class_boxes"][cls_name]
            boxes[cls_name].extend(cls_boxes)

    num_predictions = np.sum(len(confidences[k]) for k in confidences.keys())

    # point_metrics["Mean Confidence Score"]["Cross-Class Weighted Average"] = 0
    # point_metrics["Max Confidence Score"]["Cross-Class Weighted Average"] = 0
    # point_metrics["Mean Box Area"]["Cross-Class Weighted Average"] = 0
    # point_metrics["Min Box Area"]["Cross-Class Weighted Average"] = 0
    # point_metrics["Max Box Area"]["Cross-Class Weighted Average"] = 0

    for cls_name in config.arch["class_map"].keys():

        if len(confidences[cls_name]) > 0:
            mean_cls_conf = float(np.mean(confidences[cls_name]))
            max_cls_conf = float(np.max(confidences[cls_name]))
            box_areas = box_utils.box_areas_np(np.array(boxes[cls_name]))
            mean_box_area = float(np.mean(box_areas))
            min_box_area = float(np.min(box_areas))
            max_box_area = float(np.max(box_areas))

        else:
            mean_cls_conf = 0
            max_cls_conf = 0
            box_areas = np.array([])
            mean_box_area = 0
            min_box_area = 0
            max_box_area = 0


        boxplot_metrics["Confidence"][cls_name] = boxplot_data(confidences[cls_name])
        boxplot_metrics["Box Area"][cls_name] = boxplot_data(box_areas)

        # point_metrics["Mean Confidence Score"][cls_name] = mean_cls_conf
        # point_metrics["Max Confidence Score"][cls_name] = max_cls_conf
        # point_metrics["Mean Box Area"][cls_name] = mean_box_area
        # point_metrics["Max Box Area"][cls_name] = max_box_area
        # point_metrics["Min Box Area"][cls_name] = min_box_area
        # if num_predictions > 0:
        #     point_metrics["Mean Confidence Score"]["Cross-Class Weighted Average"] += \
        #         (len(confidences[cls_name]) / num_predictions) * mean_cls_conf
        #     point_metrics["Max Confidence Score"]["Cross-Class Weighted Average"] += \
        #         (len(confidences[cls_name]) / num_predictions) * max_cls_conf

        #     point_metrics["Mean Box Area"]["Cross-Class Weighted Average"] += \
        #         (len(confidences[cls_name]) / num_predictions) * mean_box_area
        #     point_metrics["Min Box Area"]["Cross-Class Weighted Average"] += \
        #         (len(confidences[cls_name]) / num_predictions) * min_box_area
        #     point_metrics["Max Box Area"]["Cross-Class Weighted Average"] += \
        #         (len(confidences[cls_name]) / num_predictions) * max_box_area



def collect_metrics(predictions, dataset, config,
                    collect_patch_metrics=True, calculate_mAP=True):

    logger = logging.getLogger(__name__)

    # datasets = {}
    # if len(image_set.training_dataset.images) > 0:
    #     datasets["training"] = image_set.training_dataset
    # if len(image_set.validation_dataset.images) > 0:
    #     datasets["validation"] = image_set.validation_dataset    
    # if len(image_set.test_dataset.images) > 0:
    #     datasets["test"] = image_set.test_dataset

    # datasets["all"] = image_set.all_dataset
    # # datasets = {
    # #     "training": image_set.training_dataset,
    # #     "validation": image_set.validation_dataset,
    # #     "test": image_set.test_dataset
    # # }

    # datasets["all"] = DataSet({

    # })

    num_classes = len(config.arch["class_map"].keys())

    annotated_image_counts = {}
    pred_image_counts = {}

    class_map = config.arch["class_map"]
    reverse_class_map = {v: k for k, v in config.arch["class_map"].items()}

    image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes) #config.arch["num_classes"])

    annotations = w3c_io.load_annotations(dataset.annotations_path, config.arch["class_map"])

    #for (dataset_name, dataset) in datasets.items():


    annotated_image_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
    pred_image_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}

    for image in dataset.completed_images:

        #img_abs_boxes, img_classes = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map) #config.arch["class_map"])
        image_abs_boxes = annotations[image.image_name]["boxes"]
        image_classes = annotations[image.image_name]["classes"]
        
        
        unique, counts = np.unique(image_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        cur_image_class_counts = {k: 0 for k in class_map} #config.arch["class_map"].keys()}
        for class_num in class_num_to_count.keys():
            cur_image_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
            #cur_img_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

        cur_image_pred_class_counts = predictions["image_predictions"][image.image_name]["pred_class_counts"]

        for class_name in class_map.keys(): #config.arch["class_map"].keys():
            annotated_image_counts[class_name].append(cur_image_class_counts[class_name])
            pred_image_counts[class_name].append(cur_image_pred_class_counts[class_name])

        #annotated_img_count = np.shape(img_abs_boxes)[0]

        #pred_img_count = predictions["image_predictions"][img.img_name]["pred_count"]
        
        if calculate_mAP:
            pred_abs_boxes = np.array(predictions["image_predictions"][image.image_name]["pred_image_abs_boxes"])
            pred_classes = np.array(predictions["image_predictions"][image.image_name]["pred_classes"])
            pred_scores = np.array(predictions["image_predictions"][image.image_name]["pred_scores"])

            pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                                                                image_abs_boxes, image_classes)
            image_metric_fn.add(pred_for_mAP, true_for_mAP)

            #annotated_img_counts.append(annotated_img_count)
        #pred_img_counts.append(pred_img_count)


    #annotated_img_counts = np.array(annotated_img_counts)
    #pred_img_counts = np.array(pred_img_counts)

    # annotated_patch_counts = []
    # pred_patch_counts = []

    if collect_patch_metrics:
        annotated_patch_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
        pred_patch_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
        for patch_name, patch_pred in predictions["patch_predictions"].items():

            if "patch_classes" in patch_pred:
                patch_classes = patch_pred["patch_classes"]
                unique, counts = np.unique(patch_classes, return_counts=True)
                class_num_to_count = dict(zip(unique, counts))
                cur_patch_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
                for class_num in class_num_to_count.keys():
                    cur_patch_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
                    #cur_patch_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

                pred_patch_classes = patch_pred["pred_classes"]
                pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
                class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
                cur_patch_pred_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
                for class_num in class_num_to_pred_count.keys():
                    cur_patch_pred_class_counts[reverse_class_map[class_num]] = class_num_to_pred_count[class_num]
                    #cur_patch_pred_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_pred_count[class_num]

                for class_name in class_map.keys(): #config.arch["class_map"].keys():
                    annotated_patch_counts[class_name].append(cur_patch_class_counts[class_name])
                    pred_patch_counts[class_name].append(cur_patch_pred_class_counts[class_name])


    print("annotated_image_counts", annotated_image_counts)
    total_obj_count = float(np.sum([np.sum(np.array(annotated_image_counts[class_name])) for class_name in annotated_image_counts.keys()]))

    logger.info("Started calculating count metrics.")

    # if "metrics" not in predictions:
    #     predictions["metrics"] = {}

    # if "point" not in predictions["metrics"]:
    #     predictions["metrics"]["point"] = {}

    # if "boxplot" not in predictions["metrics"]:
    #     predictions["metrics"]["boxplot"] = {}


    point_metrics = predictions["metrics"]["all"]["point"]
    boxplot_metrics = predictions["metrics"]["all"]["boxplot"]



    point_metrics["Image Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
    point_metrics["Image Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
    point_metrics["Image R Squared"] = {"Cross-Class Weighted Average": 0}
    point_metrics["Image Non-Zero Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
    if collect_patch_metrics:
        point_metrics["Patch Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
        point_metrics["Patch Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
        point_metrics["Patch R Squared"] = {"Cross-Class Weighted Average": 0}
        point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}


    boxplot_metrics["Difference in Count (Image)"] = {}
    boxplot_metrics["Absolute Difference in Count (Image)"] = {} #{"Cross-Class Weighted Average": 0}
    boxplot_metrics["Percent Difference in Count (Image)"] = {} #{"Cross-Class Weighted Average": 0}
    #boxplot_metrics["Diff. in Count (Patch)"] = {}
    #boxplot_metrics["Absolute Diff. in Count (Patch)"] = {} #{"Cross-Class Weighted Average": 0}
    #boxplot_metrics["Percent Diff. in Count (Patch)"] = {} #{"Cross-Class Weighted Average": 0}
    
    # boxplot_metrics["Confidence"] = {"Cross-Class Weighted Average": 0}
    # boxplot_metrics["Box Area"] = {"Cross-Class Weighted Average": 0}
    # boxplot_metrics["Inference Time (Per Image)"] = {"Cross-Class Weighted Average": 0}
    # boxplot_metrics["Inference Time (Per Patch)"] = {"Cross-Class Weighted Average": 0}
    
    #predictions["metrics"]["Patch Five Num. Summary Diff. in Count"] = {}
    #predictions["metrics"]["Diff. in Count Occurrences"] = {}

    for class_name in annotated_image_counts.keys():

        image_class_annotated_count = np.array(annotated_image_counts[class_name])
        image_class_pred_count = np.array(pred_image_counts[class_name])

        point_metrics["Image Mean Abs. Diff. in Count"][class_name] = \
                        float(np.mean(abs_DiC(image_class_annotated_count, image_class_pred_count)))
        point_metrics["Image Mean Sq. Diff. in Count"][class_name] = \
                        float(np.mean(squared_DiC(image_class_annotated_count, image_class_pred_count)))
        point_metrics["Image R Squared"][class_name] = \
                        r_squared(image_class_annotated_count, image_class_pred_count)
        point_metrics["Image Non-Zero Mean Abs. Diff. in Count"][class_name] = \
                        float(np.mean(nonzero_abs_DiC(image_class_annotated_count, image_class_pred_count)))


        



        #predictions["metrics"]["Patch Five Num. Summary Diff. in Count"][class_name] = five_num_summary_DiC(patch_class_annotated_count, patch_class_pred_count)
        #predictions["metrics"]["Diff. in Count Occurrences"][class_name] = DiC_occurrences(patch_class_annotated_count, patch_class_pred_count)
        
        boxplot_metrics["Difference in Count (Image)"][class_name] = \
                    boxplot_data(DiC(image_class_annotated_count, image_class_pred_count))

        boxplot_metrics["Absolute Difference in Count (Image)"][class_name] = \
                    boxplot_data(abs_DiC(image_class_annotated_count, image_class_pred_count))

        boxplot_metrics["Percent Difference in Count (Image)"][class_name] = \
                    boxplot_data(pct_DiC(image_class_annotated_count, image_class_pred_count))

        # boxplot_metrics["Diff. in Count (Patch)"][class_name] = \
        #             boxplot_data(DiC(patch_class_annotated_count, patch_class_pred_count))

        # boxplot_metrics["Absolute Diff. in Count (Patch)"][class_name] = \
        #             boxplot_data(abs_DiC(patch_class_annotated_count, patch_class_pred_count))

        # boxplot_metrics["Percent Diff. in Count (Patch)"][class_name] = \
        #             boxplot_data(pct_DiC(patch_class_annotated_count, patch_class_pred_count))



        
        total_class_annotated_count = float(np.sum(image_class_annotated_count))

        point_metrics["Image Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
            (total_class_annotated_count / total_obj_count) * point_metrics["Image Mean Abs. Diff. in Count"][class_name]
        point_metrics["Image Mean Sq. Diff. in Count"]["Cross-Class Weighted Average"] += \
            (total_class_annotated_count / total_obj_count) * point_metrics["Image Mean Sq. Diff. in Count"][class_name]
        point_metrics["Image R Squared"]["Cross-Class Weighted Average"] += \
            (total_class_annotated_count / total_obj_count) * point_metrics["Image R Squared"][class_name]
        point_metrics["Image Non-Zero Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
            (total_class_annotated_count / total_obj_count) * point_metrics["Image Non-Zero Mean Abs. Diff. in Count"][class_name]



        if collect_patch_metrics:

            patch_class_annotated_count = np.array(annotated_patch_counts[class_name])
            patch_class_pred_count = np.array(pred_patch_counts[class_name])

            point_metrics["Patch Mean Abs. Diff. in Count"][class_name] = \
                            float(np.mean(abs_DiC(patch_class_annotated_count, patch_class_pred_count)))
            point_metrics["Patch Mean Sq. Diff. in Count"][class_name] = \
                            float(np.mean(squared_DiC(patch_class_annotated_count, patch_class_pred_count)))
            point_metrics["Patch R Squared"][class_name] = \
                            r_squared(patch_class_annotated_count, patch_class_pred_count)
            point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"][class_name] = \
                            float(np.mean(nonzero_abs_DiC(patch_class_annotated_count, patch_class_pred_count)))

            point_metrics["Patch Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
                (total_class_annotated_count / total_obj_count) * point_metrics["Patch Mean Abs. Diff. in Count"][class_name]
            point_metrics["Patch Mean Sq. Diff. in Count"]["Cross-Class Weighted Average"] += \
                (total_class_annotated_count / total_obj_count) * point_metrics["Patch Mean Sq. Diff. in Count"][class_name]
            point_metrics["Patch R Squared"]["Cross-Class Weighted Average"] += \
                (total_class_annotated_count / total_obj_count) * point_metrics["Patch R Squared"][class_name]
            point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
                (total_class_annotated_count / total_obj_count) * point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"][class_name]




    logger.info("Finished calculating count metrics.")

    if calculate_mAP:
        logger.info("Started calculating mAP scores.")

        pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
        coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        logger.info("Finished calculating mAP scores.")

        point_metrics["Image PASCAL VOC mAP"] = {}
        point_metrics["Image PASCAL VOC mAP"]["---"] = float(pascal_voc_mAP)
        point_metrics["Image MS COCO mAP"] = {}
        point_metrics["Image MS COCO mAP"]["---"] = float(coco_mAP)




def get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                              true_abs_boxes, true_classes):

    if pred_abs_boxes.size > 0:
        pred_abs_boxes = box_utils.swap_xy_np(pred_abs_boxes)
    else:
        pred_abs_boxes = np.expand_dims(pred_abs_boxes, axis=-1)
    pred_classes = np.expand_dims(pred_classes, axis=-1)
    pred_scores = np.expand_dims(pred_scores, axis=-1)
    pred = np.hstack([pred_abs_boxes, pred_classes, pred_scores])

    if true_abs_boxes.size > 0:
        true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    else:
        true_abs_boxes = np.expand_dims(true_abs_boxes, axis=-1)
    #true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    true_classes = np.expand_dims(true_classes, axis=-1)
    difficult = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    crowd = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    true = np.hstack([true_abs_boxes, true_classes, difficult, crowd])  

    return pred, true