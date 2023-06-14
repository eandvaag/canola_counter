import os
import glob
import shutil
import time
import math as m
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
# from vendi_score import vendi
from scipy.stats import entropy
import pandas as pd
import random
import uuid
import urllib3
import logging
import matplotlib.pyplot as plt
from matplotlib import ticker
import tqdm
import cv2
from natsort import natsorted

import image_set_actions as isa
import server
from models.common import annotation_utils, box_utils, inference_metrics
from io_utils import json_io
import fine_tune_eval
from image_set import Image

from lock_queue import LockQueue
import diversity_test
import fine_tune_experiment
import geo_locations


my_plot_colors = ["orangered", "royalblue", "forestgreen", "orange", "mediumorchid"]
#["firebrick", "mediumblue", "darkorange", "darkmagenta", "darkgreen"]
#"salmon", "royalblue", "forestgreen", "orange", "mediumorchid"]




def quality_score_vs_accuracy():
    pass







def create_fine_tune_plot_averaged(baseline, test_set, methods, num_annotations_to_select_lst, num_dups):
    test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
    test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    

    mapping = get_mapping_for_test_set(test_set_image_set_dir)
    # annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
    # annotations = annotation_utils.load_annotations(annotations_path)
    results = {}
    # labels = []

    
    pre_fine_tune_result_name = baseline["model_name"] + "_pre_finetune" #+ str(num_images_to_select)
    result_uuid = mapping[pre_fine_tune_result_name]
    result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)

    predictions_path = os.path.join(result_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path)

    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    accuracies = []
    for image_name in annotations.keys():
        sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
        accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
        accuracies.append(accuracy)

    pre_fine_tune_accuracy = np.mean(accuracies)

    # label_lookup = {
    #     "selected_patches_match_both": "selected_patches",
    #     "random_images": "random_images"
    # }

    max_num_fine_tuning_boxes = 0
    for i, method in enumerate(methods):
        results[method] = []
        for j in range(len(num_annotations_to_select_lst)):

            dup_accuracies = []
            for dup_num in range(num_dups):

                result_name = baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select_lst[j]) + "_annotations_dup_" + str(dup_num)
                result_uuid = mapping[result_name]
                result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)

                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)

                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)


                num_fine_tuning_boxes = 0
                num_fine_tuning_regions = 0
                accuracies = []
                for image_name in annotations.keys():
                    sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                    accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
                    accuracies.append(accuracy)

                    num_fine_tuning_boxes += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                    num_fine_tuning_regions += len(annotations[image_name]["training_regions"])


                if num_fine_tuning_boxes > max_num_fine_tuning_boxes:
                    max_num_fine_tuning_boxes = num_fine_tuning_boxes
                # global_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, list(annotations.keys())) #assessment_images_lst)

                # test_set_accuracy = global_accuracy #np.mean(accuracy)
                test_set_accuracy = np.mean(accuracies)
                dup_accuracies.append(test_set_accuracy)
                # results[method].append((num_fine_tuning_boxes, test_set_accuracy))
                # results[method].append((num_fine_tuning_regions, test_set_accuracy))

            # results.append(np.mean(dup_accuracies))
            # labels.append(method)
            results[method].append((num_annotations_to_select_lst[j], 
                                    np.mean(dup_accuracies), 
                                    np.std(dup_accuracies),
                                    dup_accuracies))

    print(results)


    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]) #[0.35, 0.15, 0.5, 0.7])
    ax = fig.add_subplot(111)


    # for i in range(len(results["random_patches_second"])):
    #     # ax.plot([results["random_patches_second"][i][0], results["selected_patches_first"][i][0]], 
    #     #         [results["random_patches_second"][i][1], results["selected_patches_first"][i][1]], c="black", zorder=1)
        
    #     ax.plot([i, i], 
    #             [results["random_patches_second"][i][1], results["selected_patches_first"][i][1]], c="black", zorder=1)
    # for i, method in enumerate(list(results.keys())):
    #     ax.scatter([x[0] for x in results[method]], [x[1] for x in results[method]], s=50, c=my_plot_colors[i], label=method, zorder=2)

    # ax.scatter([x[0] for x in results["random_patches_second"]], [x[1] for x in results["random_patches_second"]], marker="_", c=my_plot_colors[0], label="random_patches_second", zorder=2)
    # ax.scatter([x[0] for x in results["selected_patches_first"]], [x[1] for x in results["selected_patches_first"]], marker="_", c=my_plot_colors[1], label="selected_patches_first", zorder=2)
    
    # for x in results["random_patches_second"]:
    #     ax.plot([x[0], x[0]], [x[1]-x[2], x[1]+x[2]], c=my_plot_colors[0])
    

    # for x in results["selected_patches_first"]:
    #     ax.plot([x[0], x[0]], [x[1]-x[2], x[1]+x[2]], c=my_plot_colors[1])


    ax.plot([x[0] for x in results["random_patches_second"]], [x[1] for x in results["random_patches_second"]], c=my_plot_colors[0], label="Random Patches")
    ax.plot([x[0] for x in results["selected_patches_first"]], [x[1] for x in results["selected_patches_first"]], c=my_plot_colors[1], label="Selected Patches")
    
    # ax.fill_between([x[0] for x in results["random_patches_second"]], 
    #                 [x[1] - x[2] for x in results["random_patches_second"]], 
    #                 [x[1] + x[2] for x in results["random_patches_second"]], edgecolor=my_plot_colors[0], facecolor=my_plot_colors[0], alpha=0.15)
    # ax.fill_between([x[0] for x in results["selected_patches_first"]], 
    #                 [x[1] - x[2] for x in results["selected_patches_first"]], 
    #                 [x[1] + x[2] for x in results["selected_patches_first"]], edgecolor=my_plot_colors[1], facecolor=my_plot_colors[1], alpha=0.15)
    for x in results["random_patches_second"]:
        print(x)
        ax.scatter([x[0]] * len(x[3]), x[3], s=50, c=my_plot_colors[0])

    for x in results["selected_patches_first"]:
        ax.scatter([x[0]] * len(x[3]), x[3], s=50, c=my_plot_colors[1])   

    # ax.scatter([x[3] for x in results["random_patches_second"]], [x[4] for x in results["random_patches_second"]], s=50, c=my_plot_colors[0], label="random_patches_second", zorder=2)
    # ax.scatter([x[3] for x in results["selected_patches_first"]], [x[4] for x in results["selected_patches_first"]], s=50, c=my_plot_colors[1], label="selected_patches_first", zorder=2)


    # ax.plot([0, max_num_fine_tuning_boxes], [pre_fine_tune_accuracy, pre_fine_tune_accuracy], c="black", linestyle="dashed", label="No Fine-Tuning")

    # ax.scatter(results, np.arange(len(labels))) #, color=colors) #, width=0.4)

    # ax.set_yticks(np.arange(len(labels)))
    # ax.set_yticklabels(labels)

    plt.axhline(y=pre_fine_tune_accuracy, c="black", linestyle="dashdot", label="No Fine-Tuning")
    ax.legend()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Number of Patches") #Annotations")

    ax.set_ylim([0.5, 1])

    plt.tight_layout()

    out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_averaged.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)


def create_fine_tune_plot(baseline, test_set, methods, num_annotations_to_select_lst, num_dups):
    test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
    test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    

    mapping = get_mapping_for_test_set(test_set_image_set_dir)
    # annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
    # annotations = annotation_utils.load_annotations(annotations_path)
    results = {}
    # labels = []

    
    pre_fine_tune_result_name = baseline["model_name"] + "_pre_finetune" #+ str(num_images_to_select)
    result_uuid = mapping[pre_fine_tune_result_name]
    result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)

    predictions_path = os.path.join(result_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path)

    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    accuracies = []
    for image_name in annotations.keys():
        sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
        accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
        accuracies.append(accuracy)

    pre_fine_tune_accuracy = np.mean(accuracies)

    # label_lookup = {
    #     "selected_patches_match_both": "selected_patches",
    #     "random_images": "random_images"
    # }

    max_num_fine_tuning_boxes = 0
    for i, method in enumerate(methods):
        results[method] = []
        for j in range(len(num_annotations_to_select_lst)):

            dup_accuracies = []
            for dup_num in range(num_dups):

                result_name = baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select_lst[j]) + "_annotations_dup_" + str(dup_num)
                
                if result_name in mapping:
                    result_uuid = mapping[result_name]
                    result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)

                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)

                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)


                    num_fine_tuning_boxes = 0
                    num_fine_tuning_regions = 0
                    accuracies = []
                    for image_name in annotations.keys():
                        sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                        accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
                        accuracies.append(accuracy)

                        num_fine_tuning_boxes += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                        num_fine_tuning_regions += len(annotations[image_name]["training_regions"])


                    if num_fine_tuning_boxes > max_num_fine_tuning_boxes:
                        max_num_fine_tuning_boxes = num_fine_tuning_boxes
                    # global_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, list(annotations.keys())) #assessment_images_lst)

                    # test_set_accuracy = global_accuracy #np.mean(accuracy)
                    test_set_accuracy = np.mean(accuracies)
                    dup_accuracies.append(test_set_accuracy)
                    # results[method].append((num_fine_tuning_boxes, test_set_accuracy))
                    results[method].append((num_annotations_to_select_lst[j], num_fine_tuning_regions, test_set_accuracy))

                # results.append(np.mean(dup_accuracies))
                # labels.append(method)
                # results[method].append((num_annotations_to_select_lst[j], np.mean(dup_accuracies)))

    print(results)


    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]) #[0.35, 0.15, 0.5, 0.7])
    ax = fig.add_subplot(111)


    # for i in range(len(results["random_patches_second"])):
    #     # ax.plot([results["random_patches_second"][i][0], results["selected_patches_first"][i][0]], 
    #     #         [results["random_patches_second"][i][1], results["selected_patches_first"][i][1]], c="black", zorder=1)
        
    #     ax.plot([i, i], 
    #             [results["random_patches_second"][i][1], results["selected_patches_first"][i][1]], c="black", zorder=1)
    
    
    # for i, method in enumerate(list(results.keys())):
    #     ax.scatter([x[0] for x in results[method]], [x[1] for x in results[method]], s=50, c=my_plot_colors[i], label=method, zorder=2)

    # ax.scatter([x[0] for x in results["random_patches_second"]], [x[1] for x in results["random_patches_second"]], s=50, c=my_plot_colors[0], label="random_patches_second", zorder=2)
    # ax.scatter([x[0] for x in results["selected_patches_first"]], [x[1] for x in results["selected_patches_first"]], s=50, c=my_plot_colors[1], label="selected_patches_first", zorder=2)
    # ax.scatter([i for i in range(len(results["random_patches_second"]))], [x[1] for x in results["random_patches_second"]], s=50, c=my_plot_colors[0], label="random_patches_second", zorder=2)
    # ax.scatter([i for i in range(len(results["selected_patches_first"]))], [x[1] for x in results["selected_patches_first"]], s=50, c=my_plot_colors[1], label="selected_patches_first", zorder=2)

    ax.scatter([x[0] for x in results["random_patches_second"]], [x[2] for x in results["random_patches_second"]], s=50, c=my_plot_colors[0], label="Random Patch-Regions", zorder=2)
    ax.scatter([x[0] for x in results["selected_patches_first"]], [x[2] for x in results["selected_patches_first"]], s=50, c=my_plot_colors[1], label="Selected Patch-Regions", zorder=2)


    # ax.plot([0, max_num_fine_tuning_boxes], [pre_fine_tune_accuracy, pre_fine_tune_accuracy], c="black", linestyle="dashed", label="No Fine-Tuning")

    # ax.scatter(results, np.arange(len(labels))) #, color=colors) #, width=0.4)

    # ax.set_yticks(np.arange(len(labels)))
    # ax.set_yticklabels(labels)

    plt.xlim([0, num_annotations_to_select_lst[-1] + 250])

    plt.axhline(y=pre_fine_tune_accuracy, c="deeppink", linestyle="dashdot", label="No Fine-Tuning")
    ax.legend()
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Number of Patches") #Annotations")

    plt.tight_layout()

    out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_annotations.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)



def get_mapping_for_test_set(test_set_image_set_dir):

    mapping = {}
    results_dir = os.path.join(test_set_image_set_dir, "model", "results")
    for result_dir in glob.glob(os.path.join(results_dir, "*")):
        request_path = os.path.join(result_dir, "request.json")
        request = json_io.load_json(request_path)
        if request["results_name"] in mapping:
            raise RuntimeError("Duplicate result name: {}, {}".format(test_set_image_set_dir, request["results_name"]))
        mapping[request["results_name"]] = request["request_uuid"]
    return mapping

# def setBoxColors(bp):
#     plt.setp(bp['boxes'][0], color='blue')
#     plt.setp(bp['caps'][0], color='blue')
#     plt.setp(bp['caps'][1], color='blue')
#     plt.setp(bp['whiskers'][0], color='blue')
#     plt.setp(bp['whiskers'][1], color='blue')
#     plt.setp(bp['fliers'][0], color='blue')
#     plt.setp(bp['fliers'][1], color='blue')
#     plt.setp(bp['medians'][0], color='blue')

#     plt.setp(bp['boxes'][1], color='red')
#     plt.setp(bp['caps'][2], color='red')
#     plt.setp(bp['caps'][3], color='red')
#     plt.setp(bp['whiskers'][2], color='red')
#     plt.setp(bp['whiskers'][3], color='red')
#     plt.setp(bp['fliers'][2], color='red')
#     plt.setp(bp['fliers'][3], color='red')
#     plt.setp(bp['medians'][1], color='red')
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def accuracy_sampling_efficiency(test_set, baseline_model, out_dirname):

    test_set_image_set_dir = os.path.join("usr", "data", test_set["username"], "image_sets",
                                           test_set["farm_name"], test_set["field_name"], test_set["mission_date"])
    
    test_set_str = test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]

    mapping = get_mapping_for_test_set(test_set_image_set_dir)
    result_dir = os.path.join(test_set_image_set_dir, "model", "results", mapping[baseline_model["model_name"]])
    
    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    predictions_path = os.path.join(result_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path)


    true_global_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, list(annotations.keys()))


    image_results = {}
    patch_results = {}

    # image estimate
    num_images = len(annotations.keys())
    assessment_image_numbers = np.arange(1, num_images+1)
    num_replications_per_number = 50 #30

    for num_assessment_images in tqdm.tqdm(assessment_image_numbers, desc="Getting accuracy estimates"):
        # print("{}/{}".format(num_assessment_images, num_images))
        image_results[num_assessment_images] = []
        patch_results[num_assessment_images] = []
        for replication in range(num_replications_per_number):

            # num_assessment_images = 15
            candidates = list(annotations.keys())
            total_true_positives = 0
            total_false_positives = 0
            total_false_negatives = 0
            sel_candidates = random.sample(candidates, k=num_assessment_images)
            for sel_candidate in sel_candidates:
                annotated_boxes = annotations[sel_candidate]["boxes"]
                pred_boxes = predictions[sel_candidate]["boxes"]
                pred_scores = predictions[sel_candidate]["scores"]

                sel_pred_boxes = pred_boxes[pred_scores > 0.50]

                num_predicted = sel_pred_boxes.shape[0]
                num_annotated = annotated_boxes.shape[0]

                if num_predicted > 0:
                    if num_annotated > 0:
                        true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(annotated_boxes, sel_pred_boxes, 0.50)
                    else:
                        true_positive = 0
                        false_positive = num_predicted
                        false_negative = 0
                else:
                    if num_annotated > 0:
                        true_positive = 0
                        false_positive = 0
                        false_negative = num_annotated
                    else:
                        true_positive = 0
                        false_positive = 0
                        false_negative = 0

                total_true_positives += true_positive
                total_false_positives += false_positive
                total_false_negatives += false_negative

            denom = (total_true_positives + total_false_positives + total_false_negatives)
            if denom == 0:
                image_global_accuracy_estimate = 1.0
            else:
                image_global_accuracy_estimate = total_true_positives / denom


            # image_global_accuracy_estimate = total_true_positives / (total_true_positives + total_false_positives + total_false_negatives)
            # image_accuracy_estimate_differences.append(abs(true_global_accuracy - image_global_accuracy_estimate))
            image_results[num_assessment_images].append(image_global_accuracy_estimate)

            # image_ave_accuracy_estimate_difference = np.mean(image_accuracy_estimate_differences)

            metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            patch_size = annotation_utils.get_patch_size(annotations, ["test_regions"])

            # patch_divisor = 4

            # num_patches_per_image = patch_divisor * patch_divisor #diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)
            
            num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)
            # print("num_patches_per_image: {}".format(num_patches_per_image))
            num_patches_used = num_patches_per_image * num_assessment_images

            # print("looking at {} patches".format(num_patches_used))
            # num_patch_sample_images = m.floor(num_patches_used / 

            num_sample_patches_per_image = m.floor(num_patches_used / num_images)
            num_additional = num_patches_used - (num_sample_patches_per_image * num_images)


            patch_candidates = {}
            num_patch_candidates = 0

            for image_name in annotations.keys():
                image_w = metadata["images"][image_name]["width_px"]
                image_h = metadata["images"][image_name]["height_px"]


                # region_w = image_w / patch_divisor
                # region_h = image_h / patch_divisor

                # patch_candidates[image_name] = []
                # for h_index in range(patch_divisor):
                #     for w_index in range(patch_divisor):

                #         patch_coords = [
                #             patch_size * h_index,
                #             region_h * w_index,
                #             min((region_h * h_index) + region_h, image_h),
                #             min((region_w * w_index) + region_w, image_w)
                #         ]

                #         patch_candidates[image_name].append(patch_coords)
                #         num_patch_candidates += 1


                patch_overlap_percent = 0
                overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

                incr = patch_size - overlap_px
                w_covered = max(image_w - patch_size, 0)
                num_w_patches = m.ceil(w_covered / incr) + 1

                h_covered = max(image_h - patch_size, 0)
                num_h_patches = m.ceil(h_covered / incr) + 1

                patch_candidates[image_name] = []
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        patch_candidates[image_name].append(patch_coords)
                        num_patch_candidates += 1


            # print("num_patch_candidates: {}".format(num_patch_candidates))
            # print("num_sample_patches_per_image: {}".format(num_sample_patches_per_image))
            # print("num_additional: {}".format(num_additional))

            total_true_positives = 0
            total_false_positives = 0
            total_false_negatives = 0
            for image_name in patch_candidates.keys():
                if num_additional > 0:
                    num_samples = num_sample_patches_per_image + 1
                    num_additional -= 1
                else:
                    num_samples = num_sample_patches_per_image

                sel_patches = random.sample(patch_candidates[image_name], k=num_samples)


                annotated_boxes = annotations[image_name]["boxes"]
                pred_boxes = predictions[image_name]["boxes"]
                pred_scores = predictions[image_name]["scores"]

                sel_pred_boxes = pred_boxes[pred_scores > 0.50]
                # for sel_patch in sel_patches:
                sample_anno_box_inds = box_utils.get_contained_inds(annotated_boxes, sel_patches)
                sample_pred_box_inds = box_utils.get_contained_inds(sel_pred_boxes, sel_patches)

                sample_anno_boxes = annotated_boxes[sample_anno_box_inds]
                sample_pred_boxes = sel_pred_boxes[sample_pred_box_inds]


                num_predicted = sample_anno_boxes.shape[0]
                num_annotated = sample_pred_boxes.shape[0]

                if num_predicted > 0:
                    if num_annotated > 0:
                        true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(sample_anno_boxes, sample_pred_boxes, 0.50)
                    else:
                        true_positive = 0
                        false_positive = num_predicted
                        false_negative = 0
                else:
                    if num_annotated > 0:
                        true_positive = 0
                        false_positive = 0
                        false_negative = num_annotated
                    else:
                        true_positive = 0
                        false_positive = 0
                        false_negative = 0

                total_true_positives += true_positive
                total_false_positives += false_positive
                total_false_negatives += false_negative

            


            denom = (total_true_positives + total_false_positives + total_false_negatives)
            if denom == 0:
                patch_global_accuracy_estimate = 1.0
            else:
                patch_global_accuracy_estimate = total_true_positives / denom
            patch_results[num_assessment_images].append(patch_global_accuracy_estimate)


    # print("True accuracy: {}".format(true_global_accuracy))
    # print("Image estimate: {}".format(image_global_accuracy_estimate))
    # print("Patch estimate: {}".format(patch_global_accuracy_estimate))


    fig = plt.figure(figsize=(15, 5))
    # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])


    # position = 1
    # for num_assessment_images in assessment_image_numbers:
    #     plot_data = []
    #     plot_data.append(image_results[num_assessment_images])
    #     plot_data.append(patch_results[num_assessment_images])
    #     bp = plt.boxplot(plot_data, positions=[position, position+1], widths=0.6)
    #     position += 3
    #     setBoxColors(bp)

    # plt.xlim(0, position+1)
    # plt.ylim(0, 1.0)

    # ax.set_xticklabels(assessment_image_numbers)
    # ax.set_xticks([x + 0.5 for x in range(1, position, 3)])
    data_a = [image_results[num] for num in assessment_image_numbers]
    data_b = [patch_results[num] for num in assessment_image_numbers]

    for i in range(len(data_a)):
        pos = i * 2.0 - 0.2
        plt.plot([pos, pos], [min(data_a[i]), max(data_a[i])], color=my_plot_colors[0], linewidth=2)

    for i in range(len(data_b)):
        pos = i * 2.0 + 0.2
        plt.plot([pos, pos], [min(data_b[i]), max(data_b[i])], color=my_plot_colors[1], linewidth=2)        

    # bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    # bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    # set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    # set_box_color(bpr, '#2C7BB6')


    plt.plot([0, (len(data_a)-1) * 2], [true_global_accuracy, true_global_accuracy], color="black", linestyle="dashed")

    # plt.xlim(0, position+1)
    # plt.ylim(0, 1.0)

    plt.plot([], c=my_plot_colors[0], label='Image Sampling')
    plt.plot([], c=my_plot_colors[1], label='Patch Region Sampling')
    plt.legend()

    plt.xticks(ticks=[i * 2.0 for i in range(num_images)], labels=assessment_image_numbers)
    # plt.xticklabels(assessment_image_numbers)

    plt.tight_layout()


    out_path = os.path.join("eval_charts", out_dirname, test_set_str + ".svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)









def create_eval_size_plot_individual_test_sets(test_sets, baseline_sets, out_dirname):

    results = {} #[]
    mappings = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    # json_io.print_json(mappings)
    for k in baseline_sets.keys():
        # results[k] = []
        # print(k)
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]

            # test_set_accuracies = []
            for test_set in test_sets:
                test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                if test_set_str not in results:
                    results[test_set_str] = {}
                if k not in results[test_set_str]:
                    results[test_set_str][k] = []
                
                # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
                rep_accuracies = []
                for rep_num in range(5):

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                    rep_accuracies.append(rep_accuracy)

                test_set_accuracy = np.mean(rep_accuracies)
                test_set_stdev = np.std(rep_accuracies)
                results[test_set_str][k].append((patch_num, test_set_accuracy, test_set_stdev))

    for test_set_str in results.keys():

        fig = plt.figure(figsize=(10,10))
    
        for i, k in enumerate(list(results[test_set_str].keys())):
            plt.plot([x[0] for x in results[test_set_str][k]], [x[1] for x in results[test_set_str][k]], color=my_plot_colors[i], marker="o", linestyle="dashed", label=k, linewidth=1)
    
            for x in results[test_set_str][k]:
                plt.plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[i], linestyle="solid", linewidth=1)

        plt.xlabel("Number of Patches")
        plt.ylabel("Test Accuracy")
        plt.legend()

        out_path = os.path.join("eval_charts", out_dirname, test_set_str + ".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)

def create_individual_image_sets_eval_size_plot_id_ood(id_test_sets, ood_test_sets, baseline_sets, out_dirname):

    print("num id test sets: {}".format(len(id_test_sets)))
    print("num ood test sets: {}".format(len(ood_test_sets)))
    results = {}
    mappings = {}
    test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):

        if i == 0:
            label = "id"
        else:
            label = "ood"

        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            test_set_str_to_label[test_set_str] = label

    if os.path.exists(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json")) and \
        os.path.exists(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json")):
            
        image_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json"))
        instance_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"))
    else:
        image_based_test_set_results = {}
        instance_based_test_set_results = {}

        json_io.print_json(mappings)
        for k in baseline_sets.keys():
            results[k] = []
            baselines = baseline_sets[k]
            for baseline in baselines:
                model_name = baseline["model_name"]
                patch_num = baseline["patch_num"]

                print(model_name)

                for test_set in id_test_sets:

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    
                    print("\tid: {}".format(test_set_str))
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])                
                    
                    id_test_set_accuracies = []
                    id_global_test_set_accuracies = []
                    for rep_num in range(5):
                        print("\t\t{}".format(rep_num))
                        

                        

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)

                        # inds = df["Annotated Count"] > 10
                        # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        assessment_images = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                assessment_images.append(image_name)

                        id_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                        # id_global_test_set_accuracy = 0

                        id_test_set_accuracies.append(id_test_set_accuracy)
                        id_global_test_set_accuracies.append(id_global_test_set_accuracy)


                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []
                    image_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(id_test_set_accuracies))
                    )
                    if test_set_str not in instance_based_test_set_results:
                        instance_based_test_set_results[test_set_str] = []
                    instance_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(id_global_test_set_accuracies))
                    )
                

                for test_set in ood_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

                    print("\tood: {}".format(test_set_str))
                    ood_test_set_accuracies = []
                    ood_global_test_set_accuracies = []
                    for rep_num in range(5):
                        print("\t\t{}".format(rep_num))
                    

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)
                        # inds = df["Annotated Count"] > 10
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                        

                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        assessment_images = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                assessment_images.append(image_name)
                        # assessment_images = random.sample(assessment_images, 2)
                        # assessment_images = (np.array(assessment_images)[[0,3]]).tolist()

                        # ood_test_set_accuracy = df[df["Image Name"].astype(str).isin(assessment_images)]["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                        # print(assessment_images)
                        # print(df["Image Name"])
                        # print(ood_test_set_accuracy)
                        ood_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                        # ood_global_test_set_accuracy = 0

                        ood_test_set_accuracies.append(ood_test_set_accuracy)
                        ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)


                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []

                    image_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(ood_test_set_accuracies))
                    )

                    if test_set_str not in instance_based_test_set_results:
                        instance_based_test_set_results[test_set_str] = []

                    instance_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(ood_global_test_set_accuracies))
                    )

    json_io.save_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json"), image_based_test_set_results)
    json_io.save_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"), instance_based_test_set_results)

    json_io.print_json(instance_based_test_set_results)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    for test_set_str in image_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "id":
            plot_color = my_plot_colors[0]
        else:
            plot_color = my_plot_colors[1]
        axs[0].plot(
            [x[0] for x in image_based_test_set_results[test_set_str]],
            [x[1] for x in image_based_test_set_results[test_set_str]],
            color=plot_color,
            alpha=0.5
        )

    for test_set_str in instance_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "id":
            plot_color = my_plot_colors[0]
        else:
            plot_color = my_plot_colors[1]
        axs[1].plot(
            [x[0] for x in instance_based_test_set_results[test_set_str]],
            [x[1] for x in instance_based_test_set_results[test_set_str]],
            color=plot_color,
            alpha=0.5
        )

    largest_set_id_results = []
    for test_set_str in instance_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "id":
            largest_set_id_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    
    largest_set_id_results.sort(key=lambda x: x[1])
        
    print("largest_set_id_results")
    for v in largest_set_id_results:
        print(v)
    # print(largest_set_id_results)


    out_path = os.path.join("eval_charts", out_dirname, "id_ood_individual_image_sets.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)


def create_eval_size_plot_id_ood(id_test_sets, ood_test_sets, baseline_sets, out_dirname):


    results = {} #[]
    mappings = {}
    # test_set_types = [("ood", ood_test_sets), ("id", id_test_sets)]
    # labels = {}
    # test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):

        # if i == 0:
        #     label = "id"
        # else:
        #     label = "ood"

        # test_set_label = test_set_type[0]
        # test_sets = test_set_type[1]
        # labels[test_set_label] = []
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            # labels[test_set_label].append(test_set_str)
            # test_set_str_to_label[test_set_str] = label

    # image_based_test_set_results = {}
    # instance_based_test_set_results = {}
    bad_id_test_sets = [
        'eval BlaineLake River 2021-06-09',
        'eval row_spacing brown 2021-06-08',
        'eval Biggar Dennis1 2021-06-04',
        'eval BlaineLake Serhienko9S 2022-06-14'
    ]

    json_io.print_json(mappings)
    for k in baseline_sets.keys():
        results[k] = []
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]
            
            # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
            ood_rep_accuracies = []
            ood_global_rep_accuracies = []
            id_rep_accuracies = []
            id_global_rep_accuracies = []
            for rep_num in range(5):

                id_test_set_accuracies = []
                id_global_test_set_accuracies = []

                for test_set in id_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    if test_set_str in bad_id_test_sets:
                        continue
                    
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)

                    # inds = df["Annotated Count"] > 10
                    # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)



                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)

                    id_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")



                    id_test_set_accuracies.append(id_test_set_accuracy)
                    id_global_test_set_accuracies.append(id_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(id_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(id_global_test_set_accuracy)

                id_rep_accuracy = np.mean(id_test_set_accuracies)
                id_rep_accuracies.append(id_rep_accuracy)

                id_global_rep_accuracy = np.mean(id_global_test_set_accuracies)
                id_global_rep_accuracies.append(id_global_rep_accuracy)                


                ood_test_set_accuracies = []
                ood_global_test_set_accuracies = []

                for test_set in ood_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    # inds = df["Annotated Count"] > 10
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)

                    ood_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    ood_test_set_accuracies.append(ood_test_set_accuracy)
                    ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(ood_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(ood_global_test_set_accuracy)



                ood_rep_accuracy = np.mean(ood_test_set_accuracies)
                ood_rep_accuracies.append(ood_rep_accuracy)
                # print(model_name, rep_accuracy)


                ood_global_rep_accuracy = np.mean(ood_global_test_set_accuracies)
                ood_global_rep_accuracies.append(ood_global_rep_accuracy)                



            id_baseline_accuracy = np.mean(id_rep_accuracies)
            id_baseline_stdev = np.std(id_rep_accuracies)

            ood_baseline_accuracy = np.mean(ood_rep_accuracies)
            ood_baseline_stdev = np.std(ood_rep_accuracies)


            id_global_baseline_accuracy = np.mean(id_global_rep_accuracies)
            id_global_baseline_stdev = np.std(id_global_rep_accuracies)

            ood_global_baseline_accuracy = np.mean(ood_global_rep_accuracies)
            ood_global_baseline_stdev = np.std(ood_global_rep_accuracies)

            # print(ood_baseline_accuracy, ood_global_baseline_accuracy)


            results[k].append((patch_num, 
                               id_baseline_accuracy, 
                               id_baseline_stdev, 
                               ood_baseline_accuracy, 
                               ood_baseline_stdev,
                               id_global_baseline_accuracy,
                               id_global_baseline_stdev,
                               ood_global_baseline_accuracy,
                               ood_global_baseline_stdev
                               ))

    # fig = plt.figure(figsize=(8, 6))

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    xticks = [0, 40000]


    for i, k in enumerate(list(results.keys())):
        # axs[0].scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
        # axs[0].scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

        axs[0].plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], label="In Domain")
        axs[0].plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
        
        axs[0].fill_between([x[0] for x in results[k]], [x[1] - x[2] for x in results[k]], [x[1] + x[2] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
        axs[0].fill_between([x[0] for x in results[k]], [x[3] - x[4] for x in results[k]], [x[3] + x[4] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

    axs[0].set_xlabel("Number of Training Patches")
    axs[0].set_ylabel("Image-Based Accuracy")
    axs[0].set_ylim([0.45, 0.8])
    # axs[0].set_xscale("log")
    # axs[0].set_xticks(xticks)
    # axs[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # axs[0].set_xticklabels(xticklabels)
    axs[0].legend()

    for i, k in enumerate(list(results.keys())):
        # plt.scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
        # plt.scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

        # for x in results[k]:
        #     plt.plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
        # for x in results[k]:
        #     plt.plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

        axs[1].plot([x[0] for x in results[k]], [x[5] for x in results[k]], color=my_plot_colors[0], label="In Domain")
        axs[1].plot([x[0] for x in results[k]], [x[7] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
        
        axs[1].fill_between([x[0] for x in results[k]], [x[5] - x[6] for x in results[k]], [x[5] + x[6] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
        axs[1].fill_between([x[0] for x in results[k]], [x[7] - x[8] for x in results[k]], [x[7] + x[8] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

    axs[1].set_xlabel("Number of Training Patches")
    axs[1].set_ylabel("Instance-Based Accuracy")
    axs[1].set_ylim([0.45, 0.8])
    # axs[1].set_xscale("log")
    # axs[1].set_xticks(xticks)
    # axs[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # plt.ylim([0.48, 0.87])
    axs[1].legend()
    plt.suptitle("Effect of Training Set Size")

    # plt.legend()
    plt.tight_layout()

    out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_no_bad_id.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)




    # fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # num_dups = 5
    # for test_set_str in image_based_test_set_results:
    #     # each row is a duplication
    #     r = np.array(image_based_test_set_results[test_set_str]).reshape(num_dups, -1)
    #     q = np.mean(r, axis=0)
    #     axs[0].plot()










def create_eval_size_plot(test_sets, baseline_sets, out_dirname):



    results = {} #[]
    mappings = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    json_io.print_json(mappings)
    for k in baseline_sets.keys():
        results[k] = []
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]
            
            # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
            rep_accuracies = []
            for rep_num in range(5):
                test_set_accuracies = []

                for test_set in test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    print(test_set_str)

                    
                    
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                    # predictions_path = os.path.join(result_dir, "predictions.json")
                    # predictions = annotation_utils.load_predictions(predictions_path)
                    # annotations_path = os.path.join(result_dir, "annotations.json")
                    # annotations = annotation_utils.load_annotations(annotations_path)
                    # assessment_images = []
                    # for image_name in annotations.keys():
                    #     if len(annotations[image_name]["test_regions"]) > 0:
                    #         assessment_images.append(image_name)

                    # test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    test_set_accuracies.append(test_set_accuracy)

                rep_accuracy = np.mean(test_set_accuracies)
                rep_accuracies.append(rep_accuracy)
                print(model_name, rep_accuracy)

            baseline_accuracy = np.mean(rep_accuracies)
            baseline_stdev = np.std(rep_accuracies)
            results[k].append((patch_num, baseline_accuracy, baseline_stdev))
            # results.append((patch_num, baseline_accuracy))
    fig = plt.figure(figsize=(8, 6))
    # colors = ["salmon", "royalblue", "forestgreen", "orange", "mediumorchid"]
    for i, k in enumerate(list(results.keys())):
        plt.scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[i], marker="_", label=k, zorder=2)
        # plt.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color="black", marker=None, linestyle="dotted", label=k, linewidth=1, zorder=1)
        for x in results[k]:
            plt.plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[i], linestyle="solid", linewidth=1, zorder=2)

    plt.title("Effect of Training Set Size")
    plt.xlabel("Number of Training Patches")
    plt.ylabel("Test Accuracy")
    # plt.legend()
    plt.tight_layout()

    out_path = os.path.join("eval_charts", out_dirname, "plot.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)


    # individual_results = {}
    # for k in individual_test_set_results.keys():
    #     individual_results[k] = []
    #     rep_0_res = individual_test_set_results[k][list(individual_test_set_results[k].keys())[0]]
    #     for i in range():
    #         rep_results = []
    #         for rep_num in individual_test_set_results[k].keys():
    #             rep_results.append(individual_test_set_results[k][rep_num][i])
            
    #         individual_results[k].append((rep_0_res[i], np.mean(rep_results), np.std(rep_results)))


def create_weed_comparison_plot(test_sets, baselines):


    mappings = {}
    # results = {
    #     "overall": {
    #         "no_weed": [],
    #         "weed": []
    #     }
    # }
    # results = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "no_weed": [],
        #     "weed": []
        # }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    no_weed_results = []
    weed_results = []

    for i in range(2):
        # if i == 0:
        #     baselines = no_weed_baselines
        #     result_key = "no_weed"
        # else:
        #     baselines = weed_baselines
        #     result_key = "weed"


        for baseline in baselines:
            baseline_accuracies = []
            for test_set in test_sets:
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

                rep_accuracies = []
                for rep_num in range(3):
                    if i == 0:
                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    else:
                        model_name = baseline["model_name"] + "_and_CottonWeedDet12_rep_" + str(rep_num)

                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                    print("\t{}: {}: {}".format(test_set_str, model_name, rep_accuracy))
                    rep_accuracies.append(rep_accuracy)

                baseline_accuracy = np.mean(rep_accuracies)
                # baseline_variance = np.std(rep_accuracies)
                baseline_accuracies.append(baseline_accuracy)


                # results[test_set_str][result_key].append(baseline_accuracy)

                #     # (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



            overall_baseline_accuracy = np.mean(baseline_accuracies)

            if i == 0:
                no_weed_results.append(overall_baseline_accuracy)
            else:
                weed_results.append(overall_baseline_accuracy)

            # results["overall"][result_key].append(
            #     # (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], 
            #     overall_baseline_accuracy)
            #     # np.min(baseline_accuracies),
            #     # np.max(baseline_accuracies)))

    
    labels = []
    for baseline in baselines:
        labels.append(baseline["model_name"])
    # weed_labels = []
    # for weed_baseline in weed_baselines:
    #     weed_labels.append(weed_baseline["model_name"])

    # labels = 

    labels = np.array(labels)
    no_weed_results = np.array(no_weed_results)
    weed_results = np.array(weed_results)

    inds = np.argsort(weed_results)

    labels = labels[inds]
    no_weed_results = no_weed_results[inds]
    weed_results = weed_results[inds]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.30, 0.05, 0.65, 0.9])

    ax.scatter(no_weed_results, np.arange(len(labels)), 
               label="no_weed", marker="o", color="blue")
    ax.scatter(weed_results, np.arange(len(labels)), 
               label="weed", marker="o", color="red")
    
    
    ax.set_yticks(np.arange(0, len(labels)))
    ax.set_yticklabels(labels)


    ax.legend()
    ax.set_xlabel("Test Accuracy")

    out_path = os.path.join("eval_charts", "weed_comparisons", "overall.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    
def create_eval_improvement_plot(test_sets, single_baselines, single_baselines_improved, diverse_baselines, diverse_baselines_improved):


    mappings = {}
    results = {
        "overall": {
            "single": [],
            "diverse": [],
            "single_improved": [],
            "diverse_improved": []
        }

    }
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        results[test_set_str] = {
            "single": [],
            "diverse": [],
            "single_improved": [],
            "diverse_improved": []
        }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    for i in range(4):
        if i == 0:
            baselines = single_baselines
            result_key = "single"
        elif i == 1:
            baselines = diverse_baselines
            result_key = "diverse"
        elif i == 2:
            baselines = single_baselines_improved
            result_key = "single_improved"
        else:
            baselines = diverse_baselines_improved
            result_key = "diverse_improved"



        for baseline in baselines:
            baseline_accuracies = []
            for test_set in test_sets:
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

                rep_accuracies = []
                for rep_num in range(1):
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                    rep_accuracies.append(rep_accuracy)

                baseline_accuracy = np.mean(rep_accuracies)
                # baseline_variance = np.std(rep_accuracies)
                baseline_accuracies.append(baseline_accuracy)


                results[test_set_str][result_key].append(
                    (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



            overall_baseline_accuracy = np.mean(baseline_accuracies)

            results["overall"][result_key].append(
                (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], 
                overall_baseline_accuracy,
                np.min(baseline_accuracies),
                np.max(baseline_accuracies)))


    for test_set_str in results:
        single_tuples = results[test_set_str]["single"]
        diverse_tuples = results[test_set_str]["diverse"]
        single_improved_tuples = results[test_set_str]["single_improved"]
        diverse_improved_tuples = results[test_set_str]["diverse_improved"]



        single_tuples.sort(key=lambda x: x[1])
        diverse_tuples.sort(key=lambda x: x[1])

        labels = []
        for single_tuple in single_tuples:
            labels.append(single_tuple[0])
        for diverse_tuple in diverse_tuples:
            labels.append(diverse_tuple[0])

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0.30, 0.05, 0.65, 0.9])

        ax.scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), marker="x", color="red", label="Single Image Set (630 patches)", zorder=2)
        ax.scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), marker="x", color="blue", label="Diverse Random Selection (630 patches)", zorder=2)
        
        ax.scatter([x[1] for x in single_improved_tuples], np.arange(len(single_tuples)), marker="o", color="red", label="Single Image Set (1500 patches)", zorder=2)
        ax.scatter([x[1] for x in diverse_improved_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), marker="o", color="blue", label="Diverse Random Selection (1500 patches)", zorder=2)
        
        # if test_set_str == "overall":
        #     ax.scatter([x[2] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
        #     ax.scatter([x[2] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")
            
        #     ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
        #     ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")

        ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
        ax.set_yticklabels(labels)


        ax.legend()
        ax.set_xlabel("Test Accuracy")

        out_path = os.path.join("eval_charts", "peturbation_comparisons", test_set_str + ".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)





def create_patch_merging_plot(test_sets, baseline, merging_prefixes):


    mappings = {}
    # results = {
    #     "overall": {
    #         "single": [],
    #         "diverse": [],
    #         "single_improved": [],
    #         "diverse_improved": []
    #     }
    # }
    results = {}
    results["overall"] = {}
    # for merging_prefix in merging_prefixes:
    #     results["overall"][merging_prefix] = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": [],
        #     "single_improved": [],
        #     "diverse_improved": []
        # }
        results[test_set_str] = {}
        # for merging_prefix in merging_prefixes:
        #     results[test_set_str][merging_prefix] = []

        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    # for i in range(4):
        # if i == 0:
        #     baselines = single_baselines
        #     result_key = "single"
        # elif i == 1:
        #     baselines = diverse_baselines
        #     result_key = "diverse"
        # elif i == 2:
        #     baselines = single_baselines_improved
        #     result_key = "single_improved"
        # else:
        #     baselines = diverse_baselines_improved
        #     result_key = "diverse_improved"


    for merging_prefix in merging_prefixes:
        # for baseline in baselines:
        baseline_accuracies = []
        baseline_true_positives_lst = []
        baseline_false_positives_lst = []
        baseline_false_negatives_lst = []
        all_count_diffs = []
        for test_set in test_sets:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])

            # rep_accuracies = []
            # for rep_num in range(1):
            model_name = baseline["model_name"] + "_rep_" + str(0)
            model_dir = os.path.join(test_set_image_set_dir, "model", "results")
            result_dir = os.path.join(model_dir, mappings[test_set_str][merging_prefix + model_name])
            excel_path = os.path.join(result_dir, "metrics.xlsx")
            df = pd.read_excel(excel_path, sheet_name=0)
            rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
            baseline_true_positives = df["True Positives (IoU=.50, conf>.50)"].sum(skipna=True)
            baseline_false_positives = df["False Positives (IoU=.50, conf>.50)"].sum(skipna=True)
            baseline_false_negatives = df["False Negatives (IoU=.50, conf>.50)"].sum(skipna=True)
            sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
            annotated_counts = sub_df["Annotated Count"]
            predicted_counts = sub_df["Predicted Count"]
            count_diffs = (predicted_counts - annotated_counts).tolist()

            # rep_accuracies.append(rep_accuracy)

            baseline_accuracy = rep_accuracy #np.mean(rep_accuracies)
            # baseline_variance = np.std(rep_accuracies)
            baseline_accuracies.append(baseline_accuracy)
            baseline_true_positives_lst.append(baseline_true_positives)
            baseline_false_positives_lst.append(baseline_false_positives)
            baseline_false_negatives_lst.append(baseline_false_negatives)
            all_count_diffs.extend(count_diffs)
            
            results[test_set_str][merging_prefix] = [
                    baseline_accuracy,
                    baseline_true_positives,
                    baseline_false_positives,
                    baseline_false_negatives,
                    count_diffs
             ]

            # results[test_set_str][merging_prefix].append(baseline_accuracy)

                # (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



        # overall_baseline_accuracy = np.mean(baseline_accuracies)

        results["overall"][merging_prefix] = [
            np.mean(baseline_accuracies),
            np.sum(baseline_true_positives_lst),
            np.sum(baseline_false_positives_lst),
            np.sum(baseline_false_negatives_lst),
            all_count_diffs
        ]

        # results["overall"][merging_prefix].append(
        #     # (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], 
        #     overall_baseline_accuracy) #,
        #     # np.min(baseline_accuracies),
        #     # np.max(baseline_accuracies)))


    # color_options = ["salmon", "royalblue", "forestgreen"]
    # colors = {}
    # for i, merging_prefix in enumerate(merging_prefixes):
    #     colors[merging_prefix] = color_options[i]

    label_lookup = {
        "no_overlap_": "0% overlap + NMS",
        "no_prune_": "50% overlap + NMS",
        "": "50% overlap + prune + NMS",
        # "alt_prune_": "50% overlap \n+ alt_prune \n+ NMS"
    }

    for test_set_str in ["overall"]:
        total_min_diff = 100000
        total_max_diff = -100000
        for merging_prefix in merging_prefixes:
            print(results[test_set_str][merging_prefix][4])
            min_diff = np.min(results[test_set_str][merging_prefix][4])
            max_diff = np.max(results[test_set_str][merging_prefix][4])
            if min_diff < total_min_diff:
                total_min_diff = min_diff
            if max_diff > total_max_diff:
                total_max_diff = max_diff
            # bars.append(results[test_set_str][merging_prefix][i])
            # labels.append(label_lookup[merging_prefix])
        end_point = max(abs(total_min_diff), abs(total_max_diff))
        fig, axs = plt.subplots(len(merging_prefixes), 1, figsize=(10, 8))
        fig.suptitle("Methods for Merging Patch Predictions: Difference in Count (Predicted - Annotated)")
        for i, merging_prefix in enumerate(merging_prefixes):
            counts, bins = np.histogram(results[test_set_str][merging_prefix][4], bins=2*end_point, range=(-end_point, end_point))
            axs[i].stairs(counts, bins, color=my_plot_colors[i], fill=True)
            # label=label_lookup[merging_prefix], 
            # axs[i].legend()
            props = dict(boxstyle="round", facecolor="white", alpha=0.5)
            textstr = label_lookup[merging_prefix]
            axs[i].text(0.985, 0.94, textstr, transform=axs[i].transAxes,
                verticalalignment='top', horizontalalignment="right", bbox=props, fontsize=12)
            # axs[i].set_title(textstr)

            # plt.locator_params(axis="y", nbins=4)
            yticks = ticker.MaxNLocator(4)
            axs[i].yaxis.set_major_locator(yticks)
        plt.tight_layout()
        out_path = os.path.join("eval_charts", "patch_merge_comparisons", "hist.png")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=200)

    # for test_set_str in ["overall"]: #results:

    #     for i in range(4):

    #         if i == 0:
    #             ylabel = "Test Accuracy"
    #         elif i == 1:
    #             ylabel = "Number of True Positives"
    #         elif i == 2:
    #             ylabel = "Number of False Positives"
    #         else:
    #             ylabel = "Number of False Negatives"
    #         # labels = []
    #         # for merging_prefix in merging_prefixes:
    #         #     # tuples = results[test_set_str][merging_prefix]

    #         #     # tuples.sort(key=lambda x: x[1])

    #         #     if i == 0:
    #         #         for tup in tuples:
    #         #             labels.append(tup[0])
    #         # tuples = results[test_set_str]

    #         # tuples.sort(key=lambda x: x[1])

    #         # labels = []
    #         # for tup in results[test_set_str][list(results[test_set_str].keys())[0]]:
    #         #     labels.append(tup[0])

    #         fig = plt.figure(figsize=(5, 5))
    #         ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])


    #         bars = []
    #         labels = []
    #         for merging_prefix in merging_prefixes:
    #             bars.append(results[test_set_str][merging_prefix][i])
    #             labels.append(label_lookup[merging_prefix])
    #         print(bars)
    #         print(merging_prefixes)
    #         print(color_options)
    #             # ax.scatter([x[1] for x in results[test_set_str][merging_prefix]], np.arange(len(labels)), marker="o", color=colors[merging_prefix], label=label_lookup[merging_prefix], zorder=2) #label="Single Image Set (630 patches)", zorder=2)
    #         ax.bar(labels, bars, color=color_options, width=0.4)
    #         # ax.scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), marker="x", color="blue", label="Diverse Random Selection (630 patches)", zorder=2)
            
    #         # ax.scatter([x[1] for x in single_improved_tuples], np.arange(len(single_tuples)), marker="o", color="red", label="Single Image Set (1500 patches)", zorder=2)
    #         # ax.scatter([x[1] for x in diverse_improved_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), marker="o", color="blue", label="Diverse Random Selection (1500 patches)", zorder=2)
            
    #         # if test_set_str == "overall":
    #         #     ax.scatter([x[2] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
    #         #     ax.scatter([x[2] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")
                
    #         #     ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
    #         #     ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")

    #         # ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
    #         # ax.set_yticklabels(labels)


    #         # ax.legend()
    #         # ax.set_xlabel("Test Accuracy")
    #         ax.set_ylabel(ylabel)

    #         out_path = os.path.join("eval_charts", "patch_merge_comparisons", ylabel + ".svg")
    #         out_dir = os.path.dirname(out_path)
    #         os.makedirs(out_dir, exist_ok=True)
    #         plt.savefig(out_path)





# def create_correlation_plot(test_sets, baseline_sets, out_dirname):


    


def create_eval_min_num_plot_individual_test_sets(test_sets, single_baselines, diverse_baselines, out_dirname):


    mappings = {}
    # results = {
    #     "overall": {
    #         "single": [],
    #         "diverse": []
    #     }

    # }
    results = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": []
        # }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    for i in range(2):
        if i == 0:
            baselines = single_baselines
            result_key = "single"
        else:
            baselines = diverse_baselines
            result_key = "diverse"

        for baseline in baselines:

            for test_set in test_sets:

                rep_accuracies = []
                for rep_num in range(1):
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                    rep_accuracies.append(rep_accuracy)

                test_set_accuracy = np.mean(rep_accuracies)
                test_set_std = np.std(rep_accuracies)
                if test_set_str not in results:
                    results[test_set_str] = {}
                if result_key not in results[test_set_str]:
                    results[test_set_str][result_key] = []
                results[test_set_str][result_key].append(
                    (baseline["model_label"], 
                     test_set_accuracy,
                     test_set_std
                ))


    for test_set_str in results:
        print(test_set_str)
        single_tuples = results[test_set_str]["single"]
        diverse_tuples = results[test_set_str]["diverse"]

        single_tuples.sort(key=lambda x: x[1])
        diverse_tuples.sort(key=lambda x: x[1])

        labels = []
        for single_tuple in single_tuples:
            labels.append(single_tuple[0])
        for diverse_tuple in diverse_tuples:
            labels.append(diverse_tuple[0])

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0.35, 0.05, 0.60, 0.9])

        ax.scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="d", zorder=2)
        ax.scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="d", zorder=2)
        for i, x in enumerate(single_tuples):
            ax.plot([x[1] - x[2], x[1] + x[2]], [i, i], color=my_plot_colors[1])

        for i, x in enumerate(diverse_tuples):
            ax.plot([x[1] - x[2], x[1] + x[2]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

        ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
        ax.set_yticklabels(labels)

        ax.set_xlabel("Test Accuracy")
        ax.set_title("Effect of Training Set Diversity")

        out_path = os.path.join("eval_charts", out_dirname, test_set_str + ".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)



def get_min_num_results(test_sets, single_baselines, diverse_baselines):

    mappings = {}
    results = {
        "overall": {
            "single": [],
            "diverse": []
        }

    }
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": []
        # }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    for i in range(2):
        if i == 0:
            baselines = single_baselines
            result_key = "single"
        else:
            baselines = diverse_baselines
            result_key = "diverse"

        for baseline in baselines:
            rep_accuracies = []
            for rep_num in range(5):

                model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                test_set_accuracies = []
                for test_set in test_sets:
                    print(test_set)
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                    # rep_accuracies = []
                    # for rep_num in range(3):
                    # model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                    # rep_accuracies.append(rep_accuracy)

                    test_set_accuracies.append(test_set_accuracy)

                rep_accuracy = np.mean(test_set_accuracies)
                rep_accuracies.append(rep_accuracy)

            # if i == 1:
            print(baseline["model_name"], rep_accuracies)

            baseline_accuracy = np.mean(rep_accuracies)
            baseline_std = np.std(rep_accuracies)
        

        
                    # baseline_accuracy = np.mean(rep_accuracies)
                    # #  baseline_variance = np.std(rep_accuracies)
                    # baseline_accuracies.append(baseline_accuracy)


                    # results[test_set_str][result_key].append(
                    #     (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



            # overall_baseline_accuracy = np.mean(baseline_accuracies)



            results["overall"][result_key].append(
                (baseline["model_label"], 
                baseline_accuracy, #overall_baseline_accuracy,
                baseline_std))
                
                #np.min(baseline_accuracies),
                #np.max(baseline_accuracies)))

    return results




# def create_disturbed_plot(id_test_sets, ood_test_sets, baselines, out_dirname):



def create_removal_plot(test_sets, baselines, out_dirname):

    mappings = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    results = []
    for baseline in baselines:
        # rep_accuracies_iou_10 = []
        # rep_accuracies_iou_50 = []
        # rep_abs_dics = []
        baseline_abs_dics = []
        for rep_num in range(1):

            model_name = baseline["model_name"] + "_rep_" + str(rep_num)
            
            # test_set_accuracies_iou_10 = []
            # test_set_accuracies_iou_50 = []
            # test_set_dics = []

            rep_abs_dics = []

            for test_set in test_sets:

                # print(test_set)
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])



                # annotations_path = os.path.join(result_dir, "annotations.json")
                # annotations = annotation_utils.load_annotations(annotations_path)

                # predictions_path = os.path.join(result_dir, "predictions.json")
                # predictions = annotation_utils.load_predictions(predictions_path)

                # accuracies_iou_10 = []
                # accuracies_iou_50 = []
                # for image_name in annotations.keys():
                #     if len(annotations[image_name]["test_regions"]) > 0:
                #         annotated_boxes = annotations[image_name]["boxes"]
                #         predicted_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                #         accuracy_iou_10 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.10)
                #         accuracy_iou_50 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.50)
                        
                #         accuracies_iou_10.append(accuracy_iou_10)
                #         accuracies_iou_50.append(accuracy_iou_50)

                # test_set_accuracy_iou_10 = np.mean(accuracies_iou_10)
                # test_set_accuracy_iou_50 = np.mean(accuracies_iou_50)


                excel_path = os.path.join(result_dir, "metrics.xlsx")
                df = pd.read_excel(excel_path, sheet_name=0)
                # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                annotated_counts = df["Annotated Count"][df["Annotated Count"].notnull()]
                predicted_counts = df["Predicted Count"][df["Annotated Count"].notnull()]

                abs_dics = np.abs(np.array((predicted_counts - annotated_counts)))

                rep_abs_dics.extend(abs_dics.tolist())

            baseline_abs_dics.append(rep_abs_dics)

        # mean absolute difference in count for each image
        baseline_mean_abs_dics = np.mean(np.array(baseline_abs_dics), axis=0)

        # mean absolute difference in count across all images
        baseline_mean_abs_dic = np.mean(baseline_mean_abs_dics)

        print()
        print(baseline["removal_percentage"])
        print(baseline_abs_dics)
        print(baseline_mean_abs_dic)
        print(len(baseline_abs_dics[0]))
        print()

        results.append((
            baseline["removal_percentage"],
            baseline_mean_abs_dic
        ))


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(
        [x[0] for x in results], [x[1] for x in results]
    )

    ax.set_xlabel("Removal Percentage")
    ax.set_ylabel("Mean Absolute Difference in Count")

    out_path = os.path.join("eval_charts", out_dirname, "mean_abs_dic.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)



def create_id_ood_removal_plot(id_test_sets, ood_test_sets, baselines, out_dirname):

    test_sets = id_test_sets + ood_test_sets
    test_set_map = {
        "id": id_test_sets,
        "ood": ood_test_sets
    }

    mappings = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    result_map = {
        "id": [],
        "ood": []
    }
    total_min_diff = 10000000
    total_max_diff = -10000000
    for baseline in baselines:
        # rep_accuracies_iou_10 = []
        # rep_accuracies_iou_50 = []
        # rep_dics = []
        # for rep_num in range(1):

        model_name = baseline["model_name"] + "_rep_" + str(0)
        
        test_set_accuracies_iou_10 = []
        test_set_accuracies_iou_50 = []
        test_set_dics = []
        for k in test_set_map.keys():
            test_sets = test_set_map[k]

            for test_set in test_sets:

                # print(test_set)
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])



                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)

                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)

                accuracies_iou_10 = []
                accuracies_iou_50 = []
                for image_name in annotations.keys():
                    if len(annotations[image_name]["test_regions"]) > 0:
                        annotated_boxes = annotations[image_name]["boxes"]
                        predicted_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                        accuracy_iou_10 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.10)
                        accuracy_iou_50 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.50)
                        
                        accuracies_iou_10.append(accuracy_iou_10)
                        accuracies_iou_50.append(accuracy_iou_50)

                test_set_accuracy_iou_10 = np.mean(accuracies_iou_10)
                test_set_accuracy_iou_50 = np.mean(accuracies_iou_50)


                excel_path = os.path.join(result_dir, "metrics.xlsx")
                df = pd.read_excel(excel_path, sheet_name=0)
                # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                annotated_counts = df["Annotated Count"][df["Annotated Count"].notnull()]
                predicted_counts = df["Predicted Count"][df["Annotated Count"].notnull()]

                dics = np.array((predicted_counts - annotated_counts))

                min_diff = np.min(dics)
                max_diff = np.max(dics)
                if min_diff < total_min_diff:
                    total_min_diff = min_diff
                if max_diff > total_max_diff:
                    total_max_diff = max_diff

                test_set_dics.extend(dics.tolist())
                # rep_accuracies.append(rep_accuracy)

                test_set_accuracies_iou_10.append(test_set_accuracy_iou_10)
                test_set_accuracies_iou_50.append(test_set_accuracy_iou_50)


                # rep_accuracy_iou_10 = np.mean(test_set_accuracies_iou_10)
                # rep_accuracies_iou_10.append(rep_accuracy_iou_10)

                # rep_accuracy_iou_50 = np.mean(test_set_accuracies_iou_50)
                # rep_accuracies_iou_50.append(rep_accuracy_iou_50)

                # rep_abs_dic = np.mean(test_set_abs_dics)
                # rep_dics = test_set_dics #.append(rep_abs_dic)

            # if i == 1:
            # print(baseline["model_name"], rep_accuracies)

            baseline_accuracy_iou_10 = np.mean(test_set_accuracies_iou_10)
            baseline_accuracy_iou_50 = np.mean(test_set_accuracies_iou_50)
            # baseline_std = np.std(rep_accuracies)
            # baseline_abs_dic = np.mean(rep_abs_dics)
            baseline_dics = test_set_dics
            result_map[k].append(
                    # (baseline["model_label"], 
                    (baseline["removal_percentage"],
                    baseline_accuracy_iou_10, #overall_baseline_accuracy,
                    baseline_accuracy_iou_50,
                    baseline_dics))
        
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # ax = fig.add_axes([0.32, 0.05, 0.66, 0.9])

    # print(results)

    plt.suptitle("Removal effect")
    axs[0].set_title("IoU=0.5")
    axs[0].scatter([x[0] for x in result_map["id"]], [x[2] for x in result_map["id"]], label="In Domain", color=my_plot_colors[0])
    axs[0].plot([x[0] for x in result_map["id"]], [x[2] for x in result_map["id"]], color=my_plot_colors[0], linestyle="dotted")
    axs[0].scatter([x[0] for x in result_map["ood"]], [x[2] for x in result_map["ood"]], label="Out of Domain", color=my_plot_colors[1])
    axs[0].plot([x[0] for x in result_map["ood"]], [x[2] for x in result_map["ood"]], color=my_plot_colors[1], linestyle="dotted")
    
    axs[1].set_title("IoU=0.1")
    axs[1].scatter([x[0] for x in result_map["id"]], [x[1] for x in result_map["id"]], label="In Domain", color=my_plot_colors[0])
    axs[1].plot([x[0] for x in result_map["id"]], [x[1] for x in result_map["id"]], color=my_plot_colors[0], linestyle="dotted")
    axs[1].scatter([x[0] for x in result_map["ood"]], [x[1] for x in result_map["ood"]], label="Out of Domain", color=my_plot_colors[1])
    axs[1].plot([x[0] for x in result_map["ood"]], [x[1] for x in result_map["ood"]], color=my_plot_colors[1], linestyle="dotted")





    axs[0].legend()
    axs[1].legend()

    axs[0].set_ylabel("Image-Based Accuracy")
    axs[0].set_xlabel("Dilation Sigma (Pixels)")
    
    axs[1].set_ylabel("Image-Based Accuracy")
    axs[1].set_xlabel("Dilation Sigma (Pixels)")



    out_path = os.path.join("eval_charts", out_dirname, "id_ood_accuracy_plot.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)

    return

def create_dilation_plot(test_sets, baselines, out_dirname):

    mappings = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    results = []
    for baseline in baselines:
        # rep_accuracies_iou_10 = []
        # rep_accuracies_iou_50 = []
        # rep_abs_dics = []
        baseline_abs_dics = []
        for rep_num in range(1):

            model_name = baseline["model_name"] + "_rep_" + str(rep_num)
            
            # test_set_accuracies_iou_10 = []
            # test_set_accuracies_iou_50 = []
            # test_set_dics = []

            rep_abs_dics = []

            for test_set in test_sets:

                # print(test_set)
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])



                # annotations_path = os.path.join(result_dir, "annotations.json")
                # annotations = annotation_utils.load_annotations(annotations_path)

                # predictions_path = os.path.join(result_dir, "predictions.json")
                # predictions = annotation_utils.load_predictions(predictions_path)

                # accuracies_iou_10 = []
                # accuracies_iou_50 = []
                # for image_name in annotations.keys():
                #     if len(annotations[image_name]["test_regions"]) > 0:
                #         annotated_boxes = annotations[image_name]["boxes"]
                #         predicted_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                #         accuracy_iou_10 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.10)
                #         accuracy_iou_50 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.50)
                        
                #         accuracies_iou_10.append(accuracy_iou_10)
                #         accuracies_iou_50.append(accuracy_iou_50)

                # test_set_accuracy_iou_10 = np.mean(accuracies_iou_10)
                # test_set_accuracy_iou_50 = np.mean(accuracies_iou_50)


                excel_path = os.path.join(result_dir, "metrics.xlsx")
                df = pd.read_excel(excel_path, sheet_name=0)
                # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                annotated_counts = df["Annotated Count"][df["Annotated Count"].notnull()]
                predicted_counts = df["Predicted Count"][df["Annotated Count"].notnull()]

                abs_dics = np.abs(np.array((predicted_counts - annotated_counts)))

                rep_abs_dics.extend(abs_dics.tolist())

            baseline_abs_dics.append(rep_abs_dics)

        # mean absolute difference in count for each image
        baseline_mean_abs_dics = np.mean(np.array(baseline_abs_dics), axis=0)

        # mean absolute difference in count across all images
        baseline_mean_abs_dic = np.mean(baseline_mean_abs_dics)

        print()
        print(baseline["dilation_sigma"])
        print(baseline_abs_dics)
        print(baseline_mean_abs_dic)
        print(len(baseline_abs_dics[0]))
        print()

        results.append((
            baseline["dilation_sigma"],
            baseline_mean_abs_dic
        ))


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(
        [x[0] for x in results], [x[1] for x in results]
    )

    ax.set_xlabel("Dilation Sigma")
    ax.set_ylabel("Mean Absolute Difference in Count")

    out_path = os.path.join("eval_charts", out_dirname, "mean_abs_dic.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)



def create_id_ood_dilation_plot(id_test_sets, ood_test_sets, baselines, out_dirname):

    test_sets = id_test_sets + ood_test_sets
    test_set_map = {
        "id": id_test_sets,
        "ood": ood_test_sets
    }

    mappings = {}
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    result_map = {
        "id": [],
        "ood": []
    }
    total_min_diff = 10000000
    total_max_diff = -10000000
    for baseline in baselines:
        # rep_accuracies_iou_10 = []
        # rep_accuracies_iou_50 = []
        # rep_dics = []
        # for rep_num in range(1):

        model_name = baseline["model_name"] + "_rep_" + str(0)
        
        test_set_accuracies_iou_10 = []
        test_set_accuracies_iou_50 = []
        test_set_dics = []
        for k in test_set_map.keys():
            test_sets = test_set_map[k]

            for test_set in test_sets:

                # print(test_set)
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])



                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)

                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)

                accuracies_iou_10 = []
                accuracies_iou_50 = []
                for image_name in annotations.keys():
                    if len(annotations[image_name]["test_regions"]) > 0:
                        annotated_boxes = annotations[image_name]["boxes"]
                        predicted_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                        accuracy_iou_10 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.10)
                        accuracy_iou_50 = fine_tune_eval.get_accuracy(annotated_boxes, predicted_boxes, iou_thresh=0.50)
                        
                        accuracies_iou_10.append(accuracy_iou_10)
                        accuracies_iou_50.append(accuracy_iou_50)

                test_set_accuracy_iou_10 = np.mean(accuracies_iou_10)
                test_set_accuracy_iou_50 = np.mean(accuracies_iou_50)


                excel_path = os.path.join(result_dir, "metrics.xlsx")
                df = pd.read_excel(excel_path, sheet_name=0)
                # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                annotated_counts = df["Annotated Count"][df["Annotated Count"].notnull()]
                predicted_counts = df["Predicted Count"][df["Annotated Count"].notnull()]

                dics = np.array((predicted_counts - annotated_counts))

                min_diff = np.min(dics)
                max_diff = np.max(dics)
                if min_diff < total_min_diff:
                    total_min_diff = min_diff
                if max_diff > total_max_diff:
                    total_max_diff = max_diff

                test_set_dics.extend(dics.tolist())
                # rep_accuracies.append(rep_accuracy)

                test_set_accuracies_iou_10.append(test_set_accuracy_iou_10)
                test_set_accuracies_iou_50.append(test_set_accuracy_iou_50)


                # rep_accuracy_iou_10 = np.mean(test_set_accuracies_iou_10)
                # rep_accuracies_iou_10.append(rep_accuracy_iou_10)

                # rep_accuracy_iou_50 = np.mean(test_set_accuracies_iou_50)
                # rep_accuracies_iou_50.append(rep_accuracy_iou_50)

                # rep_abs_dic = np.mean(test_set_abs_dics)
                # rep_dics = test_set_dics #.append(rep_abs_dic)

            # if i == 1:
            # print(baseline["model_name"], rep_accuracies)

            baseline_accuracy_iou_10 = np.mean(test_set_accuracies_iou_10)
            baseline_accuracy_iou_50 = np.mean(test_set_accuracies_iou_50)
            # baseline_std = np.std(rep_accuracies)
            # baseline_abs_dic = np.mean(rep_abs_dics)
            baseline_dics = test_set_dics
            result_map[k].append(
                    # (baseline["model_label"], 
                    (baseline["dilation_sigma"],
                    baseline_accuracy_iou_10, #overall_baseline_accuracy,
                    baseline_accuracy_iou_50,
                    baseline_dics))
        
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # ax = fig.add_axes([0.32, 0.05, 0.66, 0.9])

    # print(results)

    plt.suptitle("-- effect")
    axs[0].set_title("IoU=0.5")
    axs[0].scatter([x[0] for x in result_map["id"]], [x[2] for x in result_map["id"]], label="In Domain", color=my_plot_colors[0])
    axs[0].plot([x[0] for x in result_map["id"]], [x[2] for x in result_map["id"]], color=my_plot_colors[0], linestyle="dotted")
    axs[0].scatter([x[0] for x in result_map["ood"]], [x[2] for x in result_map["ood"]], label="Out of Domain", color=my_plot_colors[1])
    axs[0].plot([x[0] for x in result_map["ood"]], [x[2] for x in result_map["ood"]], color=my_plot_colors[1], linestyle="dotted")
    
    axs[1].set_title("IoU=0.1")
    axs[1].scatter([x[0] for x in result_map["id"]], [x[1] for x in result_map["id"]], label="In Domain", color=my_plot_colors[0])
    axs[1].plot([x[0] for x in result_map["id"]], [x[1] for x in result_map["id"]], color=my_plot_colors[0], linestyle="dotted")
    axs[1].scatter([x[0] for x in result_map["ood"]], [x[1] for x in result_map["ood"]], label="Out of Domain", color=my_plot_colors[1])
    axs[1].plot([x[0] for x in result_map["ood"]], [x[1] for x in result_map["ood"]], color=my_plot_colors[1], linestyle="dotted")





    axs[0].legend()
    axs[1].legend()

    axs[0].set_ylabel("Image-Based Accuracy")
    axs[0].set_xlabel("Dilation Sigma (Pixels)")
    
    axs[1].set_ylabel("Image-Based Accuracy")
    axs[1].set_xlabel("Dilation Sigma (Pixels)")



    out_path = os.path.join("eval_charts", out_dirname, "id_ood_accuracy_plot.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)

    # return



    # end_point = max(abs(total_min_diff), abs(total_max_diff))
    # fig, axs = plt.subplots(len(results), 1, figsize=(10, 8))
    # # fig.suptitle("Methods for Merging Patch Predictions: Difference in Count (Predicted - Annotated)")
    # for i, result in enumerate(results):
    #     counts, bins = np.histogram(result[3], bins=2*end_point, range=(-end_point, end_point))
    #     axs[i].stairs(counts, bins, color=my_plot_colors[i], fill=True)
    #     # label=label_lookup[merging_prefix], 
    #     # axs[i].legend()
    #     props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    #     textstr = result[0] #label_lookup[merging_prefix]
    #     axs[i].text(0.985, 0.94, textstr, transform=axs[i].transAxes,
    #         verticalalignment='top', horizontalalignment="right", bbox=props, fontsize=12)
    #     # axs[i].set_title(textstr)

    #     # plt.locator_params(axis="y", nbins=4)
    #     # yticks = ticker.MaxNLocator(4)
    #     # axs[i].yaxis.set_major_locator(yticks)
    # plt.tight_layout()








    # # ax = fig.add_subplot(111)
    # # ax = fig.add_axes([0.32, 0.05, 0.66, 0.9])

    # # plt.scatter([x[0] for x in results], [x[3] for x in results])
    
    # # ax.set_ylabel("Mean Absolute Difference In Count")
    # # ax.set_xlabel("Dilation Sigma (Pixels)")


    # out_path = os.path.join("eval_charts", out_dirname, "hist_abs_dic_plot.png") #".svg")
    # out_dir = os.path.dirname(out_path)
    # os.makedirs(out_dir, exist_ok=True)
    # plt.savefig(out_path, dpi=600)



    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    lines = []
    x_vals = []
    for result in results:
        lines.append(result[3])
        x_vals.append(result[0])
    lines = np.array(lines).T
    print(lines.shape)

    for i in range(lines.shape[0]):
        ax.plot(x_vals, lines[i, :], linewidth=1, color="black", alpha=0.2)

    out_path = os.path.join("eval_charts", out_dirname, "lines_dic_plot.svg") #".svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.80])
    bp = ax.violinplot([x[3] for x in results], showmeans=True)

    xticklabels = [0]
    for x in results:
        xticklabels.append(x[0])
    ax.set_xticks(np.arange(0, len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    plt.axhline(y=0, linestyle="dotted", color="black", linewidth=1)
    

    plt.title("Difference In Count For Different Dilation Amounts")
    ax.set_ylabel("Difference in Count (Predicted - Annotated)")
    ax.set_xlabel("Dilation Amount ($\sigma$)")
    # plt.tight_layout()

    out_path = os.path.join("eval_charts", out_dirname, "dic_boxplot.svg") #".svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)









def create_eval_min_num_plot(test_sets, single_baselines, diverse_baselines, out_dirname):
    results = get_min_num_results(test_sets, single_baselines, diverse_baselines)

    for test_set_str in results:
        single_tuples = results[test_set_str]["single"]
        diverse_tuples = results[test_set_str]["diverse"]

        single_tuples.sort(key=lambda x: x[1])
        diverse_tuples.sort(key=lambda x: x[1])

        labels = []
        for single_tuple in single_tuples:
            labels.append(single_tuple[0])
        for diverse_tuple in diverse_tuples:
            labels.append(diverse_tuple[0])

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0.32, 0.05, 0.66, 0.9])

        ax.scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2)
        ax.scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2)
        for i, x in enumerate(single_tuples):
            ax.plot([x[1] - x[2], x[1] + x[2]], [i, i], color=my_plot_colors[1])

        for i, x in enumerate(diverse_tuples):
            ax.plot([x[1] - x[2], x[1] + x[2]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

        # if test_set_str == "overall":
        #     ax.scatter([x[2] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
        #     ax.scatter([x[2] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")
            
        #     ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
        #     ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")

        ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
        ax.set_yticklabels(labels)


        # ax.legend()
        ax.set_xlabel("Test Accuracy")
        ax.set_title("Effect of Training Set Diversity")

        # plt.tight_layout()

        out_path = os.path.join("eval_charts", out_dirname, test_set_str + ".png") #".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=600)



def predict_on_test_sets(test_sets, baselines):

    for test_set in test_sets:

        test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
        
        print("\n\nProcessing test_set: {}\n\n".format(test_set_image_set_dir))
        

        annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        image_names = []
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                image_names.append(image_name)

        metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)


        regions = []
        for image_name in image_names:
            regions.append([[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]])

        for baseline in baselines:

            print("switching to model")
            model_dir = os.path.join(test_set_image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": baseline["model_name"],
                "model_creator": baseline["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": test_set["username"],
                "farm_name": test_set["farm_name"],
                "field_name": test_set["field_name"],
                "mission_date": test_set["mission_date"]
            }


            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(1)
                if not os.path.exists(switch_req_path):
                    switch_processed = True

            
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": image_names,
                "regions": regions,
                "save_result": True,
                "results_name": baseline["model_name"], # "no_prune_" + baseline["model_name"],
                "results_message": ""
            }

            request_path = os.path.join(test_set_image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)



def test(test_sets, baselines, num_reps):

    for rep_num in range(num_reps):
        for test_set in test_sets:

            org_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
            
            print("\n\nprocessing test_set: {}\n\n".format(org_image_set_dir))
            

            annotations_path = os.path.join(org_image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            num_iterations = 1 #2
            image_names = list(annotations.keys())

            metadata_path = os.path.join(org_image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)
            # candidates = image_names.copy()
            # training_images = []
            # for iteration_number in range(num_iterations):

            #     candidates = []
            #     for image_name in annotations.keys():
            #         if len(annotations[image_name]["training_regions"]) == 0:
            #             candidates.append(image_name)

            #     chosen_images = random.sample(candidates, 5)

            #     for chosen_image in chosen_images:
            #         annotations[chosen_image]["training_regions"].append([0, 0, metadata["images"][chosen_image]["height_px"], metadata["images"][chosen_image]["width_px"]])
                
            #     training_images.append(chosen_images)

            if test_set["farm_name"] == "BlaineLake" and test_set["field_name"] == "Serhienko9S":
                training_images = [["8", "19", "21", "38", "44"]]
            if test_set["farm_name"] == "BlaineLake" and test_set["field_name"] == "Serhienko11":
                training_images = [["4", "11", "19", "24", "29"]]            
            elif test_set["farm_name"] == "BlaineLake" and test_set["field_name"] == "Serhienko15":
                training_images = [["11", "12", "18", "19", "29"]]
            elif test_set["farm_name"] == "SaskatoonEast" and test_set["field_name"] == "Stevenson5SW":
                training_images = [["22", "25", "27", "35", "40"]]

            
            
            


            regions = []
            for image_name in image_names:
                regions.append([[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]])
            # print("regions", regions)
            for baseline in baselines:

                baseline_username = test_set["username"]
                baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                baseline_mission_date = test_set["mission_date"]

                baseline_dir = os.path.join("usr", "data",
                                                        baseline_username, "image_sets",
                                                        baseline_farm_name,
                                                        baseline_field_name,
                                                        baseline_mission_date)
                
                
                        
                print("\n\nprocessing baseline: {}\n\n".format(baseline_dir))

                field_dir = os.path.join("usr", "data",
                                                        baseline_username, "image_sets",
                                                        baseline_farm_name,
                                                        baseline_field_name)                                       
                #                         method["image_set"]["farm_name"], method["image_set"]["field_name"], method["image_set"]["mission_date"])
                # field_dir = os.path.join("usr", "data", method["image_set"]["username"], "image_sets",
                #                             method["image_set"]["farm_name"], method["image_set"]["field_name"])


                if os.path.exists(field_dir):
                    raise RuntimeError("Field dir exists!")
                
                print("copying directory")
                os.makedirs(field_dir)
                shutil.copytree(org_image_set_dir, baseline_dir)

                print("switching to model")
                model_dir = os.path.join(baseline_dir, "model")
                switch_req_path = os.path.join(model_dir, "switch_request.json")
                switch_req = {
                    "model_name": baseline["model_name"],
                    "model_creator": baseline["model_creator"]
                }
                json_io.save_json(switch_req_path, switch_req)

                item = {
                    "username": baseline_username,
                    "farm_name": baseline_farm_name,
                    "field_name": baseline_field_name,
                    "mission_date": baseline_mission_date
                }


                switch_processed = False
                isa.process_switch(item)
                while not switch_processed:
                    print("Waiting for process switch")
                    time.sleep(10)
                    if not os.path.exists(switch_req_path):
                        switch_processed = True


                for iteration_number in range(num_iterations):

                    if iteration_number > 0:

                        annotations_path = os.path.join(baseline_dir, "annotations", "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        for image_name in training_images[iteration_number-1]:
                            annotations[image_name]["training_regions"].append(regions[0][0])

                        annotation_utils.save_annotations(annotations_path, annotations)

                        server.sch_ctx["training_queue"].enqueue(item)
                            

                        train_queue_size = server.sch_ctx["training_queue"].size()
                        print("train_queue_size", train_queue_size)
                        while train_queue_size > 0:
                            item = server.sch_ctx["training_queue"].dequeue()
                            print("running process_train")

                            re_enqueue = server.process_train(item)



                            if re_enqueue:
                                server.sch_ctx["training_queue"].enqueue(item)
                            train_queue_size = server.sch_ctx["training_queue"].size()


                
                    request_uuid = str(uuid.uuid4())
                    request = {
                        "request_uuid": request_uuid,
                        "start_time": int(time.time()),
                        "image_names": image_names,
                        "regions": regions,
                        "save_result": True,
                        "results_name": "fine_tune_" + str(iteration_number),
                        "results_message": ""
                    }

                    request_path = os.path.join(baseline_dir, "model", "prediction", 
                                                "image_set_requests", "pending", request_uuid + ".json")

                    json_io.save_json(request_path, request)
                    print("running process_predict")
                    server.process_predict(item)




def plot_my_combined_results_alt(test_sets, all_baselines, num_reps, xaxis_key):

    # all_baselines = {
    #     "org_baselines": baselines,
    #     "diverse_baselines": diverse_baselines
    # }

            
    for rep_num in range(num_reps):
        results = {}
        for k in all_baselines.keys():
        
            # direct_application_org_results = []
            # direct_application_diverse_results = []
            # fine_tune_org_results = []
            # fine_tune_diverse_results = []

            # results[k] = {
            #     "full_predictions_lst": [],
            #     "annotations_lst": [],
            # }
            results[k] = []

            # methods = []
            for baseline in all_baselines[k]:
                print(baseline)
            
                # label = k
                # results[k] = []
                predictions_lst = []
                annotations_lst = []
                assessment_images_lst = []
                all_dics = []
                test_set_accuracies = []
                for test_set in test_sets:
                    print("\t{}".format(test_set))

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

                    baseline_username = test_set["username"]
                    baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                    baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                    baseline_mission_date = test_set["mission_date"]

                    image_set_dir = os.path.join("usr", "data", baseline_username, "image_sets",
                                                 baseline_farm_name, baseline_field_name, baseline_mission_date)
                    results_dir = os.path.join(image_set_dir, "model", "results")

                    result_pairs = []
                    result_dirs = glob.glob(os.path.join(results_dir, "*"))
                    for result_dir in result_dirs:
                        request_path = os.path.join(result_dir, "request.json")
                        request = json_io.load_json(request_path)
                        end_time = request["end_time"]
                        result_pairs.append((result_dir, end_time))

                    result_pairs.sort(key=lambda x: x[1])

                    direct_application_result_dir = result_pairs[0][0]

                    annotations = annotation_utils.load_annotations(os.path.join(direct_application_result_dir, "annotations.json"))
                    # full_predictions_path = os.path.join(direct_application_result_dir, "full_predictions.json")
                    # full_predictions = json_io.load_json(full_predictions_path)
                    predictions_path = os.path.join(direct_application_result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)

                    if test_set_str in assessment_images_lookup:
                        assessment_images = assessment_images_lookup[test_set_str]
                    else:
                        assessment_images = list(annotations.keys())


                    accuracies = []
                    for image_name in assessment_images:
                        sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]

                        # abs_diff_in_count = abs((annotations[image_name]["boxes"]).size - (sel_pred_boxes).size)
                        # accuracies.append(abs_diff_in_count)
                        
                        accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
                        accuracies.append(accuracy)

                    predictions_lst.append(predictions)
                    annotations_lst.append(annotations)
                    assessment_images_lst.append(assessment_images)
                    test_set_accuracies.append(np.mean(accuracies))


                    # all_dics.extend(fine_tune_eval.get_dics(annotations, full_predictions, list(annotations.keys())))

                global_accuracy = np.mean(test_set_accuracies) #fine_tune_eval.get_global_accuracy_multiple_image_sets(annotations_lst, predictions_lst, assessment_images_lst)
                # ave_abs_dic = np.mean(np.abs(all_dics))

                results[k].append((baseline[xaxis_key], global_accuracy))


            fig = plt.figure(figsize=(8, 6))

            colors = {}
            color_list = ["red", "green", "blue", "purple", "orange", "grey", "pink"]
            c_index = 0
            for k in results.keys():
                # if k.endswith("_fine_tuned_on_5_images"):
                #     colors[k] = colors[k[:(-1) * len("_fine_tuned_on_5_images")]]
                # else:
                colors[k] = color_list[c_index]
                c_index += 1

            for k in results.keys():
                # if k.endswith("_fine_tuned_on_5_images"):
                #     marker = "x"
                # else:
                marker = "o"
                plt.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=colors[k], marker=marker, label=k, linestyle="dashed", linewidth=1)

            # label = "direct_application"
            # label = "random_images"
            # # label = "overlap_50%_direct_application"
            # plt.plot([x[0] for x in direct_application_org_results], [x[1] for x in direct_application_org_results], c="red", marker="o", label=label, linestyle="dashed", linewidth=1)
            # if len(fine_tune_org_results) > 0:
            #     label = "fine_tune_on_5"
            #     # label = "overlap_50%_fine_tune_on_5"
            #     plt.plot([x[0] for x in fine_tune_org_results], [x[1] for x in fine_tune_org_results], c="red", marker="x", label=label, linestyle="dashed", linewidth=1)

            # if len(diverse_baselines) > 0:
            #     label = "direct_application_diverse"
            #     label = "selected_patches"
            #     # label = "overlap_0%_direct_application"
            #     plt.plot([x[0] for x in direct_application_diverse_results], [x[1] for x in direct_application_diverse_results], c="blue", marker="o", label=label, linestyle="dashed", linewidth=1)
            #     if len(fine_tune_diverse_results) > 0:
            #         label = "fine_tune_on_5_diverse"
            #         # label = "overlap_0%_fine_tune_on_5"
            #         plt.plot([x[0] for x in fine_tune_diverse_results], [x[1] for x in fine_tune_diverse_results], c="blue", marker="x", label=label, linestyle="dashed", linewidth=1)

            plt.legend()
            plt.xlabel(xaxis_key)
            plt.ylabel("Accuracy") #Ave Abs DiC")

            test_set_str = "combined_test_sets" #test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
            # out_path = os.path.join("baseline_charts", "active_learning_comparison", test_set_str, "global", "accuracy", "rep_" + str(rep_num) + ".svg")
            out_path = os.path.join("baseline_charts", "fixed_epoch_comparison", test_set_str, "global", "accuracy", "redo_rep_" + str(rep_num) + ".svg")
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(out_path)


def plot_single_diverse_comparison(test_sets, single_baselines, diverse_baselines):

    # results = {}
    single_results = []
    diverse_results = []
    for rep_num in range(1):

        for (single_baseline, diverse_baseline) in zip(single_baselines, diverse_baselines):

            # results[k] = {}

            baselines = [single_baseline, diverse_baseline]


            # for model_type in ["single", "diverse"]:
            for j in range(len(baselines)):
                full_predictions_lst = []
                annotations_lst = []
                assessment_images_lst = []
                baseline = baselines[j]

                for test_set in test_sets:



                    # baseline = all_baseline_pairs[k][model_type]
                    baseline_username = test_set["username"]
                    baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                    baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                    baseline_mission_date = test_set["mission_date"]

                    image_set_dir = os.path.join("usr", "data", baseline_username, "image_sets",
                                                 baseline_farm_name, baseline_field_name, baseline_mission_date)
                    results_dir = os.path.join(image_set_dir, "model", "results")

                    result_pairs = []
                    result_dirs = glob.glob(os.path.join(results_dir, "*"))
                    for result_dir in result_dirs:
                        request_path = os.path.join(result_dir, "request.json")
                        request = json_io.load_json(request_path)
                        end_time = request["end_time"]
                        result_pairs.append((result_dir, end_time))

                    result_pairs.sort(key=lambda x: x[1])

                    direct_application_result_dir = result_pairs[0][0]

                    annotations = annotation_utils.load_annotations(os.path.join(direct_application_result_dir, "annotations.json"))
                    full_predictions_path = os.path.join(direct_application_result_dir, "full_predictions.json")
                    full_predictions = json_io.load_json(full_predictions_path)


                    full_predictions_lst.append(full_predictions)
                    annotations_lst.append(annotations)
                    assessment_images_lst.append(list(annotations.keys()))


                    # y = fine_tune_eval.get_global_accuracy(annotations, full_predictions, list(annotations.keys()))


                global_accuracy = fine_tune_eval.get_global_accuracy_multiple_image_sets(annotations_lst, full_predictions_lst, assessment_images_lst)
                # ave_abs_dic = np.mean(np.abs(all_dics))

                if j == 0:
                    single_results.append(global_accuracy)
                else:
                    diverse_results.append(global_accuracy)
                # results[k].append((baseline[xaxis_key], global_accuracy))

    
    # plt.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=colors[k], marker=marker, label=k, linestyle="dashed", linewidth=1)
    x_items = []
    for baseline in single_baselines:
        x_items.append(baseline["model_name"][len("fixed_epoch_"):len(baseline["model_name"])-len("_no_overlap")])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.30, 0.05, 0.65, 0.9]) 
    for i, (x1, x2) in enumerate(zip(single_results, diverse_results)):
        ax.plot([x1, x2], [i, i], color="black", linestyle="solid", alpha=0.5, linewidth=1, zorder=1)
    ax.scatter([x for x in single_results], np.arange(0, len(x_items)), color="red", label="Single Image Set", zorder=2)
    ax.scatter([x for x in diverse_results], np.arange(0, len(x_items)), color="blue", label="Diverse Random Selection", zorder=2)

    ax.set_yticks(np.arange(0, len(x_items)))
    ax.set_yticklabels(x_items) #, rotation=90, ha="right")


    ax.legend()
    ax.set_xlabel("Test Accuracy")

    # plt.tight_layout()
    # plt.ylabel("")

    # test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    out_path = os.path.join("baseline_charts", "single_diverse_comparisons", "plot.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)


assessment_images_lookup = {
    "erik Biggar Dennis3 2021-06-12": ["9", "12", "14", "22", "23", "24", "32", "34", "36", "37", "49"],
    "erik Biggar Dennis5 2021-06-12": ["1", "2", "3", "14", "17", "25"],
    "erik UNI CNH-DugoutROW 2022-05-30": ["UNI_CNH_May30_101", "UNI_CNH_May30_105", "UNI_CNH_May30_117", "UNI_CNH_May30_120", "UNI_CNH_May30_205", "UNI_CNH_May30_217", "UNI_CNH_May30_310", "UNI_CNH_May30_314", "UNI_CNH_May30_416"]
}

eval_assessment_images_lookup = {
    "eval Biggar Dennis3 2021-06-12": ["9", "12", "14", "22", "23", "24", "32", "34", "36", "37", "49"],
    "eval Biggar Dennis5 2021-06-12": ["1", "2", "3", "14", "17", "25"],
    "eval UNI CNH-DugoutROW 2022-05-30": ["UNI_CNH_May30_101", "UNI_CNH_May30_105", "UNI_CNH_May30_117", "UNI_CNH_May30_120", "UNI_CNH_May30_205", "UNI_CNH_May30_217", "UNI_CNH_May30_310", "UNI_CNH_May30_314", "UNI_CNH_May30_416"]
}
def plot_min_num_single_diverse_comparison(test_sets, single_baselines, diverse_baselines):

    # results = {}
    single_results = {"combined": []}
    diverse_results = {"combined": []}

    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        single_results[test_set_str] = []
        diverse_results[test_set_str] = []

    single_annotation_counts = []
    diverse_annotation_counts = []

    for rep_num in range(1):

        for i in range(2): #single_baseline in single_baselines:


            if i == 0:
                baselines = single_baselines
            else:
                baselines = diverse_baselines

            # results[k] = {}

            # baselines = [single_baseline, diverse_baseline]


            # for model_type in ["single", "diverse"]:
            for j in tqdm.trange(len(baselines), desc="Processing baselines"):
                predictions_lst = []
                annotations_lst = []
                assessment_images_lst = []
                baseline = baselines[j]
                baseline_accuracies = []
                # aps = []

                # baseline_model_path = os.path.join("usr", "data", baseline["model_creator"], 
                # "models", "available", "public", baseline["model_name"])
                # log_path = os.path.join(baseline_model_path, "log.json")
                # log = json_io.load_json(log_path)
                # add_num_training_patches(baseline)
                num_annotations = get_num_annotations_used_by_baseline(baseline)
                if i == 0:
                    single_annotation_counts.append(num_annotations)
                else:
                    diverse_annotation_counts.append(num_annotations)

                # total_true_positives = 0
                # total_false_positives = 0
                # total_false_negatives = 0                

                for test_set in test_sets:



                    # baseline = all_baseline_pairs[k][model_type]
                    baseline_username = test_set["username"]
                    baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                    baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                    baseline_mission_date = test_set["mission_date"]

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

                    image_set_dir = os.path.join("usr", "data", baseline_username, "image_sets",
                                                 baseline_farm_name, baseline_field_name, baseline_mission_date)
                    results_dir = os.path.join(image_set_dir, "model", "results")

                    result_pairs = []
                    result_dirs = glob.glob(os.path.join(results_dir, "*"))
                    for result_dir in result_dirs:
                        request_path = os.path.join(result_dir, "request.json")
                        request = json_io.load_json(request_path)
                        end_time = request["end_time"]
                        result_pairs.append((result_dir, end_time))

                    result_pairs.sort(key=lambda x: x[1])

                    direct_application_result_dir = result_pairs[0][0]


                    # excel_path = os.path.join(direct_application_result_dir, "metrics.xlsx")
                    # df = pd.read_excel(excel_path, sheet_name=0)
                    # total_true_positives += df["True Positives (IoU=.50, conf>.50)"].sum()
                    # total_false_positives += df["False Positives (IoU=.50, conf>.50)"].sum()
                    # total_false_negatives += df["False Negatives (IoU=.50, conf>.50)"].sum()
                    # print("total_true_positives", total_true_positives, df)


                    annotations = annotation_utils.load_annotations(os.path.join(direct_application_result_dir, "annotations.json"))
                    predictions_path = os.path.join(direct_application_result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)

                    # full_predictions_path = os.path.join(direct_application_result_dir, "full_predictions.json")
                    # full_predictions = annotation_utils.load_predictions(full_predictions_path)
                    if test_set_str in assessment_images_lookup:
                        assessment_images = assessment_images_lookup[test_set_str]
                    else:
                        assessment_images = list(annotations.keys())
                    
                    accuracies = []
                    for image_name in assessment_images:
                        sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]

                        # abs_diff_in_count = abs((annotations[image_name]["boxes"]).size - (sel_pred_boxes).size)
                        # accuracies.append(abs_diff_in_count)
                        
                        accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
                        accuracies.append(accuracy)

                    # global_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images)
                    if i == 0:
                        single_results[test_set_str].append(np.mean(accuracies)) #np.mean(accuracies)) #global_accuracy)
                    else:
                        diverse_results[test_set_str].append(np.mean(accuracies)) #np.mean(accuracies)) #global_accuracy)

                    predictions_lst.append(predictions)
                    annotations_lst.append(annotations)
                    assessment_images_lst.append(assessment_images)
                    baseline_accuracies.append(np.mean(accuracies)) #global_accuracy) #np.mean(accuracies))

                    # ap = fine_tune_eval.get_AP(annotations, predictions, iou_thresh=".50:.05:.95")
                    # aps.append(ap)


                    # y = fine_tune_eval.get_global_accuracy(annotations, full_predictions, list(annotations.keys()))


                # global_accuracy = fine_tune_eval.get_global_accuracy_multiple_image_sets(annotations_lst, predictions_lst, assessment_images_lst)

                # ave_abs_dic = np.mean(np.abs(all_dics))
                # global_accuracy = total_true_positives / (total_true_positives + total_false_positives + total_false_negatives)
                
                

                if i == 0:
                    single_results["combined"].append(np.mean(baseline_accuracies)) #np.mean(baseline_accuracies)) #global_accuracy)
                else:
                    diverse_results["combined"].append(np.mean(baseline_accuracies)) #np.mean(baseline_accuracies)) #global_accuracy)
                # results[k].append((baseline[xaxis_key], global_accuracy))

    print("single_annotation_counts", single_annotation_counts)
    print("diverse_annotation_counts", diverse_annotation_counts)
    
    single_labels = []
    for baseline in single_baselines:
        single_labels.append(baseline["model_name"][len("fixed_epoch_min_num_diverse_set_of_1_match_"):len(baseline["model_name"])-len("_no_overlap")])
    single_labels = np.array(single_labels)

    diverse_labels = []
    for baseline in diverse_baselines:
        diverse_labels.append("diverse") #baseline["model_name"][len("fixed_epoch_"):len(baseline["model_name"])-len("_no_overlap")])
    diverse_labels = np.array(diverse_labels)

    for test_set_str in single_results.keys():
        print(test_set_str)
        print(single_results[test_set_str])
        print()
        print(diverse_results[test_set_str])
        print()
        print(single_labels)
        print()
        print("---")
        print()

        test_set_single_results = np.array(single_results[test_set_str])
        test_set_single_labels = np.copy(single_labels)
        inds = np.argsort(test_set_single_results)
        print("inds", inds)
        test_set_single_results = test_set_single_results[inds]
        test_set_single_labels = test_set_single_labels[inds]

        single_annotation_counts_plt = np.copy(np.array(single_annotation_counts))
        single_annotation_counts_plt = single_annotation_counts_plt[inds]
        print(single_annotation_counts_plt)


        test_set_diverse_results = np.array(diverse_results[test_set_str])
        test_set_diverse_labels = np.copy(diverse_labels)
        inds = np.argsort(test_set_diverse_results)
        test_set_diverse_results = test_set_diverse_results[inds]
        test_set_diverse_labels = test_set_diverse_labels[inds]



        diverse_annotation_counts_plt = np.copy(np.array(diverse_annotation_counts))
        diverse_annotation_counts_plt = diverse_annotation_counts_plt[inds]


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0.30, 0.05, 0.65, 0.9]) 
        # for i, (x1, x2) in enumerate(zip(single_results, diverse_results)):
        #     ax.plot([x1, x2], [i, i], color="black", linestyle="solid", alpha=0.5, linewidth=1, zorder=1)

        ax.scatter([x for x in test_set_single_results], np.arange(0, len(test_set_single_labels)), color="red", label="Single Image Set", zorder=2)
        ax.scatter([x for x in test_set_diverse_results], np.arange(len(test_set_single_labels), len(test_set_single_labels)+len(test_set_diverse_labels)), color="blue", label="Diverse Random Selection", zorder=2)

        ax.set_yticks(np.arange(0, len(test_set_single_labels)+len(test_set_diverse_labels)))
        ax.set_yticklabels(np.concatenate([test_set_single_labels, test_set_diverse_labels])) #, rotation=90, ha="right")

        # ax.scatter([x for x in test_set_single_results], single_annotation_counts_plt, color="red", label="Single Image Set", zorder=2)
        # ax.scatter([x for x in test_set_diverse_results], diverse_annotation_counts_plt, color="blue", label="Diverse Random Selection", zorder=2)




        # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]) 
        # ax.scatter([x[0] for x in single_results], [x[1] for x in single_results], color="red", label="Single Image Set", zorder=2)
        # ax.scatter([x[0] for x in diverse_results], [x[1] for x in diverse_results], color="blue", label="Diverse Random Selection", zorder=2)



        ax.legend()
        ax.set_xlabel("Test Accuracy")


        # ax.set_xlabel("Number of Annotations Used")
        # ax.set_ylabel("Test Accuracy")

        out_path = os.path.join("baseline_charts", "single_diverse_comparisons", test_set_str + ".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)



# def get_vendi_diversity(model_dir):


#     log_path = os.path.join(model_dir, "log.json")
#     log = json_io.load_json(log_path)

#     taken_patches = {}
#     for image_set in log["image_sets"]:

#         username = image_set["username"]
#         farm_name = image_set["farm_name"]
#         field_name = image_set["field_name"]
#         mission_date = image_set["mission_date"]

#         image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
#         taken_patches[image_set_str] = {}
#         image_set_dir = os.path.join("usr", "data", 
#                                     username, "image_sets",
#                                     farm_name,
#                                     field_name,
#                                     mission_date)
        
#         annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#         annotations = annotation_utils.load_annotations(annotations_path)

#         metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
#         metadata = json_io.load_json(metadata_path)

#         try:
#             patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
#         except Exception as e:
#             patch_size = 416
#         if "patch_overlap_percent" in image_set:
#             patch_overlap_percent = image_set["patch_overlap_percent"]
#         else:
#             patch_overlap_percent = 50

#         if "taken_regions" in image_set:
#             for image_name in image_set["taken_regions"].keys():
#                 taken_patches[image_set_str][image_name] = []
#                 image_h = metadata["images"][image_name]["height_px"]
#                 image_w = metadata["images"][image_name]["width_px"]
#                 for region in image_set["taken_regions"][image_name]:

#                     if region[2] != image_h and region[3] != image_w:
#                         taken_patches[image_set_str][image_name].append(region)

#         else:

#             for image_name in annotations.keys():
#                 if len(annotations[image_name]["test_regions"]) > 0:
#                     taken_patches[image_set_str][image_name] = []
#                     image_h = metadata["images"][image_name]["height_px"]
#                     image_w = metadata["images"][image_name]["width_px"]
#                     for i in range(0, image_h, patch_size):
#                         for j in range(0, image_w, patch_size):
#                             taken_patches[image_set_str][image_name].append([i, j, i+patch_size, j+patch_size])
            

#     patch_arrays = []
#     for image_set_str in taken_patches:

#         pieces = image_set_str.split(" ")
#         username = pieces[0]
#         farm_name = pieces[1]
#         field_name = pieces[2]
#         mission_date = pieces[3]

#         image_set_dir = os.path.join("usr", "data", 
#                                     username, "image_sets",
#                                     farm_name,
#                                     field_name,
#                                     mission_date)


#         for image_name in taken_patches[image_set_str]:

#             image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
#             image = Image(image_path)

#             image_array = image.load_image_array()

#             for patch_coords in taken_patches[image_set_str][image_name]:

#                 patch_array = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
#                 patch_arrays.append(patch_array)





#     batch_size = 256

#     patch_arrays = np.array(patch_arrays)
#     num_patches = patch_arrays.size
#     print("processing {} patches".format(num_patches))

#     # from models.yolov4.yolov4_image_set_driver import create_default_config
#     # from models.common import model_keys
#     # config = create_default_config()
#     # model_keys.add_general_keys(config)
#     # model_keys.add_specialized_keys(config)

#     # from models.yolov4 import yolov4

#     # model = yolov4.YOLOv4TinyBackbone(config, max_pool=True)
#     # input_shape = (256, *(config["arch"]["input_image_shape"]))
#     # model.build(input_shape=input_shape)
#     # model.load_weights(os.path.join("usr", "data", "erik", "models", "available", "public", 
#     #                                 "fixed_epoch_set_of_27_no_overlap", "weights.h5"), by_name=False)

#     weights = 'imagenet'
#     model = tf.keras.applications.InceptionV3( #101( #ResNet50(
#         weights=weights,
#         include_top=False, 
#         input_shape=[None, None, 3],
#         pooling="max"
#     )

#     # if extraction_type == "box_patches":
#     #     input_image_shape = np.array([150, 150, 3])
#     # else:
#     input_image_shape = [416, 416] #config.arch["input_image_shape"]

#     all_features = []
#     for i in tqdm.trange(0, num_patches, batch_size):
#         batch_patches = []
#         for j in range(i, min(num_patches, i+batch_size)):
#             patch = tf.convert_to_tensor(patch_arrays[j], dtype=tf.float32)
#             patch = tf.image.resize(images=patch, size=input_image_shape[:2])
#             # vec = tf.reshape(patch, [-1])

#             # all_features.append(vec)
#             batch_patches.append(patch)
#         batch_patches = tf.stack(values=batch_patches, axis=0)
        
#         features = model.predict(batch_patches)
#         for f in features:
#             f = f.flatten()
#             all_features.append(f)

#     all_features = np.array(all_features)
#     print("calculating vendi score...")
#     score = vendi.score(all_features, cosine)
#     print("score is {}".format(score))

#     # print("shape of features matrix is {}".format(all_features.shape))
#     # print("calculating similarity matrix...")
#     # sim_mat = np.zeros(shape=(all_features.shape[0], all_features.shape[0]))
#     # for i in range(all_features.shape[0]):
#     #     for j in range(all_features.shape[0]):
#     #         if i == j:
#     #             sim_mat[i][j] = 1
#     #         elif i > j:
#     #             sim_mat[i][j] = sim_mat[j][i]
#     #         else:
#     #             sim = cosine(all_features[i], all_features[j])
#     #             sim_mat[i][j] = sim
#     # # sim_mat = cosine(all_features, all_features)
#     # print("sim_mat is: {}".format(sim_mat))
#     # sim_mat = sim_mat / all_features.shape[0]
#     # print("calculating eigenvalues...")
#     # w, _ = np.linalg.eig(sim_mat)
#     # print("eigenvalues are: {}".format(w))
#     # print("calculating entropy...")
#     # ent = entropy(w)
#     # print("entropy is {}".format(ent))
#     # print("calculating vendi score")
#     # score = m.exp(ent)
#     # print("score is {}".format(score))

#     return score


def plot_my_results_alt(test_sets, all_baselines, num_reps, include_fine_tune=False):

    # all_baselines = {
    #     "org_baselines": baselines,
    #     "diverse_baselines": diverse_baselines
    # }


    for rep_num in range(num_reps):

        for test_set in test_sets:
            # direct_application_org_results = []
            # direct_application_diverse_results = []
            # fine_tune_org_results = []
            # fine_tune_diverse_results = []
            results = {}

            # methods = []
            for k in all_baselines.keys():
                # label = k
                results[k] = []
                for baseline in all_baselines[k]:
                    baseline_username = test_set["username"]
                    baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                    baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                    baseline_mission_date = test_set["mission_date"]

                    image_set_dir = os.path.join("usr", "data", baseline_username, "image_sets",
                                                 baseline_farm_name, baseline_field_name, baseline_mission_date)
                    results_dir = os.path.join(image_set_dir, "model", "results")

                    result_pairs = []
                    result_dirs = glob.glob(os.path.join(results_dir, "*"))
                    for result_dir in result_dirs:
                        request_path = os.path.join(result_dir, "request.json")
                        request = json_io.load_json(request_path)
                        end_time = request["end_time"]
                        result_pairs.append((result_dir, end_time))

                    result_pairs.sort(key=lambda x: x[1])

                    direct_application_result_dir = result_pairs[0][0]

                    annotations = annotation_utils.load_annotations(os.path.join(direct_application_result_dir, "annotations.json"))
                    full_predictions_path = os.path.join(direct_application_result_dir, "full_predictions.json")
                    full_predictions = json_io.load_json(full_predictions_path)

                    y = fine_tune_eval.get_global_accuracy(annotations, full_predictions, list(annotations.keys()))
                    # y = np.mean(np.abs(fine_tune_eval.get_dics(annotations, full_predictions, list(annotations.keys()))))


                    x = baseline["num_training_sets"]
                    # x = baseline["num_training_patches"]
                    # y = global_accuracy
                    # c = "red" if k == "org_baselines" else "blue"
                    # methods.append()
                    # if k == "org_baselines":
                    #     direct_application_org_results.append((x, y))
                    # else:
                    #     direct_application_diverse_results.append((x, y))
                    results[k].append((x, y))

                    if include_fine_tune:
                        if len(result_pairs) > 1:
                            fine_tune_result_dir = result_pairs[1][0]

                            annotations = annotation_utils.load_annotations(os.path.join(fine_tune_result_dir, "annotations.json"))
                            full_predictions_path = os.path.join(fine_tune_result_dir, "full_predictions.json")
                            full_predictions = json_io.load_json(full_predictions_path)

                            global_accuracy = fine_tune_eval.get_global_accuracy(annotations, full_predictions, list(annotations.keys()))

                            x = baseline["num_training_sets"] #patches"]
                            y = global_accuracy
                            # c = "red" if k == "org_baselines" else "blue"
                            # methods.append()
                            # fine_tune_results.append((x, y, c))

                            fine_tune_k = k + "_fine_tuned_on_5_images"
                            if fine_tune_k not in results:
                                results[fine_tune_k] = []


                            results[fine_tune_k].append((x, y))

                            # if k == "org_baselines":
                            #     fine_tune_org_results.append((x, y))
                            # else:
                            #     fine_tune_diverse_results.append((x, y))



                        # method = {
                        #     "image_set":d {
                        #         "username": baseline_username,
                        #         "farm_name": baseline_farm_name,
                        #         "field_name": baseline_field_name,
                        #         "mission_date": baseline_mission_date
                        #     },
                        #     "methodd_label": baseline["num_training_sets"]
                        # }
                        # methods.append(
                        #     method
                        # )
                
            # test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
            # print(direct_application_org_results)
            # print(fine_tune_org_results)
            # print(direct_application_diverse_results)
            # print(fine_tune_diverse_results)

            fig = plt.figure(figsize=(8, 6))

            colors = {}
            color_list = ["red", "green", "blue", "purple", "orange", "grey", "pink"]
            c_index = 0
            for k in results.keys():
                if k.endswith("_fine_tuned_on_5_images"):
                    colors[k] = colors[k[:(-1) * len("_fine_tuned_on_5_images")]]
                else:
                    colors[k] = color_list[c_index]
                    c_index += 1

            for k in results.keys():
                if k.endswith("_fine_tuned_on_5_images"):
                    marker = "x"
                else:
                    marker = "o"
                plt.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=colors[k], marker=marker, label=k, linestyle="dashed", linewidth=1)

            # label = "direct_application"
            # label = "random_images"
            # # label = "overlap_50%_direct_application"
            # plt.plot([x[0] for x in direct_application_org_results], [x[1] for x in direct_application_org_results], c="red", marker="o", label=label, linestyle="dashed", linewidth=1)
            # if len(fine_tune_org_results) > 0:
            #     label = "fine_tune_on_5"
            #     # label = "overlap_50%_fine_tune_on_5"
            #     plt.plot([x[0] for x in fine_tune_org_results], [x[1] for x in fine_tune_org_results], c="red", marker="x", label=label, linestyle="dashed", linewidth=1)

            # if len(diverse_baselines) > 0:
            #     label = "direct_application_diverse"
            #     label = "selected_patches"
            #     # label = "overlap_0%_direct_application"
            #     plt.plot([x[0] for x in direct_application_diverse_results], [x[1] for x in direct_application_diverse_results], c="blue", marker="o", label=label, linestyle="dashed", linewidth=1)
            #     if len(fine_tune_diverse_results) > 0:
            #         label = "fine_tune_on_5_diverse"
            #         # label = "overlap_0%_fine_tune_on_5"
            #         plt.plot([x[0] for x in fine_tune_diverse_results], [x[1] for x in fine_tune_diverse_results], c="blue", marker="x", label=label, linestyle="dashed", linewidth=1)

            plt.legend()
            plt.xlabel("Number of Training Sets")
            plt.ylabel("Accuracy")

            test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
            out_path = os.path.join("baseline_charts", "fixed_epoch_comparison", test_set_str, "global", "accuracy", "rep_" + str(rep_num) + ".svg")
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(out_path)





def plot_my_results(test_sets, baselines, num_reps):

    for rep_num in range(num_reps):

        for test_set in test_sets:
            methods = []
            for baseline in baselines:
                baseline_username = test_set["username"]
                baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                baseline_mission_date = test_set["mission_date"]
                method = {
                    "image_set": {
                        "username": baseline_username,
                        "farm_name": baseline_farm_name,
                        "field_name": baseline_field_name,
                        "mission_date": baseline_mission_date
                    },
                    "method_label": baseline["num_training_sets"]
                }
                methods.append(
                    method
                )
                
            test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]

            fine_tune_eval.create_global_comparison(methods, 
                                                    "accuracy", 
                                                    os.path.join("baseline_charts", "comparisons", test_set_str, "global", "accuracy", "rep_" + str(rep_num) + ".svg"),
                                                    xpositions="num_annotations", 
                                                    colors=None)
            fine_tune_eval.create_global_comparison(methods, 
                                                    "AP (IoU=.50)", 
                                                    os.path.join("baseline_charts", "comparisons", test_set_str, "global", "AP (IoU=.50)", "rep_" + str(rep_num) + ".svg"),
                                                    xpositions="num_annotations", 
                                                    colors=None)



def get_num_annotations_used_by_baseline(baseline):
    model_dir = os.path.join("usr", "data", baseline["model_creator"], "models", "available", "public", baseline["model_name"])
    log_path = os.path.join(model_dir, "log.json")
    log = json_io.load_json(log_path)

    num_annotations = 0

    for image_set in log["image_sets"]:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        if "taken_regions" in image_set:
            for image_name in image_set["taken_regions"].keys():
                num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], image_set["taken_regions"][image_name]).size
        else:
            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    num_annotations += len(annotations[image_name]["boxes"])
    return num_annotations


def add_num_training_patches(baselines):
    for baseline in baselines:
        model_dir = os.path.join("usr", "data", baseline["model_creator"], "models", "available", "public", baseline["model_name"])
        log_path = os.path.join(model_dir, "log.json")
        log = json_io.load_json(log_path)
        total_num_patches = 0
        for image_set in log["image_sets"]:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            image_set_dir = os.path.join("usr", "data", 
                                        username, "image_sets",
                                        farm_name,
                                        field_name,
                                        mission_date)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            try:
                patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            except Exception as e:
                patch_size = 416
            if "patch_overlap_percent" in image_set:
                patch_overlap_percent = image_set["patch_overlap_percent"]
            else:
                patch_overlap_percent = 50

            if "taken_regions" in image_set:
                for image_name in image_set["taken_regions"].keys():
                    for region in image_set["taken_regions"][image_name]:

                        region_width = region[3] - region[1]
                        region_height = region[2] - region[0]

                        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

                        incr = patch_size - overlap_px
                        w_covered = max(region_width - patch_size, 0)
                        num_w_patches = m.ceil(w_covered / incr) + 1

                        h_covered = max(region_height - patch_size, 0)
                        num_h_patches = m.ceil(h_covered / incr) + 1

                        num_patches = num_w_patches * num_h_patches

                        total_num_patches += num_patches #len(image_set["taken_regions"][image_name])
            else:
                # annotations = image_set_info[image_set_str]["annotations"]
                # image_shape = image_set_info[image_set_str]["image_shape"]
                # image_height = metadata["images"][list(annotations.keys())[0]]["height_px"]
                # image_width = metadata["images"][list(annotations.keys())[0]]["width_px"]
                # image_height = image_shape[0]
                # image_width = image_shape[1]
                
                num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=patch_overlap_percent)
                
                for image_name in annotations.keys():
                    if len(annotations[image_name]["test_regions"]) > 0:
                        total_num_patches += num_patches_per_image


        baseline["num_training_patches"] = total_num_patches
                        

def check(test_sets, all_baselines, num_reps):
            
    for rep_num in range(num_reps):
        results = {}
        for k in all_baselines.keys():
            results[k] = []

            for baseline in all_baselines[k]:
                print(baseline)

                full_predictions_lst = []
                annotations_lst = []
                assessment_images_lst = []
                all_dics = []
                for test_set in test_sets:
                    print("\t{}".format(test_set))
                    baseline_username = test_set["username"]
                    baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                    baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                    baseline_mission_date = test_set["mission_date"]

                    image_set_dir = os.path.join("usr", "data", baseline_username, "image_sets",
                                                 baseline_farm_name, baseline_field_name, baseline_mission_date)
                    results_dir = os.path.join(image_set_dir, "model", "results")

                    result_pairs = []
                    result_dirs = glob.glob(os.path.join(results_dir, "*"))

                    if len(result_dirs) > 0:
                        print("\t\t...ok")
                    else:
                        print("\t\t...not ok")
                    # for result_dir in result_dirs:
                    #     request_path = os.path.join(result_dir, "request.json")
                    #     request = json_io.load_json(request_path)
                    #     end_time = request["end_time"]
                    #     result_pairs.append((result_dir, end_time))

                    # result_pairs.sort(key=lambda x: x[1])


                    # direct_application_result_dir = result_pairs[0][0]



def copy_to_eval_2(fixed_epoch_fixed_patch_num_baselines):

    for baseline in fixed_epoch_fixed_patch_num_baselines:
        patch_num = baseline["model_name"][len("fixed_epoch_fixed_patch_num_"):len(baseline["model_name"])-len("_no_overlap")]
        new_model_name = "set_of_27_" + patch_num + "_patches_rep_0" 

        # "model_name": "fixed_epoch_fixed_patch_num_250_no_overlap",
        # "model_creator": "erik",
        model_path = os.path.join("usr", "data", "erik", "models", "available", "public", baseline["model_name"])

        eval_model_path = os.path.join("usr", "data", "eval", "models", "available", "public", new_model_name)
        shutil.copytree(model_path, eval_model_path)
        log_path = os.path.join(eval_model_path, "log.json")
        log = json_io.load_json(log_path)
        log["model_name"] = new_model_name
        log["model_creator"] = "eval"
        json_io.save_json(log_path, log)


def copy_to_eval(min_num_baselines):
    for min_num_baseline in min_num_baselines:
        model_path = os.path.join("usr", "data", "erik", "models", "available", "public", min_num_baseline["model_name"])
        image_set_name = min_num_baseline["model_name"][len("fixed_epoch_min_num_diverse_set_of_1_match_"):len(min_num_baseline["model_name"])-len("_no_overlap")]
        # pieces = image_set_name.split("_")
        # mission_date = pieces[-1]
        # field_name = pieces[-2]
        # if len(pieces) == 3:
        #     farm_name = pieces[-3]
        # else:
        #     farm_name = pieces[-4] + "_" + pieces[-3]

        new_model_name = image_set_name + "_630_patches_rep_0"
        # new_model_name = "set_of_27_630_patches_rep_1"

        eval_model_path = os.path.join("usr", "data", "eval", "models", "available", "public", new_model_name)
        shutil.copytree(model_path, eval_model_path)
        log_path = os.path.join(eval_model_path, "log.json")
        log = json_io.load_json(log_path)
        log["model_name"] = new_model_name
        log["model_creator"] = "eval"
        json_io.save_json(log_path, log)


def compare_exg_repl_patches(baseline, exg_baseline, out_dir):

    baseline_log_path = os.path.join("usr", "data", "eval", "models", "available", "public", baseline, "log.json")
    baseline_log = json_io.load_json(baseline_log_path)
    
    exg_baseline_log_path = os.path.join("usr", "data", "eval", "models", "available", "public", exg_baseline, "log.json")
    exg_baseline_log = json_io.load_json(exg_baseline_log_path)

    taken = {}
    for image_set in baseline_log["image_sets"]:
        image_set_str = image_set["username"] + ":" + image_set["farm_name"] + ":" + image_set["field_name"] + ":" + image_set["mission_date"]
        taken[image_set_str] = image_set["taken_regions"]
    
    exg_taken = {}
    for image_set in exg_baseline_log["image_sets"]:
        image_set_str = image_set["username"] + ":" + image_set["farm_name"] + ":" + image_set["field_name"] + ":" + image_set["mission_date"]
        exg_taken[image_set_str] = image_set["taken_regions"]

    unique_taken = {}
    for image_set_str in taken.keys():
        for image_name in taken[image_set_str].keys():
            for patch_coords in taken[image_set_str][image_name]:
                if image_set_str not in exg_taken or image_name not in exg_taken[image_set_str] or patch_coords not in exg_taken[image_set_str][image_name]:

                    if image_set_str not in unique_taken:
                        unique_taken[image_set_str] = {}
                    if image_name not in unique_taken[image_set_str]:
                        unique_taken[image_set_str][image_name] = []
                    unique_taken[image_set_str][image_name].append(patch_coords)

    unique_exg_taken = {}
    for image_set_str in exg_taken.keys():
        for image_name in exg_taken[image_set_str].keys():
            for patch_coords in exg_taken[image_set_str][image_name]:
                if image_set_str not in taken or image_name not in taken[image_set_str] or patch_coords not in taken[image_set_str][image_name]:

                    if image_set_str not in unique_exg_taken:
                        unique_exg_taken[image_set_str] = {}
                    if image_name not in unique_exg_taken[image_set_str]:
                        unique_exg_taken[image_set_str][image_name] = []
                    unique_exg_taken[image_set_str][image_name].append(patch_coords)

    unique_out_dir = os.path.join(out_dir, "unique_original")
    os.makedirs(unique_out_dir, exist_ok=True)
    for image_set_str in unique_taken.keys():
        pieces = image_set_str.split(":")
        image_set_dir = os.path.join("usr", "data", pieces[0], "image_sets", pieces[1], pieces[2], pieces[3])
        print(image_set_dir)
        for image_name in unique_taken[image_set_str].keys():
            image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
            # print(image_path)
            image = Image(image_path)
            image_array = image.load_image_array()

            for patch_coords in unique_taken[image_set_str][image_name]:
                out_path = os.path.join(unique_out_dir, str(uuid.uuid4()) + ".png")
                patch = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3], :]
                # print(patch.shape)
                cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    unique_exg_out_dir = os.path.join(out_dir, "unique_exg_repl")
    os.makedirs(unique_exg_out_dir, exist_ok=True)
    for image_set_str in unique_exg_taken.keys():
        pieces = image_set_str.split(":")
        image_set_dir = os.path.join("usr", "data", pieces[0], "image_sets", pieces[1], pieces[2], pieces[3])
        for image_name in unique_exg_taken[image_set_str].keys():
            image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
            image = Image(image_path)
            image_array = image.load_image_array()

            for patch_coords in unique_exg_taken[image_set_str][image_name]:
                out_path = os.path.join(unique_exg_out_dir, str(uuid.uuid4()) + ".png")
                patch = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3], :]
                cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))



def run():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.basicConfig(level=logging.INFO)


    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()

    overlap_baselines = [

        {
            "model_name": "MORSE_Nasser_2022-05-27",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "set_of_3",
            "model_creator": "kaylie",
            "num_training_sets": 3
        },
        {
            "model_name": "set_of_6",
            "model_creator": "kaylie",
            "num_training_sets": 6
        },
        {
            "model_name": "set_of_12",
            "model_creator": "kaylie",
            "num_training_sets": 12
        },
        {
            "model_name": "set_of_18",
            "model_creator": "kaylie",
            "num_training_sets": 18
        },
        # {
        #     "model_name": "p2irc_symposium_2022",
        #     "model_creator": "kaylie",
        #     "num_training_sets": 16
        # },
        {
            "model_name": "all_stages",
            "model_creator": "kaylie",
            "num_training_sets": 27
        }
    ]







    no_overlap_baselines = [

        {
            "model_name": "set_of_1_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "set_of_3_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 3
        },
        {
            "model_name": "set_of_6_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 6
        },
        {
            "model_name": "set_of_12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 12
        },
        {
            "model_name": "set_of_18_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 18
        },
        {
            "model_name": "set_of_27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 27
        }
    ]

    diverse_baselines = [

        {
            "model_name": "diverse_set_of_27_match_1_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "diverse_set_of_27_match_3_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 3
        },
        {
            "model_name": "diverse_set_of_27_match_6_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 6
        },
        {
            "model_name": "diverse_set_of_27_match_12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 12
        },
        {
            "model_name": "diverse_set_of_27_match_18_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 18
        },
        {
            "model_name": "diverse_set_of_27_match_27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 27
        }
    ]

    CottonWeedDet12_baselines = [
        {
            "model_name": "MORSE_Nasser_2022-05-27_and_CottonWeedDet12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "set_of_3_and_CottonWeedDet12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 3
        }
    ]

    weed_ai_10000_baselines = [
        {
            "model_name": "MORSE_Nasser_2022-05-27_and_10000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        },
    ]
    weed_ai_20000_baselines = [
        {
            "model_name": "MORSE_Nasser_2022-05-27_and_20000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        }
    ]

    fixed_weed_ai_10000_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_and_10000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        }
    ]

    fixed_weed_ai_20000_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_and_20000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        }
    ]

    fixed_weed_ai_30000_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_and_30000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        }
    ]

    fixed_weed_ai_40000_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_and_40000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        }
    ]

    fixed_weed_ai_50000_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_and_50000_weed",
            "model_creator": "erik",
            "num_training_sets": 1
        }
    ]

    active_baselines = []
    for i in range(0, 13):
        active_baselines.append({
            "model_name": "selected_patches_" + str(i),
            "model_creator": "erik"
        })
    
    random_image_baselines = []
    for i in range(0, 11):
        random_image_baselines.append({
            "model_name": "random_images_" + str(i),
            "model_creator": "erik"
        })

    fixed_epoch_no_overlap_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_set_of_3_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 3
        },
        {
            "model_name": "fixed_epoch_set_of_6_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 6
        },
        {
            "model_name": "fixed_epoch_set_of_12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 12
        },
        {
            "model_name": "fixed_epoch_set_of_18_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 18
        },
        {
            "model_name": "fixed_epoch_set_of_27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 27
        },
    ]

    fixed_epoch_diverse_baselines = [
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_1_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_3_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 3
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_6_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 6
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 12
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_18_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 18
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 27
        },
    ]

    single_baselines = [
        {
            "model_name": "fixed_epoch_MORSE_Nasser_2022-05-27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_row_spacing_nasser_2021-06-01_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_BlaineLake_HornerWest_2021-06-09_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
    ]

    single_diverse_baselines = [
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_1_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        }, 
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_row_spacing_nasser_2021-06-01_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_BlaineLake_HornerWest_2021-06-09_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        }, 
    ]


    single_min_num_baselines = [

        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_BlaineLake_River_2021-06-09_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_BlaineLake_Lake_2021-06-09_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_BlaineLake_HornerWest_2021-06-09_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_UNI_LowN1_2021-06-07_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_BlaineLake_Serhienko9N_2022-06-07_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },

        ### asus
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_row_spacing_nasser_2021-06-01_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },

        ### amd
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Biggar_Dennis1_2021-06-04_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Biggar_Dennis3_2021-06-04_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_MORSE_Dugout_2022-05-27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_MORSE_Nasser_2022-05-27_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_row_spacing_brown_2021-06-01_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_row_spacing_nasser2_2022-06-02_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },       
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Saskatoon_Norheim1_2021-05-26_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Saskatoon_Norheim2_2021-05-26_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },

        # START T5600
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Saskatoon_Norheim4_2022-05-24_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Saskatoon_Norheim5_2022-05-24_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_UNI_Brown_2021-06-05_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_UNI_Dugout_2022-05-30_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },

        ## asus
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_UNI_LowN2_2021-06-07_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_UNI_Sutherland_2021-06-05_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },

        ## amd
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Saskatoon_Norheim1_2021-06-02_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_row_spacing_brown_2021-06-08_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_SaskatoonEast_Stevenson5NW_2022-06-20_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_UNI_Vaderstad_2022-06-16_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_Biggar_Dennis2_2021-06-12_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_BlaineLake_Serhienko10_2022-06-14_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_min_num_diverse_set_of_1_match_BlaineLake_Serhienko9S_2022-06-14_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
    ]

    single_min_num_diverse_baselines = [
        # {
        #     "model_name": "fixed_epoch_min_num_diverse_set_of_27_match_row_spacing_nasser_2021-06-01_no_overlap",
        #     "model_creator": "erik",
        #     "num_training_sets": 1
        # },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_1_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
    ]

    fixed_epoch_exg_baselines = [
        {
            "model_name": "fixed_epoch_exg_active_match_1_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        {
            "model_name": "fixed_epoch_exg_active_match_3_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 3
        },
        {
            "model_name": "fixed_epoch_exg_active_match_6_no_overlap",
            "model_creator": "erik",
            "num_training_sets": 6
        },
    ]

    fixed_epoch_fixed_patch_num_baselines = [
        {
            "model_name": "fixed_epoch_fixed_patch_num_250_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_500_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_1000_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_2000_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_4000_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_8000_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_16000_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_fixed_patch_num_24000_no_overlap",
            "model_creator": "erik",
        },        
        {
            "model_name": "fixed_epoch_fixed_patch_num_32000_no_overlap",
            "model_creator": "erik",
        },
        {
            "model_name": "fixed_epoch_diverse_set_of_27_match_27_no_overlap",
            "model_creator": "erik",
        },
    ]





    test_sets = [
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko9S",
            "mission_date": "2022-06-07"
        },
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko11",
            "mission_date": "2022-06-07"
        },
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko15",
            "mission_date": "2022-06-14"
        },
        {
            "username": "erik",
            "farm_name": "SaskatoonEast",
            "field_name": "Stevenson5SW",
            "mission_date": "2022-06-13"
        },
        {
            "username": "erik",
            "farm_name": "Davidson",
            "field_name": "Stone11NE",
            "mission_date": "2022-06-03"
        },
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko12",
            "mission_date": "2022-06-14"
        },        
        {
            "username": "erik",
            "farm_name": "Biggar",
            "field_name": "Dennis3",
            "mission_date": "2021-06-12"
        },
        {
            "username": "erik",
            "farm_name": "Biggar",
            "field_name": "Dennis5",
            "mission_date": "2021-06-12"
        },
        {
            "username": "erik",
            "farm_name": "UNI",
            "field_name": "CNH-DugoutROW",
            "mission_date": "2022-05-30"
        },
    ]

    num_reps = 1

    all_baselines = []
    # all_baselines.extend(active_baselines)
    # all_baselines.extend(random_image_baselines)
    # all_baselines.extend(diverse_baselines)
    # all_baselines.extend(no_overlap_baselines)
    # all_baselines.extend(CottonWeedDet12_baselines)
    # all_baselines.extend(weed_ai_10000_baselines)
    # all_baselines.extend(weed_ai_20000_baselines)
    # all_baselines.extend(fixed_weed_ai_10000_baselines)
    # all_baselines.extend(fixed_weed_ai_20000_baselines)
    # all_baselines.extend(fixed_weed_ai_30000_baselines)
    # all_baselines.extend(fixed_weed_ai_40000_baselines)
    # all_baselines.extend(fixed_weed_ai_50000_baselines)
    # all_baselines.extend(fixed_epoch_no_overlap_baselines)
    # all_baselines.extend(fixed_epoch_diverse_baselines)
    # all_baselines.extend(fixed_epoch_exg_baselines)
    # all_baselines.extend(single_min_num_baselines)
    # all_baselines.extend(single_min_num_diverse_baselines)
    all_baselines.extend(fixed_epoch_fixed_patch_num_baselines)


    # test(test_sets, all_baselines, num_reps)
    # plot_my_results(test_sets, baselines, num_reps)
    # plot_my_results_alt(test_sets, baselines, diverse_baselines, num_reps)

    add_num_training_patches(all_baselines)
    # plot_my_results_alt(test_sets, no_overlap_baselines, diverse_baselines, num_reps)

    # copy_to_eval(single_min_num_baselines)

    # copy_to_eval_2(fixed_epoch_fixed_patch_num_baselines)

    # test(test_sets, all_baselines, num_reps)
    all_baselines = {
        # "full_image_sets": no_overlap_baselines,
        # "diverse_baselines": diverse_baselines,
        # "CottonWeedDet12_supplemented": CottonWeedDet12_baselines, 
        # "WeedAI_10000_supplemented": weed_ai_10000_baselines,
        # "WeedAI_20000_supplemented": weed_ai_20000_baselines,
        # "fixed_WeedAI_10000_supplemented": fixed_weed_ai_10000_baselines,
        # "fixed_WeedAI_20000_supplemented": fixed_weed_ai_20000_baselines,
        # "fixed_WeedAI_30000_supplemented": fixed_weed_ai_30000_baselines,
        # "fixed_WeedAI_40000_supplemented": fixed_weed_ai_40000_baselines,
        # "fixed_WeedAI_50000_supplemented": fixed_weed_ai_50000_baselines,

        # "random_images": random_image_baselines,
        # "uniformly_selected_patches": diverse_baselines,
        # "selected_patches": active_baselines
        # "fixed_epoch_full_image_sets": fixed_epoch_no_overlap_baselines,
        # "fixed_epoch_diverse_baselines": fixed_epoch_diverse_baselines,
        # "fixed_epoch_exg_baselines": fixed_epoch_exg_baselines,
        "fixed_epoch_fixed_patch_num_baselines": fixed_epoch_fixed_patch_num_baselines
    }
    # plot_my_results_alt(test_sets, all_baselines, num_reps)
    # print("plotting combined results...")
    # check(test_sets, all_baselines, num_reps)
    plot_my_combined_results_alt(test_sets, all_baselines, num_reps, "num_training_patches")



    # plot_single_diverse_comparison(test_sets, single_baselines, single_diverse_baselines)


    # model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_MORSE_Nasser_2022-05-27_no_overlap")
    # score = get_vendi_diversity(model_dir)
    # print("got vendi score", score)

    # model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_diverse_set_of_27_match_1_no_overlap")
    # score = get_vendi_diversity(model_dir)
    # print("got vendi score", score)

    # model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_BlaineLake_HornerWest_2021-06-09_no_overlap")
    # score = get_vendi_diversity(model_dir)
    # print("got vendi score", score)


    # model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_diverse_set_of_27_match_1_no_overlap")
    # score = get_vendi_diversity(model_dir)
    # print("got vendi score", score)


    # plot_min_num_single_diverse_comparison(test_sets, single_min_num_baselines, single_min_num_diverse_baselines)





eval_fixed_patch_num_baselines = [
    "set_of_27_250_patches",
    "set_of_27_500_patches",
    # "set_of_27_630_patches",
    "set_of_27_1000_patches",
    # "set_of_27_1500_patches",
    # "set_of_27_1890_patches",
    "set_of_27_2000_patches",
    "set_of_27_4000_patches",
    "set_of_27_8000_patches",
    "set_of_27_16000_patches",
    "set_of_27_24000_patches",
    "set_of_27_32000_patches",
    "set_of_27_38891_patches",
]

eval_diverse_630_baselines = [
    "set_of_27_630_patches"
]

# eval_single_630_baselines = [
#     # "BlaineLake_River_2021-06-09_630_patches",
#     # "BlaineLake_Lake_2021-06-09_630_patches",
#     # "BlaineLake_HornerWest_2021-06-09_630_patches",
#     # "UNI_LowN1_2021-06-07_630_patches",
#     # "BlaineLake_Serhienko9N_2022-06-07_630_patches",
#     "row_spacing_nasser_2021-06-01_630_patches",
#     # "Biggar_Dennis1_2021-06-04_630_patches",
#     # "Biggar_Dennis3_2021-06-04_630_patches",
#     "MORSE_Dugout_2022-05-27_630_patches",
#     "MORSE_Nasser_2022-05-27_630_patches",
#     "row_spacing_brown_2021-06-01_630_patches",
#     "row_spacing_nasser2_2022-06-02_630_patches",
#     # "Saskatoon_Norheim1_2021-05-26_630_patches",
#     # "Saskatoon_Norheim2_2021-05-26_630_patches",
#     # "Saskatoon_Norheim4_2022-05-24_630_patches",
#     # "Saskatoon_Norheim5_2022-05-24_630_patches",
#     "UNI_Brown_2021-06-05_630_patches",
#     "UNI_Dugout_2022-05-30_630_patches",
#     "UNI_LowN2_2021-06-07_630_patches",
#     "UNI_Sutherland_2021-06-05_630_patches",
#     # "Saskatoon_Norheim1_2021-06-02_630_patches",
#     # "row_spacing_brown_2021-06-08_630_patches",
#     # "SaskatoonEast_Stevenson5NW_2022-06-20_630_patches",
#     # "UNI_Vaderstad_2022-06-16_630_patches",
#     # "Biggar_Dennis2_2021-06-12_630_patches",
#     # "BlaineLake_Serhienko10_2022-06-14_630_patches",
#     # "BlaineLake_Serhienko9S_2022-06-14_630_patches"
# ]


eval_single_630_baselines = [
    "BlaineLake_River_2021-06-09_630_patches",
    "BlaineLake_Lake_2021-06-09_630_patches",
    "BlaineLake_HornerWest_2021-06-09_630_patches",
    "UNI_LowN1_2021-06-07_630_patches",
    "BlaineLake_Serhienko9N_2022-06-07_630_patches",
    "row_spacing_nasser_2021-06-01_630_patches",
    "Biggar_Dennis1_2021-06-04_630_patches",
    "Biggar_Dennis3_2021-06-04_630_patches",
    "MORSE_Dugout_2022-05-27_630_patches",
    "MORSE_Nasser_2022-05-27_630_patches",
    "row_spacing_brown_2021-06-01_630_patches",
    "row_spacing_nasser2_2022-06-02_630_patches",
    "Saskatoon_Norheim1_2021-05-26_630_patches",
    "Saskatoon_Norheim2_2021-05-26_630_patches",
    "Saskatoon_Norheim4_2022-05-24_630_patches",
    "Saskatoon_Norheim5_2022-05-24_630_patches",
    "UNI_Brown_2021-06-05_630_patches",
    "UNI_Dugout_2022-05-30_630_patches",
    "UNI_LowN2_2021-06-07_630_patches",
    "UNI_Sutherland_2021-06-05_630_patches",
    "Saskatoon_Norheim1_2021-06-02_630_patches",
    "row_spacing_brown_2021-06-08_630_patches",
    "SaskatoonEast_Stevenson5NW_2022-06-20_630_patches",
    "UNI_Vaderstad_2022-06-16_630_patches",
    "Biggar_Dennis2_2021-06-12_630_patches",
    "BlaineLake_Serhienko10_2022-06-14_630_patches",
    "BlaineLake_Serhienko9S_2022-06-14_630_patches"
]

eval_single_1890_baselines = [
    "BlaineLake_River_2021-06-09_1890_patches"
]

eval_diverse_1890_baselines = [
    "set_of_27_1890_patches", #"set_of_27_1890_patches"
]

eval_single_630_CottenWeedDet12_baselines = [
    # "row_spacing_brown_2021-06-01_630_patches_and_CottonWeedDet12",
    # "row_spacing_nasser_2021-06-01_630_patches_and_CottonWeedDet12",
    # "UNI_Dugout_2022-05-30_630_patches_and_CottonWeedDet12",
    # "MORSE_Dugout_2022-05-27_630_patches_and_CottonWeedDet12",
    # "UNI_Brown_2021-06-05_630_patches_and_CottonWeedDet12",
    # "UNI_Sutherland_2021-06-05_630_patches_and_CottonWeedDet12",
    # "row_spacing_nasser2_2022-06-02_630_patches_and_CottonWeedDet12",
    # "MORSE_Nasser_2022-05-27_630_patches_and_CottonWeedDet12",
    # "UNI_LowN2_2021-06-07_630_patches_and_CottonWeedDet12",
]

eval_diverse_1500_baselines = [
    "set_of_27_1500_patches"
]

eval_single_1500_baselines = [
    "row_spacing_brown_2021-06-01_1500_patches",
    "row_spacing_nasser2_2022-06-02_1500_patches",
    "row_spacing_nasser_2021-06-01_1500_patches",
    "Saskatoon_Norheim4_2022-05-24_1500_patches",
    "Saskatoon_Norheim5_2022-05-24_1500_patches",
    "BlaineLake_HornerWest_2021-06-09_1500_patches",
    "BlaineLake_Lake_2021-06-09_1500_patches",
    "BlaineLake_River_2021-06-09_1500_patches",
    "Saskatoon_Norheim1_2021-05-26_1500_patches",
    "Saskatoon_Norheim5_2022-05-24_1500_patches",
    "UNI_LowN1_2021-06-07_1500_patches",
    "Saskatoon_Norheim1_2021-06-02_1500_patches",
    "row_spacing_brown_2021-06-08_1500_patches"

]

nonperturbed_baselines = [
    "set_of_27_16000_patches"
]

perturbed_baselines = [
    "set_of_27_perturbed_by_10_16000_patches"
]

dilation_baselines = [
    "set_of_27_dilated_by_1_16000_patches",
    "set_of_27_dilated_by_2_16000_patches",
    "set_of_27_dilated_by_3_16000_patches",
    "set_of_27_dilated_by_4_16000_patches",
    "set_of_27_dilated_by_5_16000_patches",
    "set_of_27_dilated_by_6_16000_patches",
    "set_of_27_dilated_by_7_16000_patches",
    "set_of_27_dilated_by_8_16000_patches",
    "set_of_27_dilated_by_9_16000_patches",
    "set_of_27_dilated_by_10_16000_patches",
]

remove_baselines = [
    "set_of_27_remove_0.05_16000_patches",
    "set_of_27_remove_0.1_16000_patches",
    "set_of_27_remove_0.15_16000_patches"
]

exg_repl_baselines = [
    "set_of_27_exg_repl_250_patches",
    "set_of_27_exg_repl_500_patches",
    "set_of_27_exg_repl_1000_patches",
    "set_of_27_exg_repl_2000_patches",
    "set_of_27_exg_repl_4000_patches",
    "set_of_27_exg_repl_8000_patches",
    "set_of_27_exg_repl_16000_patches",
    # "set_of_27_exg_repl_24000_patches",
    # "set_of_27_exg_repl_32000_patches",
    # "set_of_27_exg_repl_38891_patches",

]

baselines_2 = [
    # "UNI_Dugout_2022-05-30_630_patches_BlaineLake_Serhienko9S_2022-06-14_630_patches",
    "BlaineLake_River_2021-06-09_630_patches_BlaineLake_Serhienko9S_2022-06-14_630_patches"
]


eval_in_domain_test_sets = [
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "nasser",
        "mission_date": "2021-06-01"
    },
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "brown",
        "mission_date": "2021-06-01"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Dugout",
        "mission_date": "2022-05-30"
    },
    {
        "username": "eval",
        "farm_name": "MORSE",
        "field_name": "Dugout",
        "mission_date": "2022-05-27"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Brown",
        "mission_date": "2021-06-05"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Sutherland",
        "mission_date": "2021-06-05"
    },
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "nasser2",
        "mission_date": "2022-06-02"
    },
    {
        "username": "eval",
        "farm_name": "MORSE",
        "field_name": "Nasser",
        "mission_date": "2022-05-27"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "LowN2",
        "mission_date": "2021-06-07"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim4",
        "mission_date": "2022-05-24"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim5",
        "mission_date": "2022-05-24"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim1",
        "mission_date": "2021-05-26"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim2",
        "mission_date": "2021-05-26"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis1",
        "mission_date": "2021-06-04"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis3",
        "mission_date": "2021-06-04"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "River",
        "mission_date": "2021-06-09"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Lake",
        "mission_date": "2021-06-09"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "HornerWest",
        "mission_date": "2021-06-09"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "LowN1",
        "mission_date": "2021-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9N",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim1",
        "mission_date": "2021-06-02"
    },
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "brown",
        "mission_date": "2021-06-08"
    },
    {
        "username": "eval",
        "farm_name": "SaskatoonEast",
        "field_name": "Stevenson5NW",
        "mission_date": "2022-06-20"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Vaderstad",
        "mission_date": "2022-06-16"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis2",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko10",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-14"
    }
]

eval_test_sets = [
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko11",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko15",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "SaskatoonEast",
        "field_name": "Stevenson5SW",
        "mission_date": "2022-06-13"
    },
    {
        "username": "eval",
        "farm_name": "Davidson",
        "field_name": "Stone11NE",
        "mission_date": "2022-06-03"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko12",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis3",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis5",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "CNH-DugoutROW",
        "mission_date": "2022-05-30"
    },
]


def my_diversity_scale_plot():
    baseline_sets = {}

    for b in eval_single_1500_baselines:
        s = b[:-len("_1500_patches")]
        model_name_630 = s + "_630_patches"
        baseline_sets[b] = []
        baseline_sets[b].append({
            "model_name": model_name_630,
            "patch_num": 630
        })
        baseline_sets[b].append({
            "model_name": b,
            "patch_num": 1500
        })

    baseline_sets["set_of_27"] = []
    baseline_sets["set_of_27"].append({
        "model_name": "set_of_27_630_patches",
        "patch_num": 630
    })
    baseline_sets["set_of_27"].append({
        "model_name": "set_of_27_1500_patches",
        "patch_num": 1500
    })

    create_eval_size_plot(eval_test_sets, baseline_sets, "diversity_scale")    



def my_diversity_plot():
    single_baselines = []
    for baseline in eval_single_630_baselines:

        b_name = baseline[:len(baseline)-len("_630_patches")]
        pieces = b_name.split("_")
        label = "_".join(pieces[:-2]) + "/" + pieces[-2] + "/" + pieces[-1]


        single_baselines.append({
            "model_name": baseline,
            "model_creator": "eval",
            "model_label": label
        })

    diverse_baselines = []
    for baseline in eval_diverse_630_baselines:
        diverse_baselines.append({
            "model_name": baseline,
            "model_creator": "eval",
            "model_label": "Diverse"
        })        
    out_dirname = "diversity_630"
    create_eval_min_num_plot(eval_test_sets, single_baselines, diverse_baselines, out_dirname)
    create_eval_min_num_plot_individual_test_sets(eval_test_sets, single_baselines, diverse_baselines, out_dirname)


def my_multitrace_size_plot():
    baseline_sets = {}

    for i in range(3):
        baseline_sets["diverse_rep_" + str(i)] = []
        for b in eval_fixed_patch_num_baselines:
            baseline_sets["diverse_rep_" + str(i)].append({
                "model_name": b + "_rep_" + str(i),
                "patch_num": int((b[len("set_of_27_"):]).split("_")[0])
            })

    # for i in range(1):
    #     baseline_sets["exg_repl_" + str(i)] = []
    #     for b in exg_repl_baselines:
    #         baseline_sets["exg_repl_" + str(i)].append({
    #             "model_name": b + "_rep_" + str(i),
    #             "patch_num": int((b[len("set_of_27_exg_repl_"):]).split("_")[0])
    #         })

    create_eval_size_plot(eval_test_sets, baseline_sets, "size_multitrace")

def my_dilation_plot():
    baselines = []
    for rep_num in range(1):
        baselines.append({
            "model_name": "set_of_27_16000_patches",
            # "model_creator": "eval",
            "dilation_sigma": 0
        })
        # baselines.append({
        #     "model_name": "set_of_27_250_patches",
        #     # "model_creator": "eval",
        #     "dilation_sigma": 10
        # })
        


        for baseline in dilation_baselines:
        
            baselines.append({
                "model_name": baseline,
                # "model_creator": "eval",
                "dilation_sigma": int(baseline.split("_")[5])
            })


    create_dilation_plot(eval_test_sets, baselines, "dilation")
    # create_dilation_plot(eval_in_domain_test_sets, eval_test_sets, baselines, "dilation")


def my_removal_plot():
    baselines = []
    for rep_num in range(1):
        baselines.append({
            "model_name": "set_of_27_16000_patches",
            # "model_creator": "eval",
            "removal_percentage": 0
        })
        # baselines.append({
        #     "model_name": "set_of_27_250_patches",
        #     # "model_creator": "eval",
        #     "dilation_sigma": 10
        # })
        


        for baseline in remove_baselines:
        
            baselines.append({
                "model_name": baseline,
                # "model_creator": "eval",
                "removal_percentage": float(baseline.split("_")[4])
            })


    # create_removal_plot(eval_in_domain_test_sets, eval_test_sets, baselines, "removal")
    create_removal_plot(eval_test_sets, baselines, "removal")


def my_locs_plot():
    geo_locations.show_image_set_locations(eval_in_domain_test_sets, eval_test_sets, "locs")


def my_size_plot():
    baseline_sets = {}

    baseline_sets["diverse"] = []
    for b in eval_fixed_patch_num_baselines:
        baseline_sets["diverse"].append({
            "model_name": b,
            "patch_num": int((b[len("set_of_27_"):]).split("_")[0])
        })
    
    # baseline_sets["exg_repl"] = []
    # for b in exg_repl_baselines:
    #     print(b)
    #     baseline_sets["exg_repl"].append({
    #         "model_name": b,
    #         "patch_num": int((b[len("set_of_27_exg_repl_"):]).split("_")[0])
    #     })

    # create_eval_size_plot(eval_test_sets, baseline_sets, "size")
    # create_eval_size_plot_individual_test_sets(eval_test_sets, baseline_sets, "size")


    create_eval_size_plot_id_ood(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size")
    # create_individual_image_sets_eval_size_plot_id_ood(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size")


def my_patch_merging_plot():

    baseline = {
        "model_name": eval_fixed_patch_num_baselines[-1]
    }

    create_patch_merging_plot(eval_test_sets, baseline,
                              ["no_overlap_", "no_prune_", ""])


def my_accuracy_efficiency_sampling():

    baseline_model = {
        "model_name": "set_of_27_38891_patches_rep_0", #eval_fixed_patch_num_baselines[-1]
    }
    for eval_test_set in eval_test_sets:
        if eval_test_set["farm_name"] in ["BlaineLake", "SaskatoonEast", "Davidson"]:
            print("\n\n{}".format(eval_test_set))
            accuracy_sampling_efficiency(eval_test_set, baseline_model, "accuracy_assessment")
            # exit()

def get_result_uuids(baseline, test_set, methods, num_annotations_to_select_lst):

    model_names = []
    model_name = baseline["model_name"] + "_pre_finetune"
    model_names.append(model_name)
    dup_nums = 5
    for dup_num in range(dup_nums):
        for method in methods:
            for num_annotations_to_select in num_annotations_to_select_lst:
                model_name = baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select) + "_annotations_dup_" + str(dup_num)
                model_names.append(model_name)

    test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
    # test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    
    mapping = get_mapping_for_test_set(test_set_image_set_dir)

    result_uuids = []
    for model_name in model_names:
        if model_name in mapping:
            result_uuids.append(mapping[model_name])


    for result_uuid in result_uuids:
        print(result_uuid)
    




def eval_run():

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()


    single_baselines = []
    for baseline in remove_baselines: #dilation_baselines:
        for rep_num in range(1):
            single_baselines.append({
                "model_name": baseline + "_rep_" + str(rep_num),
                "model_creator": "eval"
            })


    # single_weed_baselines = []
    # for baseline in eval_single_630_CottenWeedDet12_baselines:
    #     single_weed_baselines.append({
    #         "model_name": baseline, #+ "_rep_1",
    #         "model_creator": "eval"
    #     })
    # for baseline in eval_single_630_CottenWeedDet12_baselines:
    #     for i in range(3):
    #         single_weed_baselines.append({
    #             "model_name": baseline + "_rep_" + str(i), #2",
    #             "model_creator": "eval"
    #         })

    # d_perturbed_baselines = []
    # for baseline in perturbed_baselines:
    #     d_perturbed_baselines.append({
    #         "model_name": baseline, # + "_rep_0",
    #         "model_creator": "eval"
    #     })
    # d_nonperturbed_baselines = []
    # for baseline in nonperturbed_baselines:
    #     d_nonperturbed_baselines.append({
    #         "model_name": baseline, # + "_rep_0",
    #         "model_creator": "eval"
    #     })

    # predict_on_test_sets(eval_test_sets, single_baselines)

    # my_patch_merging_plot()
    # my_dilation_plot()
    # my_size_plot()
    # my_removal_plot()
    # exit()

    baselines = [{"model_name": "set_of_27_38891_patches_rep_0", "model_creator": "eval"}]

    # # # # create_eval_improvement_plot(eval_test_sets, d_nonperturbed_baselines, d_perturbed_baselines, [], [])

    # # # # create_patch_merging_plot(eval_test_sets, baselines[0], ["no_overlap_", "no_prune_", "alt_prune_", ""])
    # exit()

    methods = [
        # "random_images",
        # "random_patches",
        # "random_patches_match_patch_num",
        # "selected_patches_match_patch_num",
        # "selected_patches_match_annotation_num",
        # "selected_patches",
        # "selected_patches_unfair_dist_score"

        "selected_patches_first",
        "random_patches_second",
    ]

    num_dups = 5
    num_annotations_to_select_lst = [250, 500] #400, 500, 600, 700]
    for num_annotations_to_select in [3250]: #1000, 1250, 1500]: #num_annotations_to_select_lst:
        # fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[2], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
        # fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[1], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=5)
        
        fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[0], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=5)

        # fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[1], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    # num_annotations_to_select_lst = [250, 500] #400, 500, 600, 700]
    # for num_annotations_to_select in [500]: #num_annotations_to_select_lst:
    #     fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[2], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    #     fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[0], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    #     fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[1], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    # for num_annotations_to_select in [750]: #num_annotations_to_select_lst:
    #     fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[2], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    #     fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[0], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    #     fine_tune_experiment.eval_fine_tune_test(server, eval_test_sets[1], baselines[0], methods, num_annotations_to_select=num_annotations_to_select, num_dups=num_dups)
    # get_result_uuids(baselines[0], eval_test_sets[3], methods, [250, 500, 750, 1000, 1250, 1500])

    # create_fine_tune_plot(baselines[0], eval_test_sets[0], methods, num_annotations_to_select_lst=[250, 500, 750, 1000, 1250, 1500], num_dups=5)
    # create_fine_tune_plot_averaged(baselines[0], eval_test_sets[1], methods, 
    # num_annotations_to_select_lst=[250, 500, 750, 1000, 1250, 1500, 1710], num_dups=num_dups)

    # create_fine_tune_plot_averaged(baselines[0], eval_test_sets[0], methods, 
    # num_annotations_to_select_lst=[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250], num_dups=5)
    # my_dilation_plot()
    # my_removal_plot()
    # my_size_plot()
    exit()

    # my_locs_plot()



    # predict_on_test_sets(eval_test_sets, single_baselines) # + d_diverse_baselines)
    # create_weed_comparison_plot(eval_test_sets, single_baselines)

    exit()
    # single_baselines = []
    # for baseline in eval_single_630_baselines:
    #     # for i in range(0, 5):
    #     single_baselines.append({
    #         "model_name": baseline, # + "_rep_" + str(i),
    #         "model_creator": "eval"
    #     })
    # for baseline in eval_single_1890_baselines:
    #     # for i in range(0, 5):
    #     single_baselines.append({
    #         "model_name": baseline, # + "_rep_" + str(i),
    #         "model_creator": "eval"
    #     })


    # diverse_baselines = []
    # for baseline in eval_diverse_630_baselines:
    #     # for i in range(0, 5):
    #     diverse_baselines.append({
    #         "model_name": baseline, # + "_rep_" + str(i),
    #         "model_creator": "eval"
    #     })

    # for baseline in eval_diverse_1890_baselines:
    #     # for i in range(0, 5):
    #     diverse_baselines.append({
    #         "model_name": baseline, # + "_rep_" + str(i),
    #         "model_creator": "eval"
    #     })
    # baselines = []
    # for baseline in eval_fixed_patch_num_baselines: #exg_repl_baselines:
    #     # for i in range(0, 5):
    #     baselines.append({
    #         "model_name": baseline + "_rep_" + str(2),
    #         "model_creator": "eval"
    #     })

        
    # my_diversity_scale_plot()
    # my_patch_merging_plot()
    # my_diversity_plot()
    # print_lex_sorted_image_sets(eval_test_sets)
    # my_size_plot()
    # my_size_plot()
    # my_multitrace_size_plot()
    # predict_on_test_sets(eval_test_sets, baselines)
    # create_eval_min_num_plot(eval_test_sets, single_baselines, diverse_baselines)

    exit()


    baselines = []
    for baseline in eval_single_630_baselines:
        baselines.append({
            "model_name": baseline,
            "model_label": baseline,
            "model_creator": "eval"
        })
    diverse_baselines = []
    for baseline in baselines_2:
        diverse_baselines.append({
            "model_name": baseline,
            "model_label": "combo_2",
            "model_creator": "eval"
        })

    out_dirname = "2_combo_diversity_630"
    create_eval_min_num_plot(eval_test_sets, baselines, diverse_baselines, out_dirname)
    create_eval_min_num_plot_individual_test_sets(eval_test_sets, baselines, diverse_baselines, out_dirname)




    # compare_exg_repl_patches("set_of_27_250_patches_rep_0", "set_of_27_exg_repl_250_patches_rep_0", "compare_exg_repl_patches")

    exit()
    my_accuracy_efficiency_sampling()
    


    exit()
    

    single_baselines = []
    single_baselines_improved = []
    for baseline in eval_single_630_baselines:
        x = baseline[:len(baseline)-len("_630_patches")]
        for z in eval_single_1500_baselines:
            if z[:len(z)-len("_1500_patches")] == x:
                
                single_baselines.append({
                    "model_name": baseline, # + "_rep_0",
                    "model_creator": "eval"
                })

                single_baselines_improved.append({
                    "model_name": z, # + "_rep_0",
                    "model_creator": "eval"
                })

    diverse_baselines = []
    for baseline in eval_diverse_630_baselines:
        diverse_baselines.append({
            "model_name": baseline, # + "_rep_0", # + str(i),
            "model_creator": "eval"
        })


    # single_baselines_improved = []
    # for baseline in eval_single_1500_baselines:
    #     single_baselines_improved.append({
    #         "model_name": baseline, # + "_rep_0",
    #         "model_creator": "eval"
    #     })

    diverse_baselines_improved = []
    for baseline in eval_diverse_1500_baselines:
        diverse_baselines_improved.append({
            "model_name": baseline, # + "_rep_0", # + str(i),
            "model_creator": "eval"
        })

    # for baseline in eval_diverse_min_num_baselines:
    # for i in range(0):
    #     d_diverse_baselines.append({
    #         "model_name": eval_diverse_min_num_baselines[0], #+ "_rep_" + str(i),
    #         "model_creator": "eval"
    #     })

    # d_baselines = []
    # for baseline in eval_fixed_patch_num_baselines:
    #     d_baselines.append({
    #         "model_name": baseline, # + "_rep_0",
    #         "model_creator": "eval"
    #     })


    # predict_on_test_sets(eval_test_sets, d_single_baselines) # + d_diverse_baselines)

    # create_eval_size_plot(eval_test_sets, d_baselines)

    
    # create_eval_improvement_plot(eval_test_sets, single_baselines, single_baselines_improved, diverse_baselines, diverse_baselines_improved)

if __name__ == "__main__":
    eval_run()
