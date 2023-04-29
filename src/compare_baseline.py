import os
import glob
import shutil
import time
import math as m
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from vendi_score import vendi
from scipy.stats import entropy
import pandas as pd
import random
import uuid
import urllib3
import logging
import matplotlib.pyplot as plt
import tqdm

import image_set_actions as isa
import server
from models.common import annotation_utils, box_utils
from io_utils import json_io
import fine_tune_eval
from image_set import Image

from lock_queue import LockQueue
import diversity_test



def get_mapping_for_test_set(test_set_image_set_dir):

    mapping = {}
    results_dir = os.path.join(test_set_image_set_dir, "model", "results")
    for result_dir in glob.glob(os.path.join(results_dir, "*")):
        request_path = os.path.join(result_dir, "request.json")
        request = json_io.load_json(request_path)
        mapping[request["results_name"]] = request["request_uuid"]
    return mapping


def create_eval_size_plot(test_sets, baselines):

    results = []
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
    for baseline in baselines:
        patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])

        test_set_accuracies = []
        for test_set in test_sets:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                test_set["username"], "image_sets",
                                                test_set["farm_name"],
                                                test_set["field_name"],
                                                test_set["mission_date"])
            
            print(test_set_str)

            rep_accuracies = []
            for rep_num in range(1):
                model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                print(model_name)
                model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                excel_path = os.path.join(result_dir, "metrics.xlsx")
                df = pd.read_excel(excel_path, sheet_name=0)
                rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)






                # annotations = annotation_utils.load_annotations(os.path.join(result_dir, "annotations.json"))

                # predictions_path = os.path.join(result_dir, "predictions.json")
                # predictions = annotation_utils.load_predictions(predictions_path)

                # if test_set_str in eval_assessment_images_lookup:
                #     assessment_images = eval_assessment_images_lookup[test_set_str]
                # else:
                #     assessment_images = list(annotations.keys())


                # accuracies = []
                # for image_name in assessment_images:
                #     sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]

                #     accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
                #     accuracies.append(accuracy)
                # rep_accuracy = np.mean(accuracies)






                rep_accuracies.append(rep_accuracy)

            test_set_accuracy = np.mean(rep_accuracies)
            test_set_accuracies.append(test_set_accuracy)

        baseline_accuracy = np.mean(test_set_accuracies)
        results.append((patch_num, baseline_accuracy))


    plt.plot([x[0] for x in results], [x[1] for x in results], color="red", marker="o", linestyle="dashed", linewidth=1)
   
    plt.xlabel("Number of Patches")
    plt.ylabel("Test Accuracy")


    out_path = os.path.join("eval_charts", "training_set_size_variation", "plot.svg")
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

        out_path = os.path.join("eval_charts", "single_diverse_improved_comparisons", test_set_str + ".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)


def create_eval_min_num_plot(test_sets, single_baselines, diverse_baselines):

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
            for rep_num in range(3):

                model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                test_set_accuracies = []
                for test_set in test_sets:
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

            baseline_accuracy = np.mean(rep_accuracies)
            baseline_std = np.std(rep_accuracies)
        

        
                    # baseline_accuracy = np.mean(rep_accuracies)
                    # #  baseline_variance = np.std(rep_accuracies)
                    # baseline_accuracies.append(baseline_accuracy)


                    # results[test_set_str][result_key].append(
                    #     (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



            # overall_baseline_accuracy = np.mean(baseline_accuracies)

            results["overall"][result_key].append(
                (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], 
                baseline_accuracy, #overall_baseline_accuracy,
                baseline_std))
                
                #np.min(baseline_accuracies),
                #np.max(baseline_accuracies)))


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
        ax = fig.add_axes([0.30, 0.05, 0.65, 0.9])

        ax.scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), color="red", label="Single Image Set", zorder=2)
        ax.scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", label="Diverse Random Selection", zorder=2)
        for i, x in enumerate(single_tuples):
            ax.plot([x[1] - x[2], x[1] + x[2]], [i, i], color="red")

        for i, x in enumerate(diverse_tuples):
            ax.plot([x[1] - x[2], x[1] + x[2]], [len(single_tuples)+i, len(single_tuples)+i], color="blue")

        # if test_set_str == "overall":
        #     ax.scatter([x[2] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
        #     ax.scatter([x[2] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")
            
        #     ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color="red", marker="x")
        #     ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color="blue", marker="x")

        ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
        ax.set_yticklabels(labels)


        ax.legend()
        ax.set_xlabel("Test Accuracy")

        out_path = os.path.join("eval_charts", "single_diverse_3rep_comparisons", test_set_str + ".svg")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)



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
                "results_name": baseline["model_name"],
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



def get_vendi_diversity(model_dir):


    log_path = os.path.join(model_dir, "log.json")
    log = json_io.load_json(log_path)

    taken_patches = {}
    for image_set in log["image_sets"]:

        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]

        image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
        taken_patches[image_set_str] = {}
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
                taken_patches[image_set_str][image_name] = []
                image_h = metadata["images"][image_name]["height_px"]
                image_w = metadata["images"][image_name]["width_px"]
                for region in image_set["taken_regions"][image_name]:

                    if region[2] != image_h and region[3] != image_w:
                        taken_patches[image_set_str][image_name].append(region)

        else:

            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    taken_patches[image_set_str][image_name] = []
                    image_h = metadata["images"][image_name]["height_px"]
                    image_w = metadata["images"][image_name]["width_px"]
                    for i in range(0, image_h, patch_size):
                        for j in range(0, image_w, patch_size):
                            taken_patches[image_set_str][image_name].append([i, j, i+patch_size, j+patch_size])
            

    patch_arrays = []
    for image_set_str in taken_patches:

        pieces = image_set_str.split(" ")
        username = pieces[0]
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]

        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)


        for image_name in taken_patches[image_set_str]:

            image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
            image = Image(image_path)

            image_array = image.load_image_array()

            for patch_coords in taken_patches[image_set_str][image_name]:

                patch_array = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
                patch_arrays.append(patch_array)





    batch_size = 256

    patch_arrays = np.array(patch_arrays)
    num_patches = patch_arrays.size
    print("processing {} patches".format(num_patches))

    # from models.yolov4.yolov4_image_set_driver import create_default_config
    # from models.common import model_keys
    # config = create_default_config()
    # model_keys.add_general_keys(config)
    # model_keys.add_specialized_keys(config)

    # from models.yolov4 import yolov4

    # model = yolov4.YOLOv4TinyBackbone(config, max_pool=True)
    # input_shape = (256, *(config["arch"]["input_image_shape"]))
    # model.build(input_shape=input_shape)
    # model.load_weights(os.path.join("usr", "data", "erik", "models", "available", "public", 
    #                                 "fixed_epoch_set_of_27_no_overlap", "weights.h5"), by_name=False)

    weights = 'imagenet'
    model = tf.keras.applications.InceptionV3( #101( #ResNet50(
        weights=weights,
        include_top=False, 
        input_shape=[None, None, 3],
        pooling="max"
    )

    # if extraction_type == "box_patches":
    #     input_image_shape = np.array([150, 150, 3])
    # else:
    input_image_shape = [416, 416] #config.arch["input_image_shape"]

    all_features = []
    for i in tqdm.trange(0, num_patches, batch_size):
        batch_patches = []
        for j in range(i, min(num_patches, i+batch_size)):
            patch = tf.convert_to_tensor(patch_arrays[j], dtype=tf.float32)
            patch = tf.image.resize(images=patch, size=input_image_shape[:2])
            # vec = tf.reshape(patch, [-1])

            # all_features.append(vec)
            batch_patches.append(patch)
        batch_patches = tf.stack(values=batch_patches, axis=0)
        
        features = model.predict(batch_patches)
        for f in features:
            f = f.flatten()
            all_features.append(f)

    all_features = np.array(all_features)
    print("calculating vendi score...")
    score = vendi.score(all_features, cosine)
    print("score is {}".format(score))

    # print("shape of features matrix is {}".format(all_features.shape))
    # print("calculating similarity matrix...")
    # sim_mat = np.zeros(shape=(all_features.shape[0], all_features.shape[0]))
    # for i in range(all_features.shape[0]):
    #     for j in range(all_features.shape[0]):
    #         if i == j:
    #             sim_mat[i][j] = 1
    #         elif i > j:
    #             sim_mat[i][j] = sim_mat[j][i]
    #         else:
    #             sim = cosine(all_features[i], all_features[j])
    #             sim_mat[i][j] = sim
    # # sim_mat = cosine(all_features, all_features)
    # print("sim_mat is: {}".format(sim_mat))
    # sim_mat = sim_mat / all_features.shape[0]
    # print("calculating eigenvalues...")
    # w, _ = np.linalg.eig(sim_mat)
    # print("eigenvalues are: {}".format(w))
    # print("calculating entropy...")
    # ent = entropy(w)
    # print("entropy is {}".format(ent))
    # print("calculating vendi score")
    # score = m.exp(ent)
    # print("score is {}".format(score))

    return score


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
    "set_of_27_1000_patches",
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

def eval_run():

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()



    single_baselines = []
    for baseline in eval_single_630_baselines:
        single_baselines.append({
            "model_name": baseline, # + "_rep_0",
            "model_creator": "eval"
        })

    diverse_baselines = []
    for baseline in eval_diverse_630_baselines:
        diverse_baselines.append({
            "model_name": baseline, # + "_rep_2",
            "model_creator": "eval"
        })        

    # predict_on_test_sets(eval_test_sets, diverse_baselines) # + d_diverse_baselines)
    create_eval_min_num_plot(eval_test_sets, single_baselines, diverse_baselines)
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
