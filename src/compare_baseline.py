import os
import glob
import shutil
import time
import math as m
import random
import uuid
import urllib3
import logging
import matplotlib.pyplot as plt

import image_set_actions as isa
import server
from models.common import annotation_utils
from io_utils import json_io
import fine_tune_eval

from lock_queue import LockQueue
import diversity_test

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
            
                # label = k
                # results[k] = []
                full_predictions_lst = []
                annotations_lst = []
                assessment_images_lst = []
                for test_set in test_sets:
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

                global_accuracy = fine_tune_eval.get_global_accuracy_multiple_image_sets(annotations_lst, full_predictions_lst, assessment_images_lst)

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
            plt.ylabel("Accuracy")

            test_set_str = "combined_test_sets" #test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
            out_path = os.path.join("baseline_charts", "active_learning_comparison", test_set_str, "global", "accuracy", "rep_" + str(rep_num) + ".svg")
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(out_path)


def plot_my_results_alt(test_sets, all_baselines, num_reps):

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

                    global_accuracy = fine_tune_eval.get_global_accuracy(annotations, full_predictions, list(annotations.keys()))

                    x = baseline["num_training_sets"]
                    # x = baseline["num_training_patches"]
                    y = global_accuracy
                    # c = "red" if k == "org_baselines" else "blue"
                    # methods.append()
                    # if k == "org_baselines":
                    #     direct_application_org_results.append((x, y))
                    # else:
                    #     direct_application_diverse_results.append((x, y))
                    results[k].append((x, y))

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

            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
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
        }
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
    all_baselines.extend(fixed_epoch_no_overlap_baselines)

    # test(test_sets, all_baselines, num_reps)
    # plot_my_results(test_sets, baselines, num_reps)
    # plot_my_results_alt(test_sets, baselines, diverse_baselines, num_reps)

    add_num_training_patches(all_baselines)
    # plot_my_results_alt(test_sets, no_overlap_baselines, diverse_baselines, num_reps)

    # test(test_sets, all_baselines, num_reps)
    all_baselines = {
        "full_image_sets": no_overlap_baselines,
        # "diverse_baselines": diverse_baselines,
        # "CottonWeedDet12_supplemented": CottonWeedDet12_baselines, 
        # "WeedAI_10000_supplemented": weed_ai_10000_baselines,
        # "WeedAI_20000_supplemented": weed_ai_20000_baselines,
        # "fixed_WeedAI_10000_supplemented": fixed_weed_ai_10000_baselines,
        # "fixed_WeedAI_20000_supplemented": fixed_weed_ai_20000_baselines,
        # "random_images": random_image_baselines,
        # "uniformly_selected_patches": diverse_baselines,
        # "selected_patches": active_baselines
        "fixed_epoch_full_image_sets": fixed_epoch_no_overlap_baselines
    }
    plot_my_results_alt(test_sets, all_baselines, num_reps)

    # plot_my_combined_results_alt(test_sets, all_baselines, num_reps, "num_training_patches")

if __name__ == "__main__":
    run()
