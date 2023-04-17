import logging
import os
import glob
import shutil
import random
import urllib3
import tqdm
import time
import uuid
import math as m
import numpy as np
import cv2
from natsort import natsorted

from models.common import annotation_utils, box_utils
from io_utils import json_io
import diversity_test
from image_set import Image
import image_utils



import server
from lock_queue import LockQueue

import image_set_actions as isa

def get_num_patches_used_by_model(model_dir):

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

                    total_num_patches += num_patches
        else:
            
            num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=patch_overlap_percent)
            
            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    total_num_patches += num_patches_per_image


    return total_num_patches
                    


def run_diverse_model(training_image_sets, model_name, model_dir_to_match, prev_model_dir):

    s = {}

    if prev_model_dir is not None:
        num_patches_prev_taken = get_num_patches_used_by_model(prev_model_dir)
        prev_log_path = os.path.join(prev_model_dir, "log.json")
        prev_log = json_io.load_json(prev_log_path)
        for image_set in prev_log["image_sets"]:
            key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]
            if key not in s:
                s[key] = {}

            s[key] = image_set["taken_regions"]


    else:
        num_patches_prev_taken = 0

    num_patches_to_match = get_num_patches_used_by_model(model_dir_to_match)

    num_patches_to_take = num_patches_to_match - num_patches_prev_taken

    print("Num patches to match: {}".format(num_patches_to_match))
    print("Num patches previously taken: {}".format(num_patches_prev_taken))
    print("Num patches to take: {}".format(num_patches_to_take))

    candidates = []
    for image_set in training_image_sets:

        key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]


        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        for image_name in annotations.keys():
            image_w = metadata["images"][image_name]["width_px"]
            image_h = metadata["images"][image_name]["height_px"]
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
                            pass
                        else:
                            candidates.append((key, image_name, patch_coords))

                
    taken_candidates = random.sample(candidates, num_patches_to_take)

    for taken_candidate in taken_candidates:
        key = taken_candidate[0]
        if key not in s:
            s[key] = {}

        image_name = taken_candidate[1]
        if image_name not in s[key]:
            s[key][image_name] = []

        s[key][image_name].append(taken_candidate[2])

    log = {}
    log["model_creator"] = "erik"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for key in s:
        pieces = key.split("/")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": s[key],
            "patch_overlap_percent": 0
        })

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)


    # server.sch_ctx["baseline_queue"].enqueue(log)

    # baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    # while baseline_queue_size > 0:
    
    #     log = server.sch_ctx["baseline_queue"].dequeue()
    #     re_enqueue = server.process_baseline(log)
    #     if re_enqueue:
    #         server.sch_ctx["baseline_queue"].enqueue(log)
    #     baseline_queue_size = server.sch_ctx["baseline_queue"].size()


def run_pending_model(model_dir):

    log_path = os.path.join(model_dir, "log.json")
    log = json_io.load_json(log_path)
    log["submission_time"] = int(time.time())
    json_io.save_json(log_path, log)


    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()


def run_full_image_set_model(training_image_sets, model_name):
    log = {}
    log["model_creator"] = "erik"
    # log["model_name"] = str(num_images_to_select) ##"set_of_" + str(len(org_image_sets)) + "_no_overlap"
    log["model_name"] = model_name #"random_images_" + str(iteration_number) #cur_num)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    # log["image_sets"] = image_sets
    log["image_sets"] = []

    for image_set in training_image_sets:
        image_set["patch_overlap_percent"] = 0
        log["image_sets"].append(image_set)


    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)


    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()





def run_weed_test(training_image_sets, weed_image_sets, model_name, num_weed_patches_to_add):

    log = {}
    log["model_creator"] = "erik"
    # log["model_name"] = str(num_images_to_select) ##"set_of_" + str(len(org_image_sets)) + "_no_overlap"
    log["model_name"] = model_name #"random_images_" + str(iteration_number) #cur_num)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    # log["image_sets"] = image_sets
    log["image_sets"] = []
    # for image_set in training_image_sets:
    #     image_set["patch_overlap_percent"] = 0
    #     log["image_sets"].append(image_set)
    


    for image_set in weed_image_sets:
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = 416 #annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        num_annotated = diversity_test.get_num_fully_annotated_images(annotations, image_w, image_h)

        num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)

        image_set["num_patches"] = num_patches_per_image * num_annotated


    capacities = []
    for image_set in weed_image_sets:
        capacities.append(image_set["num_patches"])
        # diverse_image_sets_patch_num += image_set["num_patches"]

    capacities = np.array(capacities)
    org_capacities = np.copy(capacities)
    num_to_match = num_weed_patches_to_add
    print("num_to_match", num_to_match)
    print("total capacity", np.sum(capacities))
    print("capacities", capacities)

    satisfied = False
    while not satisfied:

        min_capacity_this_iteration = min(capacities[capacities != 0])
        desired_num_to_take_this_iteration = num_to_match // np.sum(capacities != 0)

        num_actually_taken_this_iteration = min(min_capacity_this_iteration, desired_num_to_take_this_iteration)
        for i in range(capacities.size):
            if capacities[i] > 0:
                capacities[i] -= num_actually_taken_this_iteration
                num_to_match -= num_actually_taken_this_iteration
 
        if min_capacity_this_iteration >= desired_num_to_take_this_iteration:
            satisfied = True

    i = 0
    while num_to_match > 0:
        if capacities[i] > 0:
            capacities[i] -= 1
            num_to_match -= 1
        i += 1

    taken = org_capacities - capacities
    print("Taken: {}".format(taken))

    for i, image_set in enumerate(weed_image_sets):
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = 416 #annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        # num_patches = num_w_patches * num_h_patches

        annotated_image_names = []
        for image_name in annotations.keys():
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                annotated_image_names.append(image_name)
        
        num_taken = 0
        taken_regions = {}
        while num_taken < taken[i]:
            
            image_name = random.sample(annotated_image_names, 1)[0]
            
            w_index = random.randrange(0, num_w_patches)
            h_index = random.randrange(0, num_h_patches)

            patch_coords = [
                patch_size * h_index,
                patch_size * w_index,
                min((patch_size * h_index) + patch_size, image_h),
                min((patch_size * w_index) + patch_size, image_w)
            ]

            if image_name not in taken_regions:
                taken_regions[image_name] = []
                taken_regions[image_name].append(patch_coords)
                num_taken += 1
            elif patch_coords not in taken_regions[image_name]:
                taken_regions[image_name].append(patch_coords)
                num_taken += 1

        image_set["taken_regions"] = taken_regions
        image_set["patch_overlap_percent"] = 0

    for image_set in training_image_sets:
        image_set["patch_overlap_percent"] = 0

    log["image_sets"].extend(training_image_sets)
    log["image_sets"].extend(weed_image_sets)



    # for item in [item2]: #[item1, item2]:
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    # run_baseline(item)

    # if run_model:
    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()




def active_patch_selection_old(prev_active_model_log, rand_img_log_to_match, training_image_sets, iteration_number):
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_sets = []
    all_pred_images = []
    # image_set_str_to_image_shape = {}
    # image_set_str_to_patch_size = {}
    image_set_info = {}




    
    annotations_to_match = 0
    patches_to_match = 0
    # if rand_img_log_to_match is not None:
    for image_set in rand_img_log_to_match["image_sets"]:
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

        num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)
        for image_name in image_set["taken_regions"].keys():
            annotations_to_match += len(annotations[image_name]["boxes"])
            patches_to_match += (num_patches_per_image)




















    taken_patches = {}

    if prev_active_model_log is None:
        all_patch_candidates = []
        for image_set in training_image_sets:
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

            # annotations = image_set_info[image_set_str]["annotations"]
            # image_shape = image_set_info[image_set_str]["image_shape"]
            image_height = metadata["images"][list(annotations.keys())[0]]["height_px"]
            image_width = metadata["images"][list(annotations.keys())[0]]["width_px"]
            # image_height = image_shape[0]
            # image_width = image_shape[1]
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            
            # patch_size = image_set_info[image_set_str]["patch_size"]

            incr = patch_size #- overlap_px
            w_covered = max(image_width - patch_size, 0)
            num_w_patches = m.ceil(w_covered / incr) + 1

            h_covered = max(image_height - patch_size, 0)
            num_h_patches = m.ceil(h_covered / incr) + 1


            image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
            print(image_set_str, patch_size)

            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    for i in range(0, num_w_patches):
                        for j in range(0, num_h_patches):
                            patch_min_y = (patch_size) * j
                            patch_min_x = (patch_size) * i

                            patch_max_y = min(patch_min_y + patch_size, image_height)
                            patch_max_x = min(patch_min_x + patch_size, image_width)

                            patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                            all_patch_candidates.append((image_set_str, image_name, patch_region))



        extra_patches = random.sample(all_patch_candidates, patches_to_match)
        for extra_patch in extra_patches:
            image_set_str = extra_patch[0]
            image_name = extra_patch[1]
            extra_patch_coords = extra_patch[2]
            if image_set_str not in taken_patches:
                taken_patches[image_set_str] = {}
            if image_name not in taken_patches[image_set_str]:
                taken_patches[image_set_str][image_name] = []
            taken_patches[image_set_str][image_name].append(extra_patch_coords)


    else:
        for image_set in training_image_sets:
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

            print("switching to model")
            model_dir = os.path.join(image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": prev_active_model_log["model_name"],
                "model_creator": prev_active_model_log["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(3)
                if not os.path.exists(switch_req_path):
                    switch_processed = True


        
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": [image_name for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "regions": [[[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]] for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "save_result": True,
                "regions_only": True,
                "calculate_vegetation_record": False,
                "results_name": "active_learning_iter_" + str(iteration_number),
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)

            # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
            # if len(result_dirs) > 1:
            #     raise RuntimeError("More than one result dir")
            # results_dir = result_dirs[0]


            results_dir = os.path.join(image_set_dir, "model", "results", request_uuid)
            predictions_path = os.path.join(results_dir, "predictions.json")
            predictions = json_io.load_json(predictions_path)

            image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date


            image_set_info[image_set_str] = {}
            image_set_info[image_set_str]["image_shape"] = [metadata["images"][list(annotations.keys())[0]]["height_px"], metadata["images"][list(annotations.keys())[0]]["width_px"]]
            
            
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            image_set_info[image_set_str]["patch_size"] = patch_size

            image_set_info[image_set_str]["annotations"] = annotations
            
            for image_name in predictions.keys():

                for i in range(len(predictions[image_name]["scores"])):
                    all_pred_boxes.append(predictions[image_name]["boxes"][i])
                    all_pred_scores.append(predictions[image_name]["scores"][i])
                    all_pred_image_sets.append(image_set_str)
                    all_pred_images.append(image_name)





        if prev_active_model_log is not None:
            for image_set in prev_active_model_log["image_sets"]:
                image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
                taken_patches[image_set_str] = image_set["taken_regions"]




        print(taken_patches.keys())
        
        for q_score in np.arange(0, 51, 1):
            print("q_score: {}".format(q_score))
            image_q_scores = (abs(0.5 -np.array(all_pred_scores)) * 100).astype(np.int64)
            # q_score_mask = image_q_scores == q_score
            q_score_inds = np.where(image_q_scores == q_score)[0]
            np.random.shuffle(q_score_inds)
            for q_score_ind in q_score_inds:
                candidate_box = np.array(all_pred_boxes[q_score_ind])
                centre = (candidate_box[..., :2] + candidate_box[..., 2:]) / 2.0
                candidate_image_set_str = all_pred_image_sets[q_score_ind]
                candidate_image_name = all_pred_images[q_score_ind]
                image_shape = image_set_info[candidate_image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[candidate_image_set_str]["patch_size"]


                patch_min_y = int((centre[0] // patch_size) * patch_size)
                patch_min_x = int((centre[1] // patch_size) * patch_size)
                patch_max_y = int(min(patch_min_y + patch_size, image_height))
                patch_max_x = int(min(patch_min_x + patch_size, image_width))
                patch = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                already_taken = False
                # if prev_active_model_log is not None:
                    # for image_set in prev_active_model_log["image_sets"]:
                        # if image_set["username"] == username and image_set["farm_name"] == farm_name and image_set["field_name"] == "field_name" and image_set["mission_date"] == "mission_date":
                if candidate_image_set_str not in taken_patches:
                    taken_patches[candidate_image_set_str] = {}
                if candidate_image_name not in taken_patches[candidate_image_set_str]:
                    taken_patches[candidate_image_set_str][candidate_image_name] = []
                if patch in taken_patches[candidate_image_set_str][candidate_image_name]:
                    already_taken = True
                # if not already_taken:
                #     if image_name not in taken_patches[image_set_str]:
                #         taken_patches[image_set_str][image_name] = []
                #     taken_patches[image_set_str][image_name].append(patch)

                cur_total = 0
                if not already_taken:
                    for image_set_str in taken_patches.keys():
                        annotations = image_set_info[image_set_str]["annotations"]
                        for image_name in taken_patches[image_set_str].keys():

                            if image_set_str == candidate_image_set_str and image_name == candidate_image_name:
                                l = taken_patches[image_set_str][image_name].copy() #annotations[candidate_image_name]["training_regions"].copy()
                                l.append(patch)
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                            else:
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                    
                    if cur_total <= annotations_to_match:
                        taken_patches[candidate_image_set_str][candidate_image_name].append(patch)

                    print(cur_total, annotations_to_match, len(taken_patches[candidate_image_set_str][candidate_image_name]))
                        # added_candidates.append((candidates[i][0], candidates[i][1]))
                    
                    # for image_name in image_names:
                    #     if image_name != candidates[i][0]:
                    #         if len(annotations[image_name]["training_regions"]) > 0:
                    #             cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                    # l = annotations[candidates[i][0]]["training_regions"].copy()
                    # l.append(candidates[i][1])
                    # cur_total += box_utils.get_contained_inds(annotations[candidates[i][0]]["boxes"], l).size
                    # if cur_total <= total_to_match:
                    #     annotations[candidates[i][0]]["training_regions"].append(candidates[i][1])
                    #     added_candidates.append((candidates[i][0], candidates[i][1]))
                if cur_total == annotations_to_match:
                    break
            if cur_total == annotations_to_match:
                break

        cur_num_patches = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                cur_num_patches += len(taken_patches[image_set_str][image_name])

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        num_to_add = patches_to_match - cur_num_patches
        if cur_num_patches < patches_to_match:
            all_patch_candidates = []
            for image_set_str in image_set_info.keys():
                annotations = image_set_info[image_set_str]["annotations"]
                # for image_name in taken_patches[image_set_str]:
                image_shape = image_set_info[image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[image_set_str]["patch_size"]

                incr = patch_size #- overlap_px
                w_covered = max(image_width - patch_size, 0)
                num_w_patches = m.ceil(w_covered / incr) + 1

                h_covered = max(image_height - patch_size, 0)
                num_h_patches = m.ceil(h_covered / incr) + 1



                for image_name in annotations.keys():

                    if len(annotations[image_name]["test_regions"]) > 0:
                        for i in range(0, num_w_patches):
                            for j in range(0, num_h_patches):
                                patch_min_y = (patch_size) * j
                                patch_min_x = (patch_size) * i

                                patch_max_y = min(patch_min_y + patch_size, image_height)
                                patch_max_x = min(patch_min_x + patch_size, image_width)

                                patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                                if image_set_str in taken_patches and image_name in taken_patches[image_set_str]:
                                    # if patch_region not in taken_patches[image_set_str][image_name]:
                                        # possible = True
                                        # all_patch_candidates.append(

                                        # )
                                    if patch_region not in taken_patches[image_set_str][image_name]: #patch_region not in annotations[image_name]["training_regions"]:
                                        num_without = box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                                        l = taken_patches[image_set_str][image_name].copy()
                                        l.append(patch_region)
                                        num_with = box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                        if num_without == num_with:
                                            all_patch_candidates.append((image_set_str, image_name, patch_region))


                                else:
                                    num_contained = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_region]).size
                                    if num_contained == 0:
                                        all_patch_candidates.append((
                                            image_set_str,
                                            image_name,
                                            patch_region
                                        ))

            print("picking {} from {} candidates".format(num_to_add, len(all_patch_candidates)))
            extra_patches = random.sample(all_patch_candidates, num_to_add)
            for extra_patch in extra_patches:
                image_set_str = extra_patch[0]
                image_name = extra_patch[1]
                extra_patch_coords = extra_patch[2]
                if image_set_str not in taken_patches:
                    taken_patches[image_set_str] = {}
                if image_name not in taken_patches[image_set_str]:
                    taken_patches[image_set_str][image_name] = []

                taken_patches[image_set_str][image_name].append(extra_patch_coords)

                
        elif cur_num_patches > patches_to_match:
            raise RuntimeError("too many patches -- not implemented")
                
            # random_image_set_str = random.choice

    if prev_active_model_log is not None:
        print("\n\n--- FINAL CHECK---\n\n")
        cur_num_patches = 0
        cur_num_annotations = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                annotations = image_set_info[image_set_str]["annotations"]
                cur_num_patches += len(taken_patches[image_set_str][image_name])
                cur_num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                # for patch_region in taken_patches[image_set_str][image_name]:
                #     if patch_region.shape[0] == 0:
                #         raise RuntimeError("BAD!")

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        if cur_num_patches != patches_to_match:
            raise RuntimeError("cur_num_patches != patches_to_match")


        print("cur_num_annotations", cur_num_annotations)
        print("annotations_to_match", annotations_to_match)
        if cur_num_annotations != annotations_to_match:
            raise RuntimeError("cur_num_annotations != annotations_to_match")

    log = {}
    log["image_sets"] = []
    for image_set_str in taken_patches.keys():
        pieces = image_set_str.split(" ")
        username = pieces[0]
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]

        log["image_sets"].append({
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "taken_regions": taken_patches[image_set_str]
        })


    log["model_creator"] = "erik"
    log["model_name"] = "selected_patches_" + str(iteration_number)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()




def active_patch_selection(training_image_sets, model_name, model_dir_to_match, prev_model_dir): #prev_active_model_log, rand_img_log_to_match, training_image_sets, iteration_number):
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_sets = []
    all_pred_images = []
    # image_set_str_to_image_shape = {}
    # image_set_str_to_patch_size = {}
    image_set_info = {}


    log_to_match_path = os.path.join(model_dir_to_match, "log.json")
    log_to_match = json_io.load_json(log_to_match_path)


    log = {}
    
    annotations_to_match = 0
    patches_to_match = 0
    # if rand_img_log_to_match is not None:
    for image_set in log_to_match["image_sets"]:
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

        # metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        # metadata = json_io.load_json(metadata_path)

        # patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        # num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)
        # for image_name in image_set["taken_regions"].keys():
        #     annotations_to_match += len(annotations[image_name]["boxes"])
        #     patches_to_match += (num_patches_per_image)


        for image_name in image_set["taken_regions"].keys():
            cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], image_set["taken_regions"][image_name]).size
            patches_to_match += len(image_set["taken_regions"][image_name])

    taken_patches = {}

    if prev_model_dir is None:
        log["image_sets"] = log_to_match["image_sets"]

    else:
        prev_active_model_log_path = os.path.join(prev_model_dir, "log.json")
        prev_active_model_log = json_io.load_json(prev_active_model_log_path)
        for image_set in training_image_sets:
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

            print("switching to model")
            model_dir = os.path.join(image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": prev_active_model_log["model_name"],
                "model_creator": prev_active_model_log["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(3)
                if not os.path.exists(switch_req_path):
                    switch_processed = True


        
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": [image_name for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "regions": [[[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]] for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "save_result": True,
                "regions_only": True,
                "calculate_vegetation_record": False,
                "results_name": "active_learning_eval",
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)

            # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
            # if len(result_dirs) > 1:
            #     raise RuntimeError("More than one result dir")
            # results_dir = result_dirs[0]


            results_dir = os.path.join(image_set_dir, "model", "results", request_uuid)
            predictions_path = os.path.join(results_dir, "predictions.json")
            predictions = json_io.load_json(predictions_path)

            image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date


            image_set_info[image_set_str] = {}
            image_set_info[image_set_str]["image_shape"] = [metadata["images"][list(annotations.keys())[0]]["height_px"], metadata["images"][list(annotations.keys())[0]]["width_px"]]
            
            
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            image_set_info[image_set_str]["patch_size"] = patch_size

            image_set_info[image_set_str]["annotations"] = annotations
            
            for image_name in predictions.keys():

                for i in range(len(predictions[image_name]["scores"])):
                    all_pred_boxes.append(predictions[image_name]["boxes"][i])
                    all_pred_scores.append(predictions[image_name]["scores"][i])
                    all_pred_image_sets.append(image_set_str)
                    all_pred_images.append(image_name)





        # if prev_active_model_log is not None:
        for image_set in prev_active_model_log["image_sets"]:
            image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
            taken_patches[image_set_str] = image_set["taken_regions"]




        print(taken_patches.keys())
        
        for q_score in np.arange(0, 51, 1):
            print("q_score: {}".format(q_score))
            image_q_scores = (abs(0.5 -np.array(all_pred_scores)) * 100).astype(np.int64)
            # q_score_mask = image_q_scores == q_score
            q_score_inds = np.where(image_q_scores == q_score)[0]
            np.random.shuffle(q_score_inds)
            for q_score_ind in q_score_inds:
                candidate_box = np.array(all_pred_boxes[q_score_ind])
                centre = (candidate_box[..., :2] + candidate_box[..., 2:]) / 2.0
                candidate_image_set_str = all_pred_image_sets[q_score_ind]
                candidate_image_name = all_pred_images[q_score_ind]
                image_shape = image_set_info[candidate_image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[candidate_image_set_str]["patch_size"]


                patch_min_y = int((centre[0] // patch_size) * patch_size)
                patch_min_x = int((centre[1] // patch_size) * patch_size)
                patch_max_y = int(min(patch_min_y + patch_size, image_height))
                patch_max_x = int(min(patch_min_x + patch_size, image_width))
                patch = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                already_taken = False
                # if prev_active_model_log is not None:
                    # for image_set in prev_active_model_log["image_sets"]:
                        # if image_set["username"] == username and image_set["farm_name"] == farm_name and image_set["field_name"] == "field_name" and image_set["mission_date"] == "mission_date":
                if candidate_image_set_str not in taken_patches:
                    taken_patches[candidate_image_set_str] = {}
                if candidate_image_name not in taken_patches[candidate_image_set_str]:
                    taken_patches[candidate_image_set_str][candidate_image_name] = []
                if patch in taken_patches[candidate_image_set_str][candidate_image_name]:
                    already_taken = True
                # if not already_taken:
                #     if image_name not in taken_patches[image_set_str]:
                #         taken_patches[image_set_str][image_name] = []
                #     taken_patches[image_set_str][image_name].append(patch)

                cur_total = 0
                if not already_taken:
                    for image_set_str in taken_patches.keys():
                        annotations = image_set_info[image_set_str]["annotations"]
                        for image_name in taken_patches[image_set_str].keys():

                            if image_set_str == candidate_image_set_str and image_name == candidate_image_name:
                                l = taken_patches[image_set_str][image_name].copy() #annotations[candidate_image_name]["training_regions"].copy()
                                l.append(patch)
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                            else:
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                    
                    if cur_total <= annotations_to_match:
                        taken_patches[candidate_image_set_str][candidate_image_name].append(patch)

                    print(cur_total, annotations_to_match, len(taken_patches[candidate_image_set_str][candidate_image_name]))
                        # added_candidates.append((candidates[i][0], candidates[i][1]))
                    
                    # for image_name in image_names:
                    #     if image_name != candidates[i][0]:
                    #         if len(annotations[image_name]["training_regions"]) > 0:
                    #             cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                    # l = annotations[candidates[i][0]]["training_regions"].copy()
                    # l.append(candidates[i][1])
                    # cur_total += box_utils.get_contained_inds(annotations[candidates[i][0]]["boxes"], l).size
                    # if cur_total <= total_to_match:
                    #     annotations[candidates[i][0]]["training_regions"].append(candidates[i][1])
                    #     added_candidates.append((candidates[i][0], candidates[i][1]))
                if cur_total == annotations_to_match:
                    break
            if cur_total == annotations_to_match:
                break

        cur_num_patches = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                cur_num_patches += len(taken_patches[image_set_str][image_name])

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        num_to_add = patches_to_match - cur_num_patches
        if cur_num_patches < patches_to_match:
            all_patch_candidates = []
            for image_set_str in image_set_info.keys():
                annotations = image_set_info[image_set_str]["annotations"]
                # for image_name in taken_patches[image_set_str]:
                image_shape = image_set_info[image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[image_set_str]["patch_size"]

                incr = patch_size #- overlap_px
                w_covered = max(image_width - patch_size, 0)
                num_w_patches = m.ceil(w_covered / incr) + 1

                h_covered = max(image_height - patch_size, 0)
                num_h_patches = m.ceil(h_covered / incr) + 1



                for image_name in annotations.keys():

                    if len(annotations[image_name]["test_regions"]) > 0:
                        for i in range(0, num_w_patches):
                            for j in range(0, num_h_patches):
                                patch_min_y = (patch_size) * j
                                patch_min_x = (patch_size) * i

                                patch_max_y = min(patch_min_y + patch_size, image_height)
                                patch_max_x = min(patch_min_x + patch_size, image_width)

                                patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                                if image_set_str in taken_patches and image_name in taken_patches[image_set_str]:
                                    # if patch_region not in taken_patches[image_set_str][image_name]:
                                        # possible = True
                                        # all_patch_candidates.append(

                                        # )
                                    if patch_region not in taken_patches[image_set_str][image_name]: #patch_region not in annotations[image_name]["training_regions"]:
                                        num_without = box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                                        l = taken_patches[image_set_str][image_name].copy()
                                        l.append(patch_region)
                                        num_with = box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                        if num_without == num_with:
                                            all_patch_candidates.append((image_set_str, image_name, patch_region))


                                else:
                                    num_contained = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_region]).size
                                    if num_contained == 0:
                                        all_patch_candidates.append((
                                            image_set_str,
                                            image_name,
                                            patch_region
                                        ))

            print("picking {} from {} candidates".format(num_to_add, len(all_patch_candidates)))
            extra_patches = random.sample(all_patch_candidates, num_to_add)
            for extra_patch in extra_patches:
                image_set_str = extra_patch[0]
                image_name = extra_patch[1]
                extra_patch_coords = extra_patch[2]
                if image_set_str not in taken_patches:
                    taken_patches[image_set_str] = {}
                if image_name not in taken_patches[image_set_str]:
                    taken_patches[image_set_str][image_name] = []

                taken_patches[image_set_str][image_name].append(extra_patch_coords)

                
        elif cur_num_patches > patches_to_match:
            raise RuntimeError("too many patches -- not implemented")
                
            # random_image_set_str = random.choice


        print("\n\n--- FINAL CHECK---\n\n")
        cur_num_patches = 0
        cur_num_annotations = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                annotations = image_set_info[image_set_str]["annotations"]
                cur_num_patches += len(taken_patches[image_set_str][image_name])
                cur_num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                # for patch_region in taken_patches[image_set_str][image_name]:
                #     if patch_region.shape[0] == 0:
                #         raise RuntimeError("BAD!")

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        if cur_num_patches != patches_to_match:
            raise RuntimeError("cur_num_patches != patches_to_match")


        print("cur_num_annotations", cur_num_annotations)
        print("annotations_to_match", annotations_to_match)
        if cur_num_annotations != annotations_to_match:
            raise RuntimeError("cur_num_annotations != annotations_to_match")


        log["image_sets"] = []
        for image_set_str in taken_patches.keys():
            pieces = image_set_str.split(" ")
            username = pieces[0]
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]

            log["image_sets"].append({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "taken_regions": taken_patches[image_set_str]
            })


    log["model_creator"] = "erik"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()



def exg_active_patch_selection(training_image_sets, model_name, model_dir_to_match, prev_model_dir): #prev_active_model_log, rand_img_log_to_match, training_image_sets, iteration_number):
    # all_pred_boxes = []
    # all_pred_scores = []
    # all_pred_image_sets = []
    # all_pred_images = []
    # image_set_str_to_image_shape = {}
    # image_set_str_to_patch_size = {}
    # image_set_info = {}


    log_to_match_path = os.path.join(model_dir_to_match, "log.json")
    log_to_match = json_io.load_json(log_to_match_path)


    log = {}
    


    if prev_model_dir is None:
        log["image_sets"] = log_to_match["image_sets"]

    else:
        candidates = []
        taken_patches = {}

        patches_to_match = get_num_patches_used_by_model(model_dir_to_match) #0
        # for image_set in log_to_match["image_sets"]:
        #     username = image_set["username"]
        #     farm_name = image_set["farm_name"]
        #     field_name = image_set["field_name"]
        #     mission_date = image_set["mission_date"]
        #     image_set_dir = os.path.join("usr", "data", 
        #                                 username, "image_sets",
        #                                 farm_name,
        #                                 field_name,
        #                                 mission_date)
        
        #     # annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        #     # annotations = annotation_utils.load_annotations(annotations_path)

        #     for image_name in image_set["taken_regions"].keys():
        #         # cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], image_set["taken_regions"][image_name]).size
        #         patches_to_match += len(image_set["taken_regions"][image_name])



        prev_active_model_log_path = os.path.join(prev_model_dir, "log.json")
        prev_active_model_log = json_io.load_json(prev_active_model_log_path)


        num_patches_prev_taken = 0
        for image_set in prev_active_model_log["image_sets"]:
            image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
            taken_patches[image_set_str] = image_set["taken_regions"]
            for image_name in image_set["taken_regions"]:
                num_patches_prev_taken += len(image_set["taken_regions"][image_name])


        for image_set in training_image_sets:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
            image_set_dir = os.path.join("usr", "data", 
                                        username, "image_sets",
                                        farm_name,
                                        field_name,
                                        mission_date)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)


            image_names = list(annotations.keys())
            image_w = metadata["images"][image_names[0]]["width_px"]
            image_h = metadata["images"][image_names[0]]["height_px"]

            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

                
            patch_overlap_percent = 0
            overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

            incr = patch_size - overlap_px
            w_covered = max(image_w - patch_size, 0)
            num_w_patches = m.ceil(w_covered / incr) + 1

            h_covered = max(image_h - patch_size, 0)
            num_h_patches = m.ceil(h_covered / incr) + 1


            print("switching to model")
            model_dir = os.path.join(image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": prev_active_model_log["model_name"],
                "model_creator": prev_active_model_log["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(3)
                if not os.path.exists(switch_req_path):
                    switch_processed = True


        
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": [image_name for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "regions": [[[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]] for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "save_result": True,
                "regions_only": True,
                "calculate_vegetation_record": False,
                "results_name": "active_learning_eval",
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)

            # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
            # if len(result_dirs) > 1:
            #     raise RuntimeError("More than one result dir")
            # results_dir = result_dirs[0]


            results_dir = os.path.join(image_set_dir, "model", "results", request_uuid)
            predictions_path = os.path.join(results_dir, "predictions.json")
            predictions = json_io.load_json(predictions_path)


            for image_name in tqdm.tqdm(list(predictions.keys()), desc="Collecting excess green scores"):
                image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
                image = Image(image_path)
                image_array = image.load_image_array()
                exg_array = image_utils.excess_green(image_array)

                pred_boxes = np.array(predictions[image_name]["boxes"])
                pred_scores = np.array(predictions[image_name]["scores"])
                sel_pred_boxes = pred_boxes[pred_scores > 0.5]



                image_w = metadata["images"][image_name]["width_px"]
                image_h = metadata["images"][image_name]["height_px"]
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]
                        if image_set_str not in taken_patches or image_name not in taken_patches[image_set_str] or patch_coords not in taken_patches[image_set_str][image_name]:
                            # if image_set_str in taken_patches and image_name in taken_patches[image_set_str] and patch_coords not in taken_patches[image_set_str][image_name]:

                            
                            inds = box_utils.get_contained_inds(sel_pred_boxes, [patch_coords])
                            patch_boxes = sel_pred_boxes[inds]
                            for patch_box in patch_boxes:
                                exg_array[min(0, patch_box[0]-patch_coords[0]):min(0, patch_box[2]-patch_coords[0]),
                                          min(0, patch_box[1]-patch_coords[1]):min(0, patch_box[3]-patch_coords[1])] = -10
                            
                            sel_vals = exg_array != -10
                            score = np.sum(sel_vals) / sel_vals.size

                            
                            candidates.append(
                                (image_set_str, image_name, patch_coords, score)
                            )
                            
                        
        candidates.sort(key=lambda x: x[3])


        num_patches_to_take = patches_to_match - num_patches_prev_taken

        print("Patch count to match: {}".format(patches_to_match))
        print("Num patches previously taken: {}".format(num_patches_prev_taken))
        print("Num patches to take: {}".format(num_patches_to_take))
        print("Num candidates: {}".format(len(candidates)))

        taken_candidates = candidates[:num_patches_to_take]

        for candidate in taken_candidates:
            image_set_str = candidate[0]
            image_name = candidate[1]
            patch_coords = candidate[2]
            if image_set_str not in taken_patches:
                taken_patches[image_set_str] = {}
            if image_name not in taken_patches[image_set_str]:
                taken_patches[image_set_str][image_name] = []

                
            taken_patches[image_set_str][image_name].append(patch_coords)
            


        print("\n\n--- FINAL CHECK---\n\n")
        cur_num_patches = 0
        # cur_num_annotations = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                # annotations = image_set_info[image_set_str]["annotations"]
                cur_num_patches += len(taken_patches[image_set_str][image_name])
                # cur_num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                # for patch_region in taken_patches[image_set_str][image_name]:
                #     if patch_region.shape[0] == 0:
                #         raise RuntimeError("BAD!")

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        if cur_num_patches != patches_to_match:
            raise RuntimeError("cur_num_patches != patches_to_match")


        # print("cur_num_annotations", cur_num_annotations)
        # print("annotations_to_match", annotations_to_match)
        # if cur_num_annotations != annotations_to_match:
        #     raise RuntimeError("cur_num_annotations != annotations_to_match")


        log["image_sets"] = []
        for image_set_str in taken_patches.keys():
            pieces = image_set_str.split(" ")
            username = pieces[0]
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]

            log["image_sets"].append({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "taken_regions": taken_patches[image_set_str],
                "patch_overlap_percent": 0
            })


    log["model_creator"] = "erik"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)

    if prev_model_dir is not None:
        candidates_dir = os.path.join(pending_model_path, "added_candidates")
        os.makedirs(candidates_dir)
        restructured_taken_candidates = {}
        for candidate in taken_candidates:
            image_set_str = candidate[0]
            image_name = candidate[1]
            patch_coords = candidate[2]
            if image_set_str not in restructured_taken_candidates:
                restructured_taken_candidates[image_set_str] = {}
            if image_name not in restructured_taken_candidates[image_set_str]:
                restructured_taken_candidates[image_set_str][image_name] = []

                
            restructured_taken_candidates[image_set_str][image_name].append(patch_coords)


        for image_set_str in restructured_taken_candidates:
            
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


            for image_name in restructured_taken_candidates[image_set_str]:

                image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
                image = Image(image_path)
                image_array = image.load_image_array()

                for patch_coords in restructured_taken_candidates[image_set_str][image_name]:
                    patch_path = os.path.join(candidates_dir, str(score) + ".png")
                    image_patch = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
                    cv2.imwrite(patch_path, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))




    log_path = os.path.join(pending_model_path, "log.json")
    log["submission_time"] = int(time.time())
    json_io.save_json(log_path, log)

    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()














def rand_img_selection(prev_log, training_image_sets, num_images_to_select, iteration_number, run_model):

    candidates = []
    for training_image_set in training_image_sets:
        username = training_image_set["username"]
        farm_name = training_image_set["farm_name"]
        field_name = training_image_set["field_name"]
        mission_date = training_image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                      username, "image_sets",
                                      farm_name,
                                      field_name,
                                      mission_date)
        

        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                image_id = username + " " + farm_name + " " + field_name + " " + mission_date + " " + image_name
                candidates.append(image_id)
    
    prev_taken = []
    if prev_log is not None:
        for image_set in prev_log["image_sets"]:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            for image_name in image_set["taken_regions"].keys():
                image_id = username + " " + farm_name + " " + field_name + " " + mission_date + " " + image_name
                prev_taken.append(image_id)

    random.shuffle(candidates)
    new_num_taken = 0
    new_taken = []
    # new_taken.extend(prev_taken)
    for i in range(len(candidates)):
        print("checking candidate", candidates[i])
        if candidates[i] not in prev_taken:
            print("\t\tadding candidate", candidates[i])
            new_taken.append(candidates[i])
            new_num_taken += 1
        if new_num_taken >= num_images_to_select:
            break

    # processed_taken = {}
    # for entry in new_taken:
    #     pieces = entry.split(" ")
    #     image_set_str = pieces[0] + " " + pieces[1] + " " + pieces[2] + " " + pieces[3]
    #     if image_set_str not in processed_taken:
    #         processed_taken[image_set_str] = []

    #     processed_taken[image_set_str].append(pieces[4])

    # region_coords = {}
    log = {}
    log["image_sets"] = []
    # if prev_log is not None:
    #     for image_set in prev_log["image_sets"]:
    #         log["image_sets"].append(image_set)
        # username = image_set["username"]
        # farm_name = image_set["farm_name"]
        # field_name = image_set["field_name"]
        # mission_date = image_set["mission_date"]
        # for image_name in image_set["taken_regions"].keys():



    # image_sets = []
    all_taken = []
    all_taken.extend(prev_taken)
    all_taken.extend(new_taken)
    for entry in all_taken: #processed_taken.keys():
        pieces = entry.split(" ")
        username = pieces[0]
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]
        image_name = pieces[4]
        image_set_dir = os.path.join("usr", "data", 
                                username, "image_sets",
                                farm_name,
                                field_name,
                                mission_date)
        
        # annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        # annotations = annotation_utils.load_annotations(annotations_path)
        # patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        # region_coords[image_set_str] = []
        image_height = metadata["images"][image_name]["height_px"]
        image_width = metadata["images"][image_name]["width_px"]
        # num_patches = 0
        # taken_regions = {}
        # for image_name in processed_taken[image_set_str]:
        #     # image_patch_coords = []
        #     image_height = metadata["images"][image_name]["height_px"]
        #     image_width = metadata["images"][image_name]["width_px"]

        #     # taken_regions[image_name] = [0, 0, image_height, image_width]

        #     # num_patches_per_image = diversity_test.get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)

        #     # num_patches += num_patches_per_image


        #     # col_covered = False
        #     # patch_min_y = 0
        #     # while not col_covered:
        #     #     patch_max_y = patch_min_y + patch_size
        #     #     max_content_y = patch_max_y
        #     #     if patch_max_y >= image_height:
        #     #         max_content_y = image_height
        #     #         col_covered = True

        #     #     row_covered = False
        #     #     patch_min_x = 0
        #     #     while not row_covered:

        #     #         patch_max_x = patch_min_x + patch_size
        #     #         max_content_x = patch_max_x
        #     #         if patch_max_x >= image_width:
        #     #             max_content_x = image_width
        #     #             row_covered = True

                    
        #     #         patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

        #     #         image_patch_coords.append(patch_coords)

                
        #     #         patch_min_x += (patch_size)

        #     #     patch_min_y += (patch_size)
        
        #     # region_coords[image_set_str][image_name] = image_patch_coords
        exists = False
        for image_set in log["image_sets"]:

            if image_set["username"] == username and image_set["farm_name"] == farm_name and image_set["field_name"] == "field_name" and image_set["mission_date"] == "mission_date":
                # for image_name in processed_taken.keys():
                image_set["taken_regions"][image_name] = [[0, 0, image_height, image_width]]
                exists = True

        if not exists:
            log["image_sets"].append({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "taken_regions": {image_name: [[0, 0, image_height, image_width]]}, # for image_name in processed_taken.keys()}
                "patch_overlap_percent": 0
            })

            

        # image_sets.append({
        #     "username": username,
        #     "farm_name": farm_name,
        #     "field_name": field_name,
        #     "mission_date": mission_date,
        #     "taken": taken_regions,
        #     "num_patches": num_patches
        # })

    # if prev_log is None:
    #     prev_num = 0
    # else:
    #     prev_num = int(prev_log["model_name"].split("_")[-1])
    # cur_num = num_images_to_select + prev_num

    log["model_creator"] = "erik"
    # log["model_name"] = str(num_images_to_select) ##"set_of_" + str(len(org_image_sets)) + "_no_overlap"
    log["model_name"] = "random_images_" + str(iteration_number) #cur_num)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    # log["image_sets"] = image_sets
    


    # for item in [item2]: #[item1, item2]:
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    # run_baseline(item)

    if run_model:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        print("baseline_queue_size", baseline_queue_size)
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()




def run_random_images_models():
    models_dir = os.path.join("usr", "data", "erik", "models")
    prev_models = glob.glob(os.path.join(models_dir, "pending", "random_images_*"))
    sorted_prev_models = natsorted(prev_models)
    for prev_model in sorted_prev_models:
        log_path = os.path.join(prev_model, "log.json")
        log = json_io.load_json(log_path)
        log["submission_time"] = int(time.time())
        json_io.save_json(log_path, log)




        server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()


def run_my_random_test(training_image_sets, run_models):
    models_dir = os.path.join("usr", "data", "erik", "models")
    for i in range(16):
        if i == 0:
            prev_log = None
        else:
            prev_models = glob.glob(os.path.join(models_dir, "available", "public", "random_images_*"))
            prev_models = glob.glob(os.path.join(models_dir, "pending", "random_images_*"))
            prev_counts = [(x, int(x.split("_")[-1])) for x in prev_models]
            prev_counts.sort(key=lambda x: x[1])
            prev_tup = prev_counts[-1]
            prev_model_path = prev_tup[0]
            prev_log_path = os.path.join(prev_model_path, "log.json")
            prev_log = json_io.load_json(prev_log_path)


        rand_img_selection(prev_log, training_image_sets, 10, i, run_model=run_models)

        
def run_my_active_test(training_image_sets):

    models_dir = os.path.join("usr", "data", "erik", "models")
    for i in range(16):
        if i == 0:
            prev_active_model_log = None
        else:
            prev_models = glob.glob(os.path.join(models_dir, "available", "public", "selected_patches_*"))
            prev_counts = [(x, int(x.split("_")[-1])) for x in prev_models]
            prev_counts.sort(key=lambda x: x[1])
            prev_tup = prev_counts[-1]
            prev_model_path = prev_tup[0]
            prev_log_path = os.path.join(prev_model_path, "log.json")
            prev_active_model_log = json_io.load_json(prev_log_path)


        rand_img_log_to_match_path = os.path.join(models_dir, "available", "public", "random_images_" + str(i), "log.json")
        rand_img_log_to_match_path = os.path.join(models_dir, "pending", "random_images_" + str(i), "log.json")
        rand_img_log_to_match = json_io.load_json(rand_img_log_to_match_path)

        
        active_patch_selection(prev_active_model_log, rand_img_log_to_match, training_image_sets, i)


def remove_previous_iter_results(training_image_sets):
    for image_set in training_image_sets:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
        # if len(result_dirs) != 1:
        #     raise RuntimeError("oh no")
        result_dir = os.path.join(image_set_dir, "model", "results")
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)

def remove_specific_iter_results(training_image_sets, result_name):
    for image_set in training_image_sets:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
        # if len(result_dirs) != 1:
        #     raise RuntimeError("oh no")
        result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
        for result_dir in result_dirs:
            request_path = os.path.join(result_dir, "request.json")
            request = json_io.load_json(request_path)
            if request["results_name"] == result_name:
                shutil.rmtree(result_dir)


if __name__ == "__main__":

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()

    training_image_sets = [
        {
            "username": "kaylie",
            "farm_name": "row_spacing",
            "field_name": "nasser",
            "mission_date": "2021-06-01"
        },
        {
            "username": "kaylie",
            "farm_name": "row_spacing",
            "field_name": "brown",
            "mission_date": "2021-06-01"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Dugout",
            "mission_date": "2022-05-30"
        },
        {
            "username": "kaylie",
            "farm_name": "MORSE",
            "field_name": "Dugout",
            "mission_date": "2022-05-27"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Brown",
            "mission_date": "2021-06-05"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Sutherland",
            "mission_date": "2021-06-05"
        },
        {
            "username": "kaylie",
            "farm_name": "row_spacing",
            "field_name": "nasser2",
            "mission_date": "2022-06-02"
        },
        {
            "username": "kaylie",
            "farm_name": "MORSE",
            "field_name": "Nasser",
            "mission_date": "2022-05-27"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "LowN2",
            "mission_date": "2021-06-07"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim4",
            "mission_date": "2022-05-24"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim5",
            "mission_date": "2022-05-24"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim1",
            "mission_date": "2021-05-26"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim2",
            "mission_date": "2021-05-26"
        },
        {
            "username": "kaylie",
            "farm_name": "Biggar",
            "field_name": "Dennis1",
            "mission_date": "2021-06-04"
        },
        {
            "username": "kaylie",
            "farm_name": "Biggar",
            "field_name": "Dennis3",
            "mission_date": "2021-06-04"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "River",
            "mission_date": "2021-06-09"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "Lake",
            "mission_date": "2021-06-09"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "HornerWest",
            "mission_date": "2021-06-09"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "LowN1",
            "mission_date": "2021-06-07"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko9N",
            "mission_date": "2022-06-07"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim1",
            "mission_date": "2021-06-02"
        },
        {
            "username": "kaylie",
            "farm_name": "row_spacing",
            "field_name": "brown",
            "mission_date": "2021-06-08"
        },
        {
            "username": "kaylie",
            "farm_name": "SaskatoonEast",
            "field_name": "Stevenson5NW",
            "mission_date": "2022-06-20"
        },
        {
            "username": "kaylie",
            "farm_name": "UNI",
            "field_name": "Vaderstad",
            "mission_date": "2022-06-16"
        },
        {
            "username": "kaylie",
            "farm_name": "Biggar",
            "field_name": "Dennis2",
            "mission_date": "2021-06-12"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko10",
            "mission_date": "2022-06-14"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko9S",
            "mission_date": "2022-06-14"
        }
    ]

    weed_image_sets = [
        {
            "username": "erik",
            "farm_name": "CottonWeedDet12",
            "field_name": "CottonWeedDet12",
            "mission_date": "2023-04-10"
        },
        {
            "username": "erik",
            "farm_name": "blue_lupins",
            "field_name": "blue_lupins",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "turnipweed_in_wheat",
            "field_name": "turnipweed_in_wheat",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "chickpea_BFLY",
            "field_name": "chickpea_BFLY",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "chickpea_BFLYS",
            "field_name": "chickpea_BFLYS",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "20200827_wheat_BFLY",
            "field_name": "20200827_wheat_BFLY",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "20201014_wheat_BFLY",
            "field_name": "20201014_wheat_BFLY",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "wheat_BFLYS",
            "field_name": "wheat_BFLYS",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "radish_in_wheat",
            "field_name": "radish_in_wheat",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "narrabri_wheat",
            "field_name": "narrabri_wheat",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "broadleaf_weeds",
            "field_name": "broadleaf_weeds",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "sustainable_ag_2021",
            "field_name": "sustainable_ag_2021",
            "mission_date": "2023-04-11"
        },
        {
            "username": "erik",
            "farm_name": "wild_carrot_canola",
            "field_name": "wild_carrot_canola",
            "mission_date": "2023-04-11"
        },      
    ]

    # remove_previous_iter_results(training_image_sets)

    # run_my_random_test(training_image_sets, run_models=False)
    # run_my_active_test(training_image_sets)

    # training_image_sets.extend(weed_image_sets)


    # run_weed_test(training_image_sets, "MORSE_Nasser_2022-05-27_and_WeedAI")

    # run_weed_test(training_image_sets, weed_image_sets, "MORSE_Nasser_2022-05-27_and_10000_weed", 10000)


    # run_full_image_set_model(training_image_sets, "fixed_epoch_set_of_12_no_overlap")



    # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "pending", "fixed_epoch_set_of_27_no_overlap")
    # prev_model_dir = os.path.join("usr", "data", "erik", "models", "pending", "fixed_epoch_diverse_set_of_27_match_18_no_overlap")

    # run_diverse_model(training_image_sets, "fixed_epoch_diverse_set_of_27_match_27_no_overlap", model_dir_to_match, prev_model_dir)


    # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_MORSE_Nasser_2022-05-27_no_overlap")

    # exg_active_patch_selection(training_image_sets, "fixed_epoch_exg_active_match_1_no_overlap", model_dir_to_match, prev_model_dir=None)


    model_dir = os.path.join("usr", "data", "erik", "models", "pending", "fixed_epoch_diverse_set_of_27_match_12_no_overlap")
    run_pending_model(model_dir)