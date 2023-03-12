import logging
import os
import glob
import random
import math as m
import numpy as np
import urllib3
import time




from models.common import annotation_utils
from io_utils import json_io

import server
from lock_queue import LockQueue



def get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent):
    patch_overlap_percent = 0
    overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

    image_names = list(annotations.keys())
    image_width = metadata["images"][image_names[0]]["width_px"]
    image_height = metadata["images"][image_names[0]]["height_px"]

    incr = patch_size - overlap_px
    w_covered = max(image_width - patch_size, 0)
    num_w_patches = m.ceil(w_covered / incr) + 1

    h_covered = max(image_height - patch_size, 0)
    num_h_patches = m.ceil(h_covered / incr) + 1

    num_patches = num_w_patches * num_h_patches

    return num_patches


def get_num_fully_annotated_images(annotations, image_w, image_h):
    num = 0
    for image_name in annotations.keys():
        if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
            num += 1
    return num



def run(org_image_sets, diverse_image_sets):

    all_image_sets = org_image_sets.copy()
    all_image_sets.extend(diverse_image_sets)

    for image_set in all_image_sets:
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

        num_annotated = get_num_fully_annotated_images(annotations, image_w, image_h)

        num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)

        image_set["num_patches"] = num_patches_per_image * num_annotated

    print("org_image_sets", org_image_sets)
    org_image_sets_patch_num = 0
    for image_set in org_image_sets:
        org_image_sets_patch_num += image_set["num_patches"]
    num_to_match = org_image_sets_patch_num
    # diverse_image_sets_patch_num = 0
    capacities = []
    for image_set in diverse_image_sets:
        capacities.append(image_set["num_patches"])
        # diverse_image_sets_patch_num += image_set["num_patches"]

    capacities = np.array(capacities)
    org_capacities = np.copy(capacities)
    print("num_to_match", num_to_match)
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

    for i, image_set in enumerate(diverse_image_sets):
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

    for image_set in org_image_sets:
        image_set["patch_overlap_percent"] = 0
    
    item1 = {}
    item1["model_creator"] = "erik"
    item1["model_name"] = "set_of_" + str(len(org_image_sets)) + "_no_overlap"
    item1["model_object"] = "canola_seedling"
    item1["public"] = "yes"
    item1["image_sets"] = org_image_sets
    # item1["submission_time"] = int(time.time())


    item2 = {}
    item2["model_creator"] = "erik"
    item2["model_name"] = "diverse_set_of_" + str(len(diverse_image_sets)) + "_match_" + str(len(org_image_sets)) + "_no_overlap"
    item2["model_object"] = "canola_seedling"
    item2["public"] = "yes"
    item2["image_sets"] = diverse_image_sets
    


    for item in [item1, item2]: #[item1, item2]:
        item["submission_time"] = int(time.time())
        
        pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", item["model_name"])

        os.makedirs(pending_model_path, exist_ok=False)
        
        log_path = os.path.join(pending_model_path, "log.json")
        json_io.save_json(log_path, item)

        # run_baseline(item)

        server.sch_ctx["baseline_queue"].enqueue(item)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        print("baseline_queue_size", baseline_queue_size)
        while baseline_queue_size > 0:
        
            item = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(item)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(item)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()







# def run_baseline(item):

#     logger = logging.getLogger(__name__)
#     try:
#         # trace = None
#         username = item["model_creator"]



#         baseline_name = item["model_name"]
#         # baseline_id = item["username"] + ":" + baseline_name

#         # baseline_dir = os.path.join("usr", "data", "baselines", baseline_name) #"training", baseline_name)
        
#         # baselines_dir = os.path.join("usr", "shared", "baselines")
#         usr_dir = os.path.join("usr", "data", username)
#         models_dir = os.path.join(usr_dir, "models")
#         pending_dir = os.path.join(models_dir, "pending")
#         baseline_pending_dir = os.path.join(pending_dir, baseline_name)
#         log_path = os.path.join(baseline_pending_dir, "log.json")

#         available_dir = os.path.join(models_dir, "available")
#         if item["public"] == "yes":
#             baseline_available_dir = os.path.join(available_dir, "public", baseline_name)
#         else:
#             baseline_available_dir = os.path.join(available_dir, "private", baseline_name)
#         # baseline_available_dir = os.path.join(available_dir, baseline_name)

#         aborted_dir = os.path.join(models_dir, "aborted")
#         baseline_aborted_dir = os.path.join(aborted_dir, baseline_name)

#         # pending_dir = os.path.join(baselines_dir, "pending")
#         # baseline_pending_dir = os.path.join(pending_dir, baseline_name)
#         # log_path = os.path.join(baseline_pending_dir, "log.json")

#         # available_dir = os.path.join(baselines_dir, "available")
#         # baseline_available_dir = os.path.join(available_dir, baseline_name)

#         # aborted_dir = os.path.join(baselines_dir, "aborted")

#         # resuming = os.path.exists(log_path)
            
#         #     raise RuntimeError("A baseline with the same name already exists!")

#         if os.path.exists(baseline_available_dir) or os.path.exists(baseline_aborted_dir):
#             return False #raise RuntimeError("A baseline with the same name already exists!")

#         logging.info("Starting to train baseline {}".format(item))
#         # isa.set_scheduler_status(username, "---", "---", "---", isa.TRAINING)

#         patches_dir = os.path.join(baseline_pending_dir, "patches")
#         annotations_dir = os.path.join(baseline_pending_dir, "annotations")
#         model_dir = os.path.join(baseline_pending_dir, "model")
#         training_dir = os.path.join(model_dir, "training")
#         weights_dir = os.path.join(model_dir, "weights")

#         log = json_io.load_json(log_path)
#         # log = {}
#         # if resuming:
#         #     log = json_io.load_json(log_path)
#         # else:
#         if "training_start_time" not in log:
#             #log = {}

#             # log["model_creator"] = item["model_creator"]
#             # log["model_object"] = item["model_object"]
#             # log["public"] = item["public"]
#             # log["model_name"] = item["model_name"]
#             # log["image_sets"] = item["image_sets"]
#             # log["start_time"] = int(time.time())

#             # os.makedirs(baseline_pending_dir)
#             os.makedirs(patches_dir)
#             os.makedirs(annotations_dir)
#             os.makedirs(model_dir)
#             os.makedirs(training_dir)
#             os.makedirs(weights_dir)


#             all_records = []
#             for image_set_index, image_set in enumerate(log["image_sets"]):
#                 logger.info("Baseline: Preparing patches from {}".format(image_set))

#                 username = image_set["username"]
#                 farm_name = image_set["farm_name"]
#                 field_name = image_set["field_name"]
#                 mission_date = image_set["mission_date"]
#                 image_set_dir = os.path.join("usr", "data", username, "image_sets", 
#                                             farm_name, field_name, mission_date)
#                 images_dir = os.path.join(image_set_dir, "images")

#                 # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
#                 # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

#                 metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
#                 metadata = json_io.load_json(metadata_path)
#                 is_ortho = metadata["is_ortho"] == "yes"

#                 annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#                 annotations = annotation_utils.load_annotations(annotations_path)



#                 # image_names = []
#                 # num_annotations = 0
#                 # for image_name in annotations.keys():
#                 #     for region_key in ["training_regions", "test_regions"]:
#                 #         for region in annotations[image_name][region_key]:
#                 #             inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], region)
#                 #             num_annotations += inds.size

#                 num_annotations = annotation_utils.get_num_annotations(annotations, ["training_regions", "test_regions"])
#                     # if annotations[image_name]["status"] == "completed_for_training" or annotations[image_name]["status"] == "completed_for_testing":
#                     #     image_names.append(image_name)
#                     #     num_annotations += annotations[image_name]["boxes"].shape[0]

#                 average_box_area = annotation_utils.get_average_box_area(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
#                 average_box_height = annotation_utils.get_average_box_height(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
#                 average_box_width = annotation_utils.get_average_box_width(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
                

#                 patch_size = annotation_utils.average_box_area_to_patch_size(average_box_area)

#                 # log["image_sets"][image_set_index]["image_names"] = image_names
#                 log["image_sets"][image_set_index]["num_annotations"] = num_annotations

#                 log["image_sets"][image_set_index]["average_box_area"] = average_box_area
#                 log["image_sets"][image_set_index]["average_box_height"] = average_box_height
#                 log["image_sets"][image_set_index]["average_box_width"] = average_box_width
#                 log["image_sets"][image_set_index]["patch_size"] = patch_size

                
#                 logger.info("Patch size: {} px".format(patch_size))

#                 for image_name in annotations.keys():
#                     regions = annotations[image_name]["training_regions"] + annotations[image_name]["test_regions"]
#                     regions = image_set["taken_regions"]
#                     if len(regions) > 0:

#                         image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
#                         image = Image(image_path)
#                         patch_records = ep.extract_patch_records_from_image_tiled(
#                             image,
#                             patch_size,
#                             image_annotations=annotations[image_name],
#                             patch_overlap_percent=50, 
#                             regions=regions,
#                             is_ortho=is_ortho,
#                             include_patch_arrays=False,
#                             out_dir=patches_dir)

#                         # ep.write_patches(patches_dir, patch_records)
#                         all_records.extend(patch_records)

#                 image_set_annotations_dir = os.path.join(annotations_dir, 
#                                                 username, 
#                                                 farm_name,
#                                                 field_name,
#                                                 mission_date)
#                 os.makedirs(image_set_annotations_dir, exist_ok=True)
#                 image_set_annotations_path = os.path.join(image_set_annotations_dir, "annotations.json")
#                 annotation_utils.save_annotations(image_set_annotations_path, annotations)


#             average_box_areas = [log["image_sets"][i]["average_box_area"] for i in range(len(log["image_sets"]))]
#             average_box_heights = [log["image_sets"][i]["average_box_height"] for i in range(len(log["image_sets"]))]
#             average_box_widths = [log["image_sets"][i]["average_box_width"] for i in range(len(log["image_sets"]))]
#             patch_sizes = [log["image_sets"][i]["patch_size"] for i in range(len(log["image_sets"]))]
#             log["average_box_area"] = np.mean(average_box_areas)
#             log["average_box_height"] = np.mean(average_box_heights)
#             log["average_box_width"] = np.mean(average_box_widths)
#             log["average_patch_size"] = np.mean(patch_sizes)

#             patch_records = np.array(all_records)

#             training_patch_records = patch_records

#             # training_size = round(patch_records.size * 0.8)
#             # training_subset = random.sample(np.arange(patch_records.size).tolist(), training_size)

#             # training_patch_records = patch_records[training_subset]
#             # validation_patch_records = np.delete(patch_records, training_subset)

#             training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
#             training_patches_record_path = os.path.join(training_dir, "training-patches-record.tfrec")
#             tf_record_io.output_patch_tf_records(training_patches_record_path, training_tf_records)

#             # validation_tf_records = tf_record_io.create_patch_tf_records(validation_patch_records, patches_dir, is_annotated=True)
#             # validation_patches_record_path = os.path.join(training_dir, "validation-patches-record.tfrec")
#             # tf_record_io.output_patch_tf_records(validation_patches_record_path, validation_tf_records)

#             loss_record = {
#                 "training_loss": { "values": [] }
#             }
#             loss_record_path = os.path.join(baseline_pending_dir, "model", "training", "loss_record.json")
#             json_io.save_json(loss_record_path, loss_record)

#             image_set_aux.reset_loss_record(baseline_pending_dir)

#             log["training_start_time"] = int(time.time())
#             json_io.save_json(log_path, log)


#         # q = mp.Queue()
#         # p = MyProcess(target=yolov4_image_set_driver.train_baseline, 
#         #                 args=(sch_ctx, baseline_pending_dir, q))
#         # p.start()
#         # p.join()

#         # if p.exception:
#         #     exception, trace = p.exception
#         #     raise exception

#         # training_finished = q.get()

#         training_finished = yolov4_image_set_driver.train_baseline(sch_ctx, baseline_pending_dir)

#         if training_finished:
#             # log = json_io.load_json(log_path)

#             log["training_end_time"] = int(time.time())
#             json_io.save_json(log_path, log)
            
#             shutil.move(os.path.join(weights_dir, "best_weights.h5"),
#                         os.path.join(baseline_pending_dir, "weights.h5"))

#             shutil.rmtree(patches_dir)
#             shutil.rmtree(model_dir)

#             shutil.move(baseline_pending_dir, baseline_available_dir)

#             isa.emit_model_change(item["model_creator"])

#             return False
#         else:
#             return True


#     except Exception as e:
#         # if trace is None:
#         trace = traceback.format_exc()
#         logger.error("Exception occurred in process_baseline")
#         logger.error(e)
#         logger.error(trace)

#         try:
#             if baseline_pending_dir is not None:

                


#                 log["aborted_time"] = int(time.time())
#                 log["error_message"] = str(e)

#                 os.makedirs(baseline_aborted_dir)
#                 json_io.save_json(os.path.join(baseline_aborted_dir, "log.json"), log)

#                 if os.path.exists(baseline_pending_dir):
#                     saved_pending_dir = os.path.join(baseline_aborted_dir, "saved_pending")
#                     shutil.move(baseline_pending_dir, saved_pending_dir)
#                     # shutil.rmtree(baseline_pending_dir)
#                 if os.path.exists(baseline_available_dir):
#                     saved_available_dir = os.path.join(baseline_aborted_dir, "saved_available")
#                     shutil.move(baseline_pending_dir, saved_available_dir)
#                     # shutil.rmtree(baseline_available_dir)

#             isa.emit_model_change(item["model_creator"])
#             #json_io.save_json(sys_block_path, {"error_message": str(e)})

#             # isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_TRAINING,
#             #                      extra_items={"error_setting": "training", "error_message": str(e)})
#         except Exception as e:
#             trace = traceback.format_exc()
#             logger.error("Exception occurred while handling original exception")
#             logger.error(e)
#             logger.error(trace)

#     # shutil.move(os.path.join(weights_dir, "best_weights.h5"),
#     #             os.path.join("usr", "additional", "baselines", baseline_name + ".h5"))
#     # shutil.rmtree(baseline_dir)
#     # if completed:
#     #     shutil.move("usr/data/baselines/training/" + baseline_name,
#     #                 "usr/data/baselines/completed/" + baseline_name)

#     # baseline_log["end_time"] = time.time()

#     # log_dir = os.path.join("usr", "data", "baseline_logs")
#     # if not os.path.exists(log_dir):
#     #     os.makedirs(log_dir)
#     # baseline_log_path = os.path.join(log_dir, baseline_name + ".json")
#     # json_io.save_json(baseline_log_path, baseline_log)








# def run(image_sets, available_image_sets):

#     patch_size = 416

#     total_patches = 0    
#     for image_set in image_sets:

#         image_set_dir = os.path.join("usr", "data", image_set["username"],
#                                      image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
        
#         annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#         annotations = json_io.load_json(annotations_path)
#         num_annotated_images = 0
#         for image_name in annotations.keys():
#             if len(annotations[image_name]["training_regions"]) > 0 or len(annotations[image_name]["test_regions"]) > 0:
#                 num_annotated_images += 1


#         metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
#         metadata = json_io.load_json(metadata_path)


#         num_patches = image_set_patch_nums[i]
#         patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

#         num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)

#         total_patches += (num_patches_per_image * num_annotated_images)


#     num_patches_per_image_set = m.floor(total_patches / len(available_image_sets))
#     num_additional = total_patches % len(available_image_sets)

#     image_set_patch_nums = []
#     for i in range(len(available_image_sets)):
#         num = num_patches_per_image_set
#         if i < num_additional:
#             num += 1
#         image_set_patch_nums.append(num)


#     for i, image_set in enumerate(available_image_sets):

#         image_set_dir = os.path.join("usr", "data", image_set["username"],
#                                      image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
        
#         annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#         annotations = json_io.load_json(annotations_path)

#         num_patches = image_set_patch_nums[i]
#         patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"]) #region_keys)
#         num_annotated_images = 0
#         for image_name in annotations.keys():
#             if len(annotations[image_name]["training_regions"]) > 0 or len(annotations[image_name]["test_regions"]) > 0:
#                 num_annotated_images += 1



if __name__ == "__main__":

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()

    org_image_sets = [
        {
            "username": "kaylie",
            "farm_name": "MORSE",
            "field_name": "Nasser",
            "mission_date": "2022-05-27"
        },
        {
            "username": "kaylie",
            "farm_name": "row_spacing",
            "field_name": "brown",
            "mission_date": "2021-06-01"
        },
        {
            "username": "kaylie",
            "farm_name": "BlaineLake",
            "field_name": "Lake",
            "mission_date": "2021-06-09"
        },
        {
            "username": "kaylie",
            "farm_name": "Saskatoon",
            "field_name": "Norheim5",
            "mission_date": "2022-05-24"
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
            "field_name": "Serhienko9N",
            "mission_date": "2022-06-07"
        }
    ]

    diverse_image_sets = [
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

    run(org_image_sets, diverse_image_sets)