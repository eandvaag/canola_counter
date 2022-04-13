import logging
import os
import glob
import tqdm

import random
import math as m
import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances

import extract_patches as ep
from graph import graph_match, graph_model
from image_set import DataSet

import manual_descriptor as md


from io_utils import w3c_io, tf_record_io


def build_inference_datasets(config):
    targets = [
        {
            "farm_name": config.inference["target_farm_name"],
            "field_name": config.inference["target_field_name"],
            "mission_date": config.inference["target_mission_date"]
        }
    ]

    if "supplementary_targets" in config.inference:
        for target_record in config.inference["supplementary_targets"]:
            targets.append(
                {
                    "farm_name": target_record["target_farm_name"],
                    "field_name": target_record["target_field_name"],
                    "mission_date": target_record["target_mission_date"]
                }
            )


    
    model_dir = os.path.join("usr", "data", "models", config.arch["model_uuid"])
    
    for target in targets:
        target_farm_name = target["farm_name"]
        target_field_name = target["field_name"]
        target_mission_date = target["mission_date"]
        annotations_path = os.path.join("usr", "data", "image_sets", 
                                        target_farm_name, target_field_name, target_mission_date,
                                        "annotations", "annotations_w3c.json")

        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

        patch_dir = os.path.join(model_dir, "target_patches", 
                                 target_farm_name + "_" + target_field_name + "_" + target_mission_date)
        os.makedirs(patch_dir)
        annotated_patches_record_path = os.path.join(patch_dir, "annotated-patches-record.tfrec")
        unannotated_patches_record_path = os.path.join(patch_dir, "unannotated-patches-record.tfrec")

        annotated_tf_records = []
        unannotated_tf_records = []

        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
        patch_size = w3c_io.get_patch_size(annotations)

        dataset = DataSet({
            "farm_name": target_farm_name,
            "field_name": target_field_name,
            "mission_date": target_mission_date
        })

        if config.inference["predict_on_completed_only"]:
            images = dataset.completed_images
        else:
            images = dataset.images

        for image in tqdm.tqdm(images, desc="Extracting target patches"):

            is_annotated = annotations[image.image_name]["status"] == "completed"
            image_patches = ep.extract_patch_records_from_image_tiled(image, patch_size, annotations[image.image_name])

            ep.write_patches(patch_dir, image_patches)
            tf_records_for_image = tf_record_io.create_patch_tf_records(image_patches, patch_dir, is_annotated=is_annotated)

            if is_annotated:
                annotated_tf_records.extend(tf_records_for_image)
            else:
                unannotated_tf_records.extend(tf_records_for_image) 

        tf_record_io.output_patch_tf_records(annotated_patches_record_path, annotated_tf_records)
        tf_record_io.output_patch_tf_records(unannotated_patches_record_path, unannotated_tf_records)


def build_training_datasets(config):

    #two_phase_patches(config)
    #return

    source_construction_params = config.training["source_construction_params"]
    method_name = source_construction_params["method_name"]
    if method_name == "even_subset":
        build_even_subset(config)
    elif method_name == "graph_subset":
        two_phase_patches(config)
        #build_graph_subset(config)
    elif method_name == "graph_subset_basic":
        build_graph_subset(config)
    elif method_name == "direct":
        build_direct(config)
    else:
        raise RuntimeError("Unrecognized source dataset construction method: {}".format(method_name))



def get_num_completed_source_images(config):
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]    

    image_set_root = os.path.join("usr", "data", "image_sets")

    num_completed_source_images = 0
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if not ((farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date):
                    
                    annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                             "annotations", "annotations_w3c.json")
                    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
                    num_completed_source_images += len(w3c_io.get_completed_images(annotations))

    return num_completed_source_images



def determine_number_of_patches_for_each_image(image_names, annotations, total_plants_needed, total_other_needed, allow_box_reuse=True):

    if len(image_names) == 0:
        raise RuntimeError("Number of images is zero")

    completed_images = w3c_io.get_completed_images(annotations)
    assigned = {}
    #left = {}

    base_other_num_per_image =  total_other_needed // len(image_names)
    for i, image_name in enumerate(image_names):

        assigned[image_name] = {}
        assigned[image_name]["plant"] = 0
        assigned[image_name]["other"] = base_other_num_per_image
        if i < total_other_needed % len(image_names):
            assigned[image_name]["other"] += 1

    #print("completed_images", completed_images)


    
    
    # def take_boxes(total_plants_needed, left_arr, assigned_arr):
    #     while total_plants_needed > 0:
    #         mask = left_arr > 0
    #         if not mask.any():
    #             break

    #         m.ceil(total_plants_needed / left_arr.size)
    #         min_rem = min(total_plants_needed / left_arr.size, np.min(left_arr[mask]))

    #         left_arr[mask] -= min_rem
    #         assigned_arr[mask] += min_rem
    #         total_plants_needed -= min_rem

    #     return total_plants_needed, left_arr, assigned_arr
    
    print("total_needed", total_plants_needed)
    
    left_arr = np.array([annotations[image_name]["boxes"].shape[0] for image_name in completed_images])
    assigned_arr = np.array([0] * len(completed_images))
    print("left_arr", left_arr)
    while total_plants_needed > 0:
        print("left_arr", left_arr)
        mask = left_arr > 0
        

        if not mask.any():
            if allow_box_reuse:
                left_arr = np.array([annotations[image_name]["boxes"].shape[0] for image_name in completed_images])
                print("replenesh left_arr", left_arr)
                mask = left_arr > 0
            else:
                raise RuntimeError("Insufficient number of annotations to perform assignment")

        
        print("mask", mask)

        per_image_needed = np.array([total_plants_needed // left_arr[mask].size] * left_arr[mask].size)
        per_image_needed[:total_plants_needed % left_arr[mask].size] += 1
        per_image_needed_full = np.zeros(left_arr.size, dtype=np.int64)
        per_image_needed_inds = np.where(left_arr > 0)[0]
        per_image_needed_full[per_image_needed_inds] = per_image_needed
        print("per_image_needed_full", per_image_needed_full)

        #taken = left_arr - per_image_needed_full
        #taken = np.clip(left_arr )
        reduction = left_arr - per_image_needed_full
        taken = np.clip(per_image_needed_full, None, left_arr)
        total_plants_needed = 0 #-= total_plants_needed
        assigned_arr += taken #np.clip(taken, 0, None)
        redistribute = (-1) * np.sum(reduction[reduction < 0])
        total_plants_needed += redistribute
        left_arr -= taken #np.clip(taken, 0, None)







    # total_plants_needed, left_arr, assigned_arr = take_boxes(total_plants_needed, left_arr, assigned_arr)
    # while total_plants_needed > 0:
    #     if allow_box_reuse:
    #         left_arr = np.array([annotations[image_name]["boxes"].shape[0] for image_name in completed_images])
    #         total_plants_needed, left_arr, assigned_arr = take_boxes(total_plants_needed, left_arr, assigned_arr)
    #     else:
    #         raise RuntimeError("Insufficient number of annotations to perform assignment")


    print("assigned_arr", assigned_arr)

    for i, image_name in enumerate(completed_images):
        assigned[image_name]["plant"] = assigned_arr[i]
    # i = 0
    # found = False
    # while total_plants_needed > 0:
    #     if left[completed_images[i]] > 0:
    #         total_plants_needed -= 1
    #         left[completed_images[i]] -= 1
    #         assigned[completed_images[i]]["plant"] += 1
    #         found = True
    #     i = (i + 1) % len(completed_images)
    #     if i == 0:
    #         if not found:
    #             raise RuntimeError("Insufficient number of annotations available")
    #         else:
    #             found = False
    
    return assigned



def two_phase_patches(config):

    logger = logging.getLogger(__name__)

            

    image_set_root = os.path.join("usr", "data", "image_sets")

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    source_construction_params = config.training["source_construction_params"]
    desired_source_size = source_construction_params["size"]

    prop_plant_patches = 0.85
    desired_plant_size = round(desired_source_size * prop_plant_patches)
    desired_other_size = desired_source_size - desired_plant_size

    #method_params = source_construction_params["method_params"]
    #extraction_type = method_params["extraction_type"]

    patch_size = "image_set_dependent"
    extraction_type = None

    pool_1_per_image_set_plant_patches = 25
    pool_1_per_image_set_other_patches = 25

    source_plant_patches = []
    source_other_patches = []
    target_plant_patches = []
    target_other_patches = []
    source_datasets = []
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                is_target = farm_name == target_farm_name and \
                            field_name == target_field_name and \
                            mission_date == target_mission_date


                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })
                print("now processing", dataset.image_set_name)

                annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
                try:
                    if patch_size == "image_set_dependent":
                        image_set_patch_size = w3c_io.get_patch_size(annotations)
                    else:
                        image_set_patch_size = patch_size
    
                    if is_target:
                        image_names = annotations.keys()
                        images = dataset.images
                    else:
                        image_names = w3c_io.get_completed_images(annotations)
                        images = dataset.completed_images

                    assigned = determine_number_of_patches_for_each_image(image_names, annotations, 
                                pool_1_per_image_set_plant_patches, pool_1_per_image_set_other_patches, allow_box_reuse=False)



                    for image in images:
                        if assigned[image.image_name]["plant"] > 0 or assigned[image.image_name]["other"] > 0:
                            plant_patches, other_patches = ep.extract_plant_and_other(
                                image, 
                                annotations[image.image_name],
                                assigned[image.image_name]["plant"],
                                assigned[image.image_name]["other"],
                                image_set_patch_size
                            )

                            if is_target:
                                target_plant_patches.extend(plant_patches)
                                target_other_patches.extend(other_patches)
                            else:
                                source_plant_patches.extend(plant_patches)
                                source_other_patches.extend(other_patches)
                        
                    if is_target:
                        target_dataset = dataset
                    else:
                        source_datasets.append(dataset)
                    

                except RuntimeError:
                    if is_target:
                        raise RuntimeError("Need annotations in target image set to determine image set dependent patch size.")
                    else:
                        pass # not enough annotations in this image set

    logger.info("all image sets have been processed")

    source_plant_patches = np.array(source_plant_patches)
    target_plant_patches = np.array(target_plant_patches)

    source_plant_features = extract_patch_features(source_plant_patches, extraction_type, config)
    target_plant_features = extract_patch_features(target_plant_patches, extraction_type, config)

    epsilon = 1e-10
    distances = pairwise_distances(target_plant_features, source_plant_features, metric="euclidean") + epsilon
    k = pool_1_per_image_set_plant_patches
    smallest_dist_ind = np.argsort(distances)[:, :k]
    smallest_dist = np.take_along_axis(distances, smallest_dist_ind, axis=1)
    plant_similarities = np.zeros(shape=np.shape(distances))
    
    largest_sim = (1 / smallest_dist)
    np.put_along_axis(plant_similarities, smallest_dist_ind, largest_sim, axis=1)
    total_plant_sim_sum = np.sum(plant_similarities)

    source_other_patches = np.array(source_other_patches)
    target_other_patches = np.array(target_other_patches)

    source_other_features = extract_patch_features(source_other_patches, extraction_type, config)
    target_other_features = extract_patch_features(target_other_patches, extraction_type, config)
    distances = pairwise_distances(target_other_features, source_other_features, metric="euclidean") + epsilon
    k = pool_1_per_image_set_other_patches
    smallest_dist_ind = np.argsort(distances)[:, :k]
    smallest_dist = np.take_along_axis(distances, smallest_dist_ind, axis=1)
    other_similarities = np.zeros(shape=np.shape(distances))
    
    largest_sim = (1 / smallest_dist)
    np.put_along_axis(other_similarities, smallest_dist_ind, largest_sim, axis=1)
    total_other_sim_sum = np.sum(other_similarities)

    total_pool_size = 25000
    source_pool_plant_size = round(prop_plant_patches * total_pool_size) #  10000
    source_pool_other_size = total_pool_size - source_pool_plant_size #10000
    source_plant_patches = []
    source_other_patches = []
    target_plant_patches = []
    target_other_patches = []


    for i in range(len(source_datasets)):
        subset_plant_sim = plant_similarities[:, pool_1_per_image_set_plant_patches*i:pool_1_per_image_set_plant_patches*(i+1)]
        print("subset_plant_sim sum", np.sum(subset_plant_sim))
        image_set_plant_prob = np.sum(subset_plant_sim) / total_plant_sim_sum
        image_set_plant_total = m.ceil(source_pool_plant_size * image_set_plant_prob)


        subset_other_sim = other_similarities[:, pool_1_per_image_set_other_patches*i:pool_1_per_image_set_other_patches*(i+1)]
        print("subset_other_sim sum", np.sum(subset_other_sim))
        image_set_other_prob = np.sum(subset_other_sim) / total_other_sim_sum
        image_set_other_total = m.ceil(source_pool_other_size * image_set_other_prob)        
        #image_set_num_per_image = m.ceil(image_set_total / len(source_datasets[i].completed_images))
        print("{} | plant prob: {} | total plant taking: {} | other prob: {} | total other taking: {}".format(
            source_datasets[i].image_set_name, image_set_plant_prob, image_set_plant_total, 
            image_set_other_prob, image_set_other_total))


        annotations = w3c_io.load_annotations(source_datasets[i].annotations_path, {"plant": 0})
        if patch_size == "image_set_dependent":
            image_set_patch_size = w3c_io.get_patch_size(annotations)
        else:
            image_set_patch_size = patch_size

        image_names = w3c_io.get_completed_images(annotations)
        images = source_datasets[i].completed_images
        assigned = determine_number_of_patches_for_each_image(image_names, annotations, 
                        image_set_plant_total, image_set_other_total)

        #print("assigned", assigned)
        print("assigned plant total", np.sum([assigned[image_name]["plant"] for image_name in assigned.keys()]))
        print("assigned other total", np.sum([assigned[image_name]["other"] for image_name in assigned.keys()]))
        for image in images:
            if assigned[image.image_name]["plant"] > 0 or assigned[image.image_name]["other"] > 0:
                plant_patches, other_patches = ep.extract_plant_and_other(
                                image, 
                                annotations[image.image_name], 
                                assigned[image.image_name]["plant"],
                                assigned[image.image_name]["other"],
                                image_set_patch_size
                            )
            
                source_plant_patches.extend(plant_patches)
                source_other_patches.extend(other_patches)
        

    annotations = w3c_io.load_annotations(target_dataset.annotations_path, {"plant": 0})
    if patch_size == "image_set_dependent":
        image_set_patch_size = w3c_io.get_patch_size(annotations)
    else:
        image_set_patch_size = patch_size
    for image in target_dataset.images:
        plant_patches, other_patches = ep.extract_plant_and_other(
                        image, 
                        annotations[image.image_name], 
                        "all",
                        m.ceil(desired_other_size / len(target_dataset.images)),
                        image_set_patch_size
                    )
        target_plant_patches.extend(plant_patches)
        target_other_patches.extend(other_patches)
    

    source_plant_patches = np.array(source_plant_patches)
    source_other_patches = np.array(source_other_patches)
    target_plant_patches = np.array(target_plant_patches)
    target_other_patches = np.array(target_other_patches)
    if target_plant_patches.shape[0] > desired_plant_size: #4000:
        subset_inds = np.random.choice(np.arange(target_plant_patches.shape[0]), desired_plant_size) #4000)
        target_plant_patches = target_plant_patches[subset_inds]
    subset_inds = np.random.choice(np.arange(target_other_patches.shape[0]), desired_other_size) #1000)
    target_other_patches = target_other_patches[subset_inds]

    print("source_plant_patches.size", source_plant_patches.size)
    print("source_other_patches.size", source_other_patches.size)
    print("target_plant_patches.size", target_plant_patches.size)
    print("target_other_patches.size", target_other_patches.size)



    source_plant_features = extract_patch_features(source_plant_patches, extraction_type, config)
    target_plant_features = extract_patch_features(target_plant_patches, extraction_type, config)
    source_other_features = extract_patch_features(source_other_patches, extraction_type, config)
    target_other_features = extract_patch_features(target_other_patches, extraction_type, config)

    selected_plant_source_patches = graph_match.bipartite_b_match(config, desired_plant_size, source_plant_features, target_plant_features, source_plant_patches, target_plant_patches)
    selected_other_source_patches = graph_match.bipartite_b_match(config, desired_other_size, source_other_features, target_other_features, source_other_patches, target_other_patches)
    

    print("selected_plant_source_patches.size", selected_plant_source_patches.size)
    print("seleted_other_source_patches.size", selected_other_source_patches.size)
    patches = np.concatenate([selected_plant_source_patches, selected_other_source_patches])
    print("patches.size", patches.size)
    usr_data_root = os.path.join("usr", "data")
    patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
    training_patches_dir = os.path.join(patches_dir, "training")
    validation_patches_dir = os.path.join(patches_dir, "validation")
    os.makedirs(training_patches_dir)
    os.makedirs(validation_patches_dir)        
            

    training_size = round(patches.size * 0.8)
    training_subset = random.sample(np.arange(patches.size).tolist(), training_size)

    training_patches = patches[training_subset]
    validation_patches = np.delete(patches, training_subset)

    logger.info("Writing patches...")
    ep.write_annotated_patch_records(training_patches, training_patches_dir)
    ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")


def two_phase_exg_patches(config, combo):


    if combo:
        extraction_func = ep.extract_patch_records_with_exg_box_combo
    else:
        extraction_func = ep.extract_patch_records_with_exg

    source_construction_params = config.training["source_construction_params"]
    method_params = source_construction_params["method_params"]
    extraction_type = method_params["extraction_type"]
    patch_size = method_params["patch_size"]
    #source_pool_size = method_params["source_pool_size"]
    #target_pool_size = method_params["target_pool_size"]
    source_pool_size = 25000
    target_pool_size = source_construction_params["size"]

    image_set_root = os.path.join("usr", "data", "image_sets")

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]


    pool_1_num_per_source_image_set = 50
    #pool_1_num_target_image_set = 50


    data = {
        "source_patches": [],
        "target_patches": [],
        "source_features": [],
        "target_features": []
    }

    source_datasets = []
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                is_target = farm_name == target_farm_name and \
                            field_name == target_field_name and \
                            mission_date == target_mission_date


                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })
                print("now processing", dataset.image_set_name)

                annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})

                try:
                    if patch_size == "image_set_dependent":
                        image_set_patch_size = w3c_io.get_patch_size(annotations)
                    else:
                        image_set_patch_size = patch_size

                    if is_target:
                        num_patches_per_target_image = m.ceil(target_pool_size / len(dataset.images))
                        image_set_target_patches = []
                        for image in dataset.images:
                            patches = extraction_func( #ep.extract_patch_records_with_exg(
                                            image, 
                                            annotations[image.image_name], 
                                            num_patches_per_target_image, 
                                            image_set_patch_size)
                            print("adding {} patches to target dataset".format(len(patches)))
                            image_set_target_patches.extend(patches)
                        data["target_patches"].extend(image_set_target_patches[:target_pool_size])
                    else:
                        
                        if len(dataset.completed_images) > 0:
                            source_datasets.append(dataset)
                            
                            num_patches_per_source_image = m.ceil(pool_1_num_per_source_image_set / len(dataset.completed_images))
                            print("want: {} | have {} images | taking {} per image".format(
                                pool_1_num_per_source_image_set, len(dataset.completed_images), num_patches_per_source_image))
                            image_set_source_patches = []
                            for image in dataset.completed_images:
                            
                                # patches = ep.extract_patch_records_from_image_tiled(
                                #     image, 
                                #     image_set_patch_size, 
                                #     annotations[image.image_name], 
                                #     patch_overlap_percent=50, 
                                #     starting_patch_num=0)
                                patches = extraction_func(
                                                image, 
                                                annotations[image.image_name], 
                                                num_patches_per_source_image, 
                                                image_set_patch_size)
                                print("adding {} patches to source dataset".format(len(patches)))
                                image_set_source_patches.extend(patches)
                            data["source_patches"].extend(image_set_source_patches[:pool_1_num_per_source_image_set])
                            
                        
                except RuntimeError:
                    if is_target:
                        raise RuntimeError("Need annotations in target image set to determine image set dependent patch size.")
                    else:
                        pass # patch size cannot be determined due to 0 annotations

    data["source_patches"] = np.array(data["source_patches"])
    data["target_patches"] = np.array(data["target_patches"])

    data["source_features"] = extract_patch_features(data["source_patches"], extraction_type, config)
    data["target_features"] = extract_patch_features(data["target_patches"], extraction_type, config)
    # refine_target = False
    # if refine_target:
    #    pass



    epsilon = 1e-10
    distances = pairwise_distances(data["target_features"], data["source_features"], metric="euclidean") + epsilon
    k = pool_1_num_per_source_image_set
    smallest_dist_ind = np.argsort(distances)[:, :k]
    smallest_dist = np.take_along_axis(distances, smallest_dist_ind, axis=1)
    similarities = np.zeros(shape=np.shape(distances))
    
    largest_sim = (1 / smallest_dist)
    np.put_along_axis(similarities, smallest_dist_ind, largest_sim, axis=1)

    #similarities = (1 / (distances))



    # or take top-k (top-1?) matches for each target, then assign probability based on originating source set
    
    print("similarities", similarities)
    total_sim_sum = np.sum(similarities)

    data["source_patches"] = []

    for i in range(len(source_datasets)):
        subset_sim = similarities[:, pool_1_num_per_source_image_set*i:pool_1_num_per_source_image_set*(i+1)]
        print("subset_sim sum", np.sum(subset_sim))
        image_set_prob = np.sum(subset_sim) / total_sim_sum
        image_set_total = m.ceil(source_pool_size * image_set_prob)
        image_set_num_per_image = m.ceil(image_set_total / len(source_datasets[i].completed_images))
        print("{} | prob: {} | total taking: {} | per image taking: {}".format(
            source_datasets[i].image_set_name, image_set_prob, image_set_total, image_set_num_per_image))
        
        if image_set_num_per_image > 0:
            annotations = w3c_io.load_annotations(source_datasets[i].annotations_path, {"plant": 0})
            if patch_size == "image_set_dependent":
                image_set_patch_size = w3c_io.get_patch_size(annotations)
            else:
                image_set_patch_size = patch_size
            
            image_set_source_patches = []
            for image in source_datasets[i].completed_images:
                patches = extraction_func(
                                image, 
                                annotations[image.image_name], 
                                image_set_num_per_image, 
                                image_set_patch_size)
            
                image_set_source_patches.extend(patches)
            data["source_patches"].extend(image_set_source_patches)
        
    data["source_patches"] = np.array(data["source_patches"])
    return data



def exg_patches(config, combo):

    logger = logging.getLogger(__name__)

    if combo:
        extraction_func = ep.extract_patch_records_with_exg_box_combo
    else:
        extraction_func = ep.extract_patch_records_with_exg


    image_set_root = os.path.join("usr", "data", "image_sets")

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]
    
    source_construction_params = config.training["source_construction_params"]
    method_params = source_construction_params["method_params"]
    patch_size = method_params["patch_size"]
    source_pool_size = method_params["source_pool_size"]
    target_pool_size = method_params["target_pool_size"]
    num_completed_source_images = get_num_completed_source_images(config)
    num_patches_per_source_image = m.ceil(source_pool_size / num_completed_source_images)
    logger.info("Extracting {} patches in each annotated source image".format(num_patches_per_source_image))
    
    target_annotations_path = os.path.join(image_set_root, 
                                     target_farm_name, target_field_name, target_mission_date,
                                     "annotations", "annotations_w3c.json")
    target_annotations = w3c_io.load_annotations(target_annotations_path, {"plant": 0})
    num_target_images = len(target_annotations.keys())
    num_patches_per_target_image = m.ceil(target_pool_size / num_target_images)
    logger.info("Extracting {} patches in each target image".format(num_patches_per_target_image))


    data = {
        "source_patches": [],
        "target_patches": []
    }

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                is_target = farm_name == target_farm_name and \
                            field_name == target_field_name and \
                            mission_date == target_mission_date


                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })

                annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                             "annotations", "annotations_w3c.json")
                annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

                try:
                    if patch_size == "image_set_dependent":
                        image_set_patch_size = w3c_io.get_patch_size(annotations)
                    else:
                        image_set_patch_size = patch_size

                    if is_target:
                        for image in dataset.images:
                            patches = extraction_func( #ep.extract_patch_records_with_exg(
                                            image, 
                                            annotations[image.image_name], 
                                            num_patches_per_target_image, 
                                            image_set_patch_size)
                            print("adding {} patches to target dataset".format(len(patches)))
                            data["target_patches"].extend(patches)
                    else:
                        for image in dataset.completed_images:
                        
                            # patches = ep.extract_patch_records_from_image_tiled(
                            #     image, 
                            #     image_set_patch_size, 
                            #     annotations[image.image_name], 
                            #     patch_overlap_percent=50, 
                            #     starting_patch_num=0)
                            patches = extraction_func(
                                            image, 
                                            annotations[image.image_name], 
                                            num_patches_per_source_image, 
                                            image_set_patch_size)
                            print("adding {} patches to source dataset".format(len(patches)))
                            data["source_patches"].extend(patches)
                            
                        
                except RuntimeError:
                    if is_target:
                        raise RuntimeError("Need annotations in target image set to determine image set dependent patch size.")
                    else:
                        pass # patch size cannot be determined due to 0 annotations

    data["source_patches"] = np.array(data["source_patches"])
    data["target_patches"] = np.array(data["target_patches"])

    return data



def patches_surrounding_boxes(config):
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]
    
    source_construction_params = config.training["source_construction_params"]
    method_params = source_construction_params["method_params"]
    patch_size = method_params["patch_size"]

    image_set_root = os.path.join("usr", "data", "image_sets")

    data = {
        "source_patches": [],
        "target_patches": []
    }

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if ((farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date):
                    is_target = True
                    patches = data["target_patches"]
                else:
                    is_target = False
                    patches = data["source_patches"]

                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })

                annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                             "annotations", "annotations_w3c.json")
                annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

                try:
                    if patch_size == "image_set_dependent":
                        image_set_patch_size = w3c_io.get_patch_size(annotations)
                    else:
                        image_set_patch_size = patch_size


                    
                    for image in dataset.completed_images:
                        patches.extend(ep.extract_patch_records_surrounding_gt_boxes(
                            image, 
                            annotations[image.image_name], 
                            image_set_patch_size))


                        # p = ep.extract_patch_records_surrounding_gt_boxes(
                        #     image, 
                        #     annotations[image.image_name], 
                        #     image_set_patch_size)

                        # if is_target:
                        #     data["target_patches"].extend(p)
                        # data["source_patches"].extend(p)


                except RuntimeError:
                    pass # patch size cannot be determined due to 0 annotations

    data["source_patches"] = np.array(data["source_patches"])
    data["target_patches"] = np.array(data["target_patches"])

    return data


def extract_patch_features(patches, extraction_type, config):
    logger = logging.getLogger(__name__)

    logger.info("Extracting patch features...")
    all_features = []

    manual = False #False
    if manual:
        #md.manual_desc(data, extraction_type, config)
        raise RuntimeError("unimplemented")
    else:
        model = graph_model.get_model("YOLOv4TinyBackbone", config)
        #model = graph_model.get_model("resnet_50", config)

        batch_size = 16 # 256 #16 #256 #1024
        #for dataset_loc in ["source", "target"]:

            #patches = data[dataset_loc + "_patches"]
        #features_lst = data[dataset_loc + "_features"]
        
        num_patches = patches.size #[0]

        if extraction_type == "box_patches":
            input_image_shape = np.array([150, 150, 3])
        else:
            input_image_shape = config.arch["input_image_shape"]


        for i in tqdm.trange(0, num_patches, batch_size):
            batch_patches = []
            for j in range(i, min(num_patches, i+batch_size)):
                patch = tf.convert_to_tensor(patches[j]["patch"], dtype=tf.float32)
                patch = tf.image.resize(images=patch, size=input_image_shape[:2])
                batch_patches.append(patch)
            batch_patches = tf.stack(values=batch_patches, axis=0)
            
            features = model.predict(batch_patches)
            for f in features:
                f = f.flatten()
                if i == 0:
                    print("f.shape: {}".format(f.shape))
                all_features.append(f)

    logger.info("Finished extracting patch features.")

    #data["source_features"] = np.array(data["source_features"])
    #data["target_features"] = np.array(data["target_features"])
    return np.array(all_features)


def extract_graph_match_data(config):

    logger = logging.getLogger(__name__)
    source_construction_params = config.training["source_construction_params"]
    method_params = source_construction_params["method_params"]
    extraction_type = method_params["extraction_type"]


    logger.info("Extracting patches...")

    if extraction_type == "surrounding_boxes":
        res = patches_surrounding_boxes(config)
    elif extraction_type == "excess_green":
        res = exg_patches(config, combo=False)
    elif extraction_type == "excess_green_box_combo":
        res = exg_patches(config, combo=True)
    elif extraction_type == "excess_green_two_phase":
        res = two_phase_exg_patches(config, combo=True)
    elif extraction_type == "excess_green_box_combo_two_phase":
        res = two_phase_exg_patches(config, combo=True)
    #elif extraction_type == "box_patches":
    #    res = box_patches(config)
    else:
        raise RuntimeError("Unrecognized extraction type: {}".format(extraction_type))

    logger.info("Finished extracting patches.")

    data = {
        "source_patches": res["source_patches"],
        "target_patches": res["target_patches"],
        "source_features": [],
        "target_features": []
    }

    print("source_patches.shape", data["source_patches"].shape)
    print("target_patches.shape", data["target_patches"].shape)

    data["source_features"] = extract_patch_features(data["source_patches"], extraction_type, config)
    data["target_features"] = extract_patch_features(data["target_patches"], extraction_type, config)


    print("source_features.shape", data["source_features"].shape)
    print("target_features.shape", data["target_features"].shape)

    print("source_patches == target_patches ??: {}".format(np.array_equal(data["source_patches"], data["target_patches"])))
    print("source_features == target_features ??: {}".format(np.array_equal(data["source_features"], data["target_features"])))

    return data


def build_graph_subset(config):
        
    logger = logging.getLogger(__name__)
    
    source_construction_params = config.training["source_construction_params"]
    desired_source_size = source_construction_params["size"]
    method_params = source_construction_params["method_params"]
    extraction_type = method_params["extraction_type"]
    match_method = method_params["match_method"]

    res = extract_graph_match_data(config)

    source_features = res["source_features"]
    target_features = res["target_features"]
    source_patches = res["source_patches"]
    target_patches = res["target_patches"]

    logger.info("Performing matching...")
    if match_method == "bipartite_b_matching":
        selected_source_patches = graph_match.bipartite_b_match(config, desired_source_size, source_features, target_features, source_patches, target_patches)
    elif match_method == "diverse_bipartite_b_matching":
        selected_source_patches = graph_match.diverse_bipartite_b_match(config, desired_source_size, source_features, target_features, source_patches, target_patches)
    elif match_method == "reciprocal_match":
        selected_source_patches = graph_match.reciprocal_match(config, desired_source_size, source_features, target_features, source_patches, target_patches)
    else:
        raise RuntimeError("Unrecognized match method: {}".format(match_method))
    logger.info("Finished performing matching.")

    usr_data_root = os.path.join("usr", "data")
    patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
    training_patches_dir = os.path.join(patches_dir, "training")
    validation_patches_dir = os.path.join(patches_dir, "validation")
    os.makedirs(training_patches_dir)
    os.makedirs(validation_patches_dir)


    training_size = round(selected_source_patches.size * 0.8)
    training_subset = random.sample(np.arange(selected_source_patches.size).tolist(), training_size)

    training_patches = selected_source_patches[training_subset]
    validation_patches = np.delete(selected_source_patches, training_subset)

    logger.info("Writing patches...")
    if extraction_type == "box_patches":
        ep.write_patches_from_gt_box_records(training_patches, training_patches_dir)
        ep.write_patches_from_gt_box_records(validation_patches, validation_patches_dir)
    else:
        ep.write_annotated_patch_records(training_patches, training_patches_dir)
        ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")



def get_source_annotations(config): 
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    image_set_root = os.path.join("usr", "data", "image_sets")

    annotation_records = []
    total_annotation_count = 0
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if not ((farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date):
                    annotations_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})


                    print("adding {}/{}/{}".format(farm_name, field_name, mission_date))
                    annotation_record = {
                        "farm_name": farm_name,
                        "field_name": field_name,
                        "mission_date": mission_date,
                        "num_annotations": w3c_io.get_num_annotations(annotations, require_completed=True),
                        #"patch_size": get_patch_size(annotations),
                        "annotations": annotations
                    }

                    total_annotation_count += annotation_record["num_annotations"]
                    annotation_records.append(annotation_record)

    return annotation_records, total_annotation_count

def build_direct(config):
        
    logger = logging.getLogger(__name__)
    #source_size = config.training["source_construction_params"]["size"]
    
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]
    
    source_construction_params = config.training["source_construction_params"]
    method_params = source_construction_params["method_params"]
    source_size = source_construction_params["size"]
    extraction_type = method_params["extraction_type"]
    patch_size = method_params["patch_size"]

    image_set_root = os.path.join("usr", "data", "image_sets")

    dataset = DataSet({
        "farm_name": target_farm_name,
        "field_name": target_field_name,
        "mission_date": target_mission_date
    })

    annotations_path = os.path.join(image_set_root, target_farm_name, target_field_name, target_mission_date,
                                    "annotations", "annotations_w3c.json")
    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    allow_empty = extraction_type != "surrounding_boxes"

    completed_images = w3c_io.get_completed_images(annotations, allow_empty=allow_empty)
    num_annotated_images = len(completed_images)
    num_annotations = w3c_io.get_num_annotations(annotations, require_completed=True)
    if num_annotated_images == 0 or num_annotations == 0:
        raise RuntimeError("Insufficient number of annotations for direct training")


    num_patches_per_image = m.ceil(source_size / num_annotated_images)

    patches = []
    if patch_size == "image_set_dependent":
        image_set_patch_size = w3c_io.get_patch_size(annotations)
    else:
        image_set_patch_size = patch_size

    if extraction_type == "surrounding_boxes":
        images = dataset.nonempty_completed_images
    else:
        images = dataset.completed_images

    for image in images:
        if extraction_type == "surrounding_boxes":
            patches.extend(ep.extract_patch_records_surrounding_gt_boxes(
                image, 
                annotations[image.image_name], 
                num_patches_per_image,
                image_set_patch_size))
        elif extraction_type == "excess_green":
            patches.extend(ep.extract_patch_records_with_exg(
                image, 
                annotations[image.image_name], 
                num_patches_per_image, 
                image_set_patch_size))
        elif extraction_type == "excess_green_box_combo":
            patches.extend(ep.extract_patch_records_with_exg_box_combo(
                image, 
                annotations[image.image_name], 
                num_patches_per_image, 
                image_set_patch_size))
        else:
            raise RuntimeError("Unrecognized extraction type: {}".format(extraction_type))

    patches = np.array(patches)
    np.random.shuffle(patches)
    patches = patches[:source_size]


    usr_data_root = os.path.join("usr", "data")
    patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
    training_patches_dir = os.path.join(patches_dir, "training")
    validation_patches_dir = os.path.join(patches_dir, "validation")
    os.makedirs(training_patches_dir)
    os.makedirs(validation_patches_dir)


    training_size = round(patches.size * 0.8)
    training_subset = random.sample(np.arange(patches.size).tolist(), training_size)

    training_patches = patches[training_subset]
    validation_patches = np.delete(patches, training_subset)

    logger.info("Writing patches...")
    ep.write_annotated_patch_records(training_patches, training_patches_dir)
    ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")



def build_even_subset(config):

    logger = logging.getLogger(__name__)

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    source_construction_params = config.training["source_construction_params"]
    method_params = source_construction_params["method_params"]
    extraction_type = method_params["extraction_type"]
    patch_size = method_params["patch_size"]
    exclude_target_from_source = method_params["exclude_target_from_source"]
    source_size = source_construction_params["size"]

    image_set_root = os.path.join("usr", "data", "image_sets")
    datasets = []

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })

                is_target = farm_name == target_farm_name and \
                            field_name == target_field_name and \
                            mission_date == target_mission_date
                exclude = is_target and exclude_target_from_source

                if (not exclude) and (len(dataset.completed_images) > 0):
                    datasets.append(dataset)

    extraction_func = ep.extract_patch_records_with_exg_box_combo
    #num_per_dataset = m.ceil(source_size / len(datasets))

    patches = []
    # for dataset in datasets:

    #     num_patches_per_image = m.ceil(num_per_dataset / len(dataset.completed_images))
    #     logger.info("Processing {} | Num per dataset: {} | Num per image: {}".format(
    #         dataset.image_set_name, num_per_dataset, num_patches_per_image))
        
    #     annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
    #     if patch_size == "image_set_dependent":
    #         image_set_patch_size = w3c_io.get_patch_size(annotations)
    #     else:
    #         image_set_patch_size = patch_size
        
    #     for image in dataset.completed_images:
    #         patches.extend(extraction_func( #ep.extract_patch_records_with_exg(
    #                         image, 
    #                         annotations[image.image_name], 
    #                         num_patches_per_image, 
    #                         image_set_patch_size))

    num_completed_images = np.sum([len(dataset.completed_images) for dataset in datasets])
    num_patches_per_image = m.ceil(source_size / num_completed_images)
    for dataset in datasets:
        logger.info("Processing {} | Num per image: {}".format(
            dataset.image_set_name, num_patches_per_image))
        
        annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
        if patch_size == "image_set_dependent":
            image_set_patch_size = w3c_io.get_patch_size(annotations)
        else:
            image_set_patch_size = patch_size
        
        for image in dataset.completed_images:
            patches.extend(extraction_func( #ep.extract_patch_records_with_exg(
                            image, 
                            annotations[image.image_name], 
                            num_patches_per_image, 
                            image_set_patch_size))
    patches = np.array(patches)
    np.random.shuffle(patches)
    patches = patches[:source_size]

    usr_data_root = os.path.join("usr", "data")
    patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
    training_patches_dir = os.path.join(patches_dir, "training")
    validation_patches_dir = os.path.join(patches_dir, "validation")
    os.makedirs(training_patches_dir)
    os.makedirs(validation_patches_dir)        
            

    training_size = round(patches.size * 0.8)
    training_subset = random.sample(np.arange(patches.size).tolist(), training_size)

    training_patches = patches[training_subset]
    validation_patches = np.delete(patches, training_subset)

    logger.info("Writing patches...")
    ep.write_annotated_patch_records(training_patches, training_patches_dir)
    ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")


# def build_even_subset(config):

#     source_size = config.training["source_construction_params"]["size"]

#     def create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes):
#         records = []
#         image_path = glob.glob(os.path.join("usr", "data", "image_sets", 
#                                     farm_name, field_name, mission_date, 
#                                     "images", image_name + ".*"))[0]
#         image_name = os.path.basename(image_path)[:-4]
#         for gt_box in gt_boxes:
#             record = {
#                 "image_path": image_path,
#                 "image_name": image_name,
#                 "patch_coords": gt_box
#             }
#             records.append(record)
#         return records


#     annotation_records, total_annotation_count = get_source_annotations(config)

#     if total_annotation_count < source_size:
#         raise RuntimeError("Insufficient number of source annotations available. Requested: {}. Found: {}.".format(
#             source_size, total_annotation_count))

#     annotation_records.sort(key=lambda x : x["num_annotations"])
#     annotations_needed = source_size
#     records_remaining = len(annotation_records)
    
#     gt_box_records = []
#     for annotation_record in annotation_records:
#         farm_name = annotation_record["farm_name"]
#         field_name = annotation_record["field_name"]
#         mission_date = annotation_record["mission_date"]
#         record_annotations_needed = m.ceil(annotations_needed / records_remaining)
#         print("Need {} annotations from {}/{}/{}.".format(record_annotations_needed, farm_name, field_name, mission_date))
#         if record_annotations_needed > annotation_record["num_annotations"]:
#             print("taking all annotations from {}/{}/{}".format(farm_name, field_name, mission_date))
#             for image_name in annotation_record["annotations"].keys():
#                 if annotation_record["annotations"][image_name]["status"] == "completed":
#                     gt_boxes = (annotation_record["annotations"][image_name]["boxes"]).tolist()
#                     gt_box_records.extend(create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes))

#             annotations_needed -= annotation_record["num_annotations"]
#         else:
#             print("taking subset of annotations from {}/{}/{}".format(farm_name, field_name, mission_date))
#             image_records = []
#             for image_name in annotation_record["annotations"].keys():
#                 if annotation_record["annotations"][image_name]["status"] == "completed":
#                     image_record = {"image_name": image_name,
#                                     "num_annotations": annotation_record["annotations"][image_name]["boxes"].shape[0]}
#                     image_records.append(image_record)
#             image_records.sort(key=lambda x: x["num_annotations"])

#             images_remaining = len(image_records)
#             for image_record in image_records:
#                 image_name = image_record["image_name"]
#                 image_annotations_needed = m.ceil(record_annotations_needed / images_remaining)
#                 print("     Need {} annotations from {}.".format(image_annotations_needed, image_name))
#                 if image_annotations_needed > image_record["num_annotations"]:
#                     gt_boxes = (annotation_record["annotations"][image_name]["boxes"]).tolist()
#                     gt_box_records.extend(create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes))
#                     #source_boxes.extend((annotation_record["annotations"][image_name]["boxes"]).tolist())
#                     record_annotations_needed -= image_record["num_annotations"]
#                     annotations_needed -= image_record["num_annotations"]
#                     print("    took all annotations from image {}: {}".format(image_name, image_record["num_annotations"]))

#                 else:
#                     indices = random.sample(np.arange(annotation_record["annotations"][image_name]["boxes"].shape[0]).tolist(), image_annotations_needed)
#                     gt_boxes = (annotation_record["annotations"][image_name]["boxes"][indices]).tolist()
#                     gt_box_records.extend(create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes))
#                     #source_boxes.extend((annotation_record["annotations"][image_name]["boxes"][indices]).tolist())
#                     record_annotations_needed -= image_annotations_needed
#                     annotations_needed -= image_annotations_needed
#                     print("    took subset of annotations from image {}: {}".format(image_name, image_annotations_needed))
#                 images_remaining -= 1
#         records_remaining -= 1

#     print("assembled source dataset")
#     print("length of dataset is ", len(gt_box_records))

#     gt_box_records = np.array(gt_box_records)

#     #extract_patches_from_gt_box_records(gt_box_records, patch_dir)
#     usr_data_root = os.path.join("usr", "data")
#     patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
#     training_patches_dir = os.path.join(patches_dir, "training")
#     validation_patches_dir = os.path.join(patches_dir, "validation")
#     os.makedirs(training_patches_dir)
#     os.makedirs(validation_patches_dir)

#     print("Extracting even subset source patches")

#     training_size = round(gt_box_records.size * 0.8)
#     training_subset = random.sample(np.arange(gt_box_records.size).tolist(), training_size)


#     training_records = gt_box_records[training_subset]
#     validation_records = np.delete(gt_box_records, training_subset)

#     ep.write_patches_from_gt_box_records(training_records, training_patches_dir)
#     ep.write_patches_from_gt_box_records(validation_records, validation_patches_dir)
