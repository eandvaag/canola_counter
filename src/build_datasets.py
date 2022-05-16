import logging
import os
import glob
import tqdm

import random
import numpy as np

import extract_patches as ep
import extract_features as ef
from image_set import DataSet, Image
from matching import match


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

    source_construction_params = config.training["source_construction_params"]
    method_name = source_construction_params["method_name"]

    if method_name == "direct_2":
        build_direct_2(config)
    elif method_name == "direct_tiled":
        build_direct_tiled(config)
    elif method_name == "transfer":
        build_transfer(config)
    else:
        raise RuntimeError("Unrecognized source dataset construction method: {}".format(method_name))




def determine_number_of_patches_for_each_image(image_names, annotations, total_plants_needed, total_other_needed, allow_box_reuse=True):

    if len(image_names) == 0:
        raise RuntimeError("Number of images is zero")

    #completed_images = w3c_io.get_completed_images(annotations)
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
    
    left_arr = np.array([annotations[image_name]["boxes"].shape[0] for image_name in image_names]) #completed_images])
    assigned_arr = np.array([0] * len(image_names)) #len(completed_images))
    print("left_arr", left_arr)
    while total_plants_needed > 0:
        print("left_arr", left_arr)
        mask = left_arr > 0
        

        if not mask.any():
            if allow_box_reuse:
                left_arr = np.array([annotations[image_name]["boxes"].shape[0] for image_name in image_names]) #completed_images])
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

    for i, image_name in enumerate(image_names):
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



def build_transfer(config):

    logger = logging.getLogger(__name__)

    logger.info("Building feature vectors...")
    ef.build_feature_vectors(config)
    ef.build_target_feature_vectors(config, completed_only=True)
    logger.info("Finished building feature vectors.")

    logger.info("Performing matching...")
    #patches = match.match(config)
    #match.retrieval(config)
    patches = match.get_nn(config)
    logger.info("Finished performing matching.")

    #exit()

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


    logger.info("Extracted {} training patches and {} validation patches".format(training_patches.size, validation_patches.size))
    logger.info("Writing patches...")
    ep.write_annotated_patch_records(training_patches, training_patches_dir)
    ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")



def build_direct_tiled(config):

    logger = logging.getLogger(__name__)

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    source_construction_params = config.training["source_construction_params"]
    #desired_training_set_size = source_construction_params["size"]

    #prop_plant_patches = 0.80
    #desired_plant_size = round(desired_training_set_size * prop_plant_patches)
    #desired_other_size = desired_training_set_size - desired_plant_size

    patch_size = "image_set_dependent"

    patches = []


    annotations_path = os.path.join("usr", "data", "image_sets", 
                               target_farm_name, target_field_name, target_mission_date, 
                               "annotations", "annotations_w3c.json")

    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    image_names = config.arch["training_validation_images"]

    images_root = os.path.join("usr", "data", "image_sets", 
                               target_farm_name, target_field_name, target_mission_date, "images")
    try:
        if patch_size == "image_set_dependent":
            image_set_patch_size = w3c_io.get_patch_size(annotations)
        else:
            image_set_patch_size = patch_size

        for image_name in image_names:

            image_path = glob.glob(os.path.join(images_root, image_name + ".*"))[0]
            image = Image(image_path)


            patches.extend(ep.extract_patch_records_from_image_tiled(
                image, 
                image_set_patch_size, 
                annotations[image_name])
            )

        logger.info("extraction complete.")

        

    except RuntimeError:
        raise RuntimeError("Need annotations in target image set to determine image set dependent patch size.")


    patches = np.array(patches)
    np.random.shuffle(patches)
    #patches = patches[:source_size]

    #assert patches.size == desired_training_set_size
    logger.info("Total number of training/validation patches is {}.".format(patches.size))

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


    logger.info("Extracted {} training patches and {} validation patches".format(training_patches.size, validation_patches.size))
    logger.info("Writing patches...")
    ep.write_annotated_patch_records(training_patches, training_patches_dir)
    ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")








def build_direct_2(config):

    logger = logging.getLogger(__name__)

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    source_construction_params = config.training["source_construction_params"]
    desired_training_set_size = source_construction_params["size"]

    prop_plant_patches = 0.80
    desired_plant_size = round(desired_training_set_size * prop_plant_patches)
    desired_other_size = desired_training_set_size - desired_plant_size

    patch_size = "image_set_dependent"

    target_plant_patches = []
    target_other_patches = []

    dataset = DataSet({
        "farm_name": target_farm_name,
        "field_name": target_field_name,
        "mission_date": target_mission_date
    })
    print("now processing", dataset.image_set_name)

    annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})

    image_names = config.arch["training_validation_images"]

    images_root = os.path.join("usr", "data", "image_sets", 
                               target_farm_name, target_field_name, target_mission_date, "images")
    try:
        if patch_size == "image_set_dependent":
            image_set_patch_size = w3c_io.get_patch_size(annotations)
        else:
            image_set_patch_size = patch_size



        assigned = determine_number_of_patches_for_each_image(image_names, annotations, 
                    desired_plant_size, desired_other_size, allow_box_reuse=True)


        logger.info("Assignment complete. Extracting patches...")


        for image_name in image_names:

            image_path = glob.glob(os.path.join(images_root, image_name + ".*"))[0]
            image = Image(image_path)



            if assigned[image.image_name]["plant"] > 0 or assigned[image.image_name]["other"] > 0:
                plant_patches, other_patches = ep.extract_plant_and_other(
                    image, 
                    annotations[image.image_name],
                    assigned[image.image_name]["plant"],
                    assigned[image.image_name]["other"],
                    image_set_patch_size
                )


                target_plant_patches.extend(plant_patches)
                target_other_patches.extend(other_patches)

        logger.info("extraction complete.")

        

    except RuntimeError:
        raise RuntimeError("Need annotations in target image set to determine image set dependent patch size.")


    patches = []
    patches.extend(target_plant_patches)
    patches.extend(target_other_patches)
    patches = np.array(patches)
    np.random.shuffle(patches)
    #patches = patches[:source_size]

    assert patches.size == desired_training_set_size
    logger.info("Total number of training/validation patches is {}.".format(patches.size))

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


    logger.info("Extracted {} training patches and {} validation patches".format(training_patches.size, validation_patches.size))
    logger.info("Writing patches...")
    ep.write_annotated_patch_records(training_patches, training_patches_dir)
    ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
    logger.info("Finished writing patches.")

