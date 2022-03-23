import logging
import os
import glob
import tqdm

import random
import math as m
import numpy as np
import tensorflow as tf

import extract_patches as ep
from graph import graph_match, graph_model
from image_set import DataSet

from io_utils import w3c_io, tf_record_io


def build_inference_datasets(config):

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    model_dir = os.path.join("usr", "data", "models", config.arch["model_uuid"])

    annotations_path = os.path.join("usr", "data", "image_sets", 
                                    target_farm_name, target_field_name, target_mission_date,
                                    "annotations", "annotations_w3c.json")

    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    patch_dir = os.path.join(model_dir, "target_patches")
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

    for image in tqdm.tqdm(dataset.images, desc="Extracting target patches"):

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
    if method_name == "even_subset":
        build_even_subset(config)
    elif method_name == "graph_subset":
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




def exg_patches(config, combo):
    if combo:
        extraction_func = ep.extract_patch_records_with_exg_box_combo
    else:
        extraction_func = ep.extract_patch_records_with_exg

    logger = logging.getLogger(__name__)

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

                    # if is_target:
                    #     for image in dataset.images:
                    #         patches = ep.extract_patch_records_with_exg(
                    #                         image, 
                    #                         annotations[image.image_name], 
                    #                         num_patches_per_target_image, 
                    #                         image_set_patch_size)
                    #         print("adding {} patches to target dataset".format(len(patches)))
                    #         data["target_patches"].extend(patches)
                    # else:
                    if is_target:
                        for image in dataset.completed_images:
                        
                            patches = extraction_func(
                                            image, 
                                            annotations[image.image_name], 
                                            num_patches_per_source_image, 
                                            image_set_patch_size)
                            print("adding {} patches to source dataset".format(len(patches)))
                            data["source_patches"].extend(patches)
                        
                            data["target_patches"].extend(patches)
                            
                        
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
                    #patches = data["target_patches"]
                else:
                    is_target = False
                    #patches = data["source_patches"]

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
                        for image in dataset.completed_images:
                            # patches.extend(ep.extract_patch_records_surrounding_gt_boxes(
                            #     image, 
                            #     annotations[image.image_name], 
                            #     image_set_patch_size))

                            patches = ep.extract_patch_records_surrounding_gt_boxes(
                                image, 
                                annotations[image.image_name], 
                                image_set_patch_size)

                            data["source_patches"].extend(patches)
                            if is_target:
                                data["target_patches"].extend(patches)


                except RuntimeError:
                    pass # patch size cannot be determined due to 0 annotations

    data["source_patches"] = np.array(data["source_patches"])
    data["target_patches"] = np.array(data["target_patches"])

    return data



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

    logger.info("Extracting patch features...")
    
    model = graph_model.get_model()

    batch_size = 256 #1024
    for dataset_loc in ["source", "target"]:

        patches = data[dataset_loc + "_patches"]
        features_lst = data[dataset_loc + "_features"]
        
        num_patches = patches.shape[0]

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
                features_lst.append(f)

    logger.info("Finished extracting patch features.")

    data["source_features"] = np.array(data["source_features"])
    data["target_features"] = np.array(data["target_features"])

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

    num_annotated_images = len(w3c_io.get_completed_images(annotations))
    num_annotations = w3c_io.get_num_annotations(annotations, require_completed=True)
    if num_annotated_images == 0 or num_annotations == 0:
        raise RuntimeError("Insufficient number of annotations for direct training")


    if "size" in source_construction_params:
        size = source_construction_params["size"]
        num_patches_per_image = m.ceil(size / num_annotated_images)


    patches = []
    if patch_size == "image_set_dependent":
        image_set_patch_size = w3c_io.get_patch_size(annotations)
    else:
        image_set_patch_size = patch_size

    for image in dataset.completed_images:
        if extraction_type == "surrounding_boxes":
            patches.extend(ep.extract_patch_records_surrounding_gt_boxes(
                image, 
                annotations[image.image_name], 
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

    source_size = config.training["source_construction_params"]["size"]

    def create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes):
        records = []
        image_path = glob.glob(os.path.join("usr", "data", "image_sets", 
                                    farm_name, field_name, mission_date, 
                                    "images", image_name + ".*"))[0]
        image_name = os.path.basename(image_path)[:-4]
        for gt_box in gt_boxes:
            record = {
                "image_path": image_path,
                "image_name": image_name,
                "patch_coords": gt_box
            }
            records.append(record)
        return records


    annotation_records, total_annotation_count = get_source_annotations(config)

    if total_annotation_count < source_size:
        raise RuntimeError("Insufficient number of source annotations available. Requested: {}. Found: {}.".format(
            source_size, total_annotation_count))

    annotation_records.sort(key=lambda x : x["num_annotations"])
    annotations_needed = source_size
    records_remaining = len(annotation_records)
    
    gt_box_records = []
    for annotation_record in annotation_records:
        farm_name = annotation_record["farm_name"]
        field_name = annotation_record["field_name"]
        mission_date = annotation_record["mission_date"]
        record_annotations_needed = m.ceil(annotations_needed / records_remaining)
        print("Need {} annotations from {}/{}/{}.".format(record_annotations_needed, farm_name, field_name, mission_date))
        if record_annotations_needed > annotation_record["num_annotations"]:
            print("taking all annotations from {}/{}/{}".format(farm_name, field_name, mission_date))
            for image_name in annotation_record["annotations"].keys():
                if annotation_record["annotations"][image_name]["status"] == "completed":
                    gt_boxes = (annotation_record["annotations"][image_name]["boxes"]).tolist()
                    gt_box_records.extend(create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes))

            annotations_needed -= annotation_record["num_annotations"]
        else:
            print("taking subset of annotations from {}/{}/{}".format(farm_name, field_name, mission_date))
            image_records = []
            for image_name in annotation_record["annotations"].keys():
                if annotation_record["annotations"][image_name]["status"] == "completed":
                    image_record = {"image_name": image_name,
                                    "num_annotations": annotation_record["annotations"][image_name]["boxes"].shape[0]}
                    image_records.append(image_record)
            image_records.sort(key=lambda x: x["num_annotations"])

            images_remaining = len(image_records)
            for image_record in image_records:
                image_name = image_record["image_name"]
                image_annotations_needed = m.ceil(record_annotations_needed / images_remaining)
                print("     Need {} annotations from {}.".format(image_annotations_needed, image_name))
                if image_annotations_needed > image_record["num_annotations"]:
                    gt_boxes = (annotation_record["annotations"][image_name]["boxes"]).tolist()
                    gt_box_records.extend(create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes))
                    #source_boxes.extend((annotation_record["annotations"][image_name]["boxes"]).tolist())
                    record_annotations_needed -= image_record["num_annotations"]
                    annotations_needed -= image_record["num_annotations"]
                    print("    took all annotations from image {}: {}".format(image_name, image_record["num_annotations"]))

                else:
                    indices = random.sample(np.arange(annotation_record["annotations"][image_name]["boxes"].shape[0]).tolist(), image_annotations_needed)
                    gt_boxes = (annotation_record["annotations"][image_name]["boxes"][indices]).tolist()
                    gt_box_records.extend(create_gt_box_records(farm_name, field_name, mission_date, image_name, gt_boxes))
                    #source_boxes.extend((annotation_record["annotations"][image_name]["boxes"][indices]).tolist())
                    record_annotations_needed -= image_annotations_needed
                    annotations_needed -= image_annotations_needed
                    print("    took subset of annotations from image {}: {}".format(image_name, image_annotations_needed))
                images_remaining -= 1
        records_remaining -= 1

    print("assembled source dataset")
    print("length of dataset is ", len(gt_box_records))

    gt_box_records = np.array(gt_box_records)

    #extract_patches_from_gt_box_records(gt_box_records, patch_dir)
    usr_data_root = os.path.join("usr", "data")
    patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
    training_patches_dir = os.path.join(patches_dir, "training")
    validation_patches_dir = os.path.join(patches_dir, "validation")
    os.makedirs(training_patches_dir)
    os.makedirs(validation_patches_dir)

    print("Extracting even subset source patches")

    training_size = round(gt_box_records.size * 0.8)
    training_subset = random.sample(np.arange(gt_box_records.size).tolist(), training_size)


    training_records = gt_box_records[training_subset]
    validation_records = np.delete(gt_box_records, training_subset)

    ep.write_patches_from_gt_box_records(training_records, training_patches_dir)
    ep.write_patches_from_gt_box_records(validation_records, validation_patches_dir)
