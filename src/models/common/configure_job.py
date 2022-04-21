import os
import glob
import uuid

import math as m
import numpy as np

import image_set
from io_utils import w3c_io





def generate_model_names(num):
    return ["model_" + str(i+1) for i in range(num)]
    #return [randomname.get_name() for _ in range(num)]


def generate_model_uuids(num):
    return [str(uuid.uuid4()) for _ in range(num)]




def add_models_to_job_config(job_config):

    if "variation_config" in job_config:
        num_models = len(job_config["variation_config"]["param_values"]) * job_config["replications"]
    else:
        num_models = job_config["replications"]

    model_uuids = generate_model_uuids(num_models)
    model_names = generate_model_names(num_models)

    job_config["model_info"] = []

    for (model_uuid, model_name) in zip(model_uuids, model_names):
        job_config["model_info"].append({
            "model_uuid": model_uuid,
            "model_name": model_name,
            "stage": "Enqueued",
        })





# def fill_job_config_full_source(job_config, target_farm_name, target_field_name, target_mission_date):
    
#     job_config.update({
#         "replications": 1,

#         "arch_config": {

#             "model_type": "yolov4_tiny",
#             "backbone_config": {
#                 "backbone_type": "csp_darknet53_tiny"
#             },
#             "neck_config": {
#                 "neck_type": "yolov4_tiny_deconv"
#             },
#             "max_detections": 50,
#             "input_image_shape": [416, 416, 3],
#             "class_map": {"plant": 0}
#         },


#         "training_config": {
#             "training_sequence": [

#                 {
#                     "training_datasets": [
#                     ],
#                     "validation_datasets": [
#                     ],


#                     "data_loader": {
#                         "type": "default"
#                         #"percent_of_batch_with_objects": 50
#                     },

#                     "learning_rate_schedule": {
#                         "schedule_type": "constant",
#                         "learning_rate": 0.0001
#                     },


#                     "data_augmentations": [
#                         {
#                             "type": "flip_vertical", 
#                             "parameters": {
#                                 "probability": 0.5
#                             }
#                         },
#                         {
#                             "type": "flip_horizontal", 
#                             "parameters": {
#                                 "probability": 0.5
#                             }
#                         },
#                         {
#                             "type": "rotate_90", 
#                             "parameters": {
#                                 "probability": 0.5
#                             }
#                         }                   
#                     ],

#                     "min_num_epochs": 15,
#                     "max_num_epochs": 300,
#                     "early_stopping": {
#                         "apply": True,
#                         "monitor": "validation_loss",
#                         "num_epochs_tolerance": 15
#                     },
#                     "batch_size": 16,

#                     "save_method": "best_validation_loss",
#                     "percent_of_training_set_used": 100,
#                     "percent_of_validation_set_used": 100                    

#                 }
#             ]
#         },

#         "inference_config": {

#             "image_sets": [],

#             "patch_border_buffer_percent": 0.10,
#             "batch_size": 16,
#             "patch_nms_iou_thresh": 0.4,
#             "image_nms_iou_thresh": 0.4,
#             "score_thresh": 0.5
#         }
#     })


#     usr_data_root = os.path.join("usr", "data")
#     image_set_root = os.path.join(usr_data_root, "image_sets")

#     for farm_path in glob.glob(os.path.join(image_set_root, "*")):
#         print(farm_path)
#         for field_path in glob.glob(os.path.join(farm_path, "*")):
#             print("  ", field_path)
#             for mission_path in glob.glob(os.path.join(field_path, "*")):

#                 annotation_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
#                 annotations = w3c_io.load_annotations(annotation_path, {"plant": 0})

#                 completed_images = image_set.get_completed_images(annotations)
#                 num_annotations = image_set.get_num_annotations(annotations)

#                 if num_annotations > 32 and len(completed_images) > 0:
    
#                     farm_name = os.path.basename(farm_path)
#                     field_name = os.path.basename(field_path)
#                     mission_date = os.path.basename(mission_path)

#                     patch_hw = get_patch_hw(farm_name, field_name, mission_date)
#                     job_config["training_config"]["training_sequence"][0]["training_datasets"].append({
#                         "farm_name": farm_name,
#                         "field_name": field_name,
#                         "mission_date": mission_date,
#                         "image_names": completed_images,
#                         "patch_extraction_params": {
#                             "method": "tile",
#                             "patch_size": patch_hw,
#                             "patch_overlap_percent": 50
#                         }                    
#                     })
#                     job_config["inference_config"]["image_sets"].append({
#                         "farm_name": farm_name,
#                         "field_name": field_name,
#                         "mission_date": mission_date,
#                         "training_image_names": completed_images,
#                         "validation_image_names": [],
#                         "test_image_names": [],
#                         "patch_extraction_params": {
#                             "method": "tile",
#                             "patch_size": patch_hw,
#                             "patch_overlap_percent": 50
#                         }
#                     })

#     annotations_path = os.path.join(image_set_root, target_farm_name, target_field_name, target_mission_date,
#                                      "annotations", "annotations_w3c.json")
#     annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

#     completed_images = image_set.get_completed_images(annotations)
    
#     patch_hw = get_patch_hw(target_farm_name, target_field_name, target_mission_date)
#     job_config["training_config"]["training_sequence"][0]["validation_datasets"].append({
#         "farm_name": target_farm_name,
#         "field_name": target_field_name,
#         "mission_date": target_mission_date,
#         "image_names": completed_images,
#         "patch_extraction_params": {
#             "method": "tile",
#             "patch_size": patch_hw,
#             "patch_overlap_percent": 50
#         }                 
#     })

#     job_config["inference_config"]["image_sets"].append({
#         "farm_name": target_farm_name,
#         "field_name": target_field_name,
#         "mission_date": target_mission_date,
#         "training_image_names": [],
#         "validation_image_names": completed_images,
#         "test_image_names": [],
#         "patch_extraction_params": {
#             "method": "tile",
#             "patch_size": patch_hw,
#             "patch_overlap_percent": 50
#         }
#     })




def fill_job_config(job_config):

    #training_images, validation_images = partition_datasets(farm_name, field_name, mission_date)
    #patch_hw = job_config.get_patch_hw(farm_name, field_name, mission_date)

    
    job_config.update({
        #"replications": 1,

        "arch_config": {

            "model_type": "yolov4_tiny",
            "backbone_config": {
                "backbone_type": "csp_darknet53_tiny"
            },
            "neck_config": {
                "neck_type": "yolov4_tiny_deconv"
            },
            "max_detections": 50,
            "input_image_shape": [416, 416, 3],
            "class_map": {"plant": 0}
        },


        "training_config": {
            "training_sequence": [

                {

                    "data_loader": {
                        "type": "default"
                    },

                    "learning_rate_schedule": {
                        "schedule_type": "constant",
                        "learning_rate": 0.0001
                    },


                    "data_augmentations": [
                        {
                            "type": "flip_vertical", 
                            "parameters": {
                                "probability": 0.5
                            }
                        },
                        {
                            "type": "flip_horizontal", 
                            "parameters": {
                                "probability": 0.5
                            }
                        },
                        {
                            "type": "rotate_90", 
                            "parameters": {
                                "probability": 0.5
                            }
                        }                   
                    ],

                    "min_num_epochs": 15,
                    "max_num_epochs": 3000,
                    "early_stopping": {
                        "apply": True,
                        "monitor": "validation_loss",
                        "num_epochs_tolerance": job_config["tol_test"] #12
                    },
                    "batch_size": 16,

                    "save_method": "best_validation_loss",
                    "percent_of_training_set_used": 100,
                    "percent_of_validation_set_used": 100                    

                }
            ]
        },


        "inference_config": {

            # "datasets": [
            #     {
            #         "farm_name": job_config["target_farm_name"],
            #         "field_name": job_config["target_field_name"],
            #         "mission_date": job_config["target_mission_date"],
            #         "training_image_names": [],
            #         "validation_image_names": [],
            #         "test_image_names": [],
            #         #"patch_extraction_params": None
            #     }
            # ],
            "patch_border_buffer_percent": 0.10,
            "batch_size": 16,
            "patch_nms_iou_thresh": 0.4,
            "image_nms_iou_thresh": 0.4,
            "score_thresh": 0.5
        }
    })

    add_models_to_job_config(job_config)






# def fill_job_config(job_config, farm_name, field_name, mission_date, config_type):

#     # if config_type == "direct":
#     #     raise RuntimeError("direct not implemented")
    
#     # elif config_type == "full_source":
#     #     fill_job_config_full_source(job_config, farm_name, field_name, mission_date)

#     # elif config_type == "random_subset":
#     #     raise RuntimeError("random_subset not implemented")
    
#     # elif config_type == "graph_subset":
#     #     raise RuntimeError("graph_subset not implemented")


#     # else:
#     #     raise RuntimeError("Unsupported training set construction type.")

#     add_models_to_job_config(job_config)