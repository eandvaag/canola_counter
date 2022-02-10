from io_utils import json_io

from models.common import group_interface


def get_completed_images(img_set_data):
    return [img for img in img_set_data["images"].keys() \
            if img_set_data["images"][img]["status"] == "completed"]


def partition_datasets(test_farm_name, test_field_name, test_mission_date, reserved_test_images):
    """
        determines which datasets are needed.
        for each dataset, determines which are for training and which are for validation
            

        need a general plan
        if num_images < X:
            get Y nearest datasets
        
        else:


    """
    #needed_datasets = [{
    #    "farm_name": "UNI",
    #    "field_name": "LowN1",
    #    "mission_date": "2021-06-07"}]


    img_set_root = os.path.join("usr", "data", test_farm_name, test_field_name, test_mission_date)
    img_set_data_path = os.path.join(img_set_root, "img_set_data.json")
    img_set_data = json_io.load_json(img_set_data_path)


    images = get_completed_images(img_set_data)
    available_images = [img for img in images if img not in reserved_test_images]

    annotation_path = os.path.join(img_set_root, "annotations", "annotations_w3c.json")
    boxes, classes = w3c_io.load_boxes_and_classes(annotation_path, class_map)

    #available_annotations = [len(boxes[img]) for img in available_images]

    print("num_available_annotations", num_available_annotations)


    # # tentatively assign half of images to training set
    # training_set = random.sample(available_images, len(available_images) // 2)
    # validation_set = [img for img in available_images if img not in training_set]
    # num_training_annotations = np.sum([np.shape(boxes[img])[0] for img in training_set])
    # num_validation_annotations = np.sum([np.shape(boxes[img])[0] for img in validation_set])

    # THRESH_HALF = 3000
    # if num_training_annotations > TRAINING_THRESH and num_validation_annotations:
    #     # assign rest to validation



    # THRESH_NO_TRANSFER = 3000
    # if num_available_annotations > THRESH_NO_TRANSFER:
    #     # need to ensure we have sufficient number of annotations that can be used in training set








def run():
    """
        passed information:
            group name
            farm, field, and mission that model is built to test well on
            images reserved for testing (can be empty)
    """

    #dataset_partitions(model_dir)

    config = {

        "group_name": "UNI_transfer_00",
        "group_description": "test",
        "on_exception": "destroy_and_raise",
        "replications": 1,

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
                    "training_datasets": [
                        {
                            "farm_name": "UNI",
                            "field_name": "LowN1",
                            "mission_date": "2021-06-07",
                            "image_names": ["1", "6", "8", "16", "26", "31", "36", "40"],
                            "patch_extraction_params": {
                                "method": "tile",
                                "patch_size": 300,
                                "patch_overlap_percent": 50
                            }
                        }
                    ],
                    "validation_datasets": [
                        {
                            "farm_name": "UNI",
                            "field_name": "LowN1",
                            "mission_date": "2021-06-07",
                            "image_names": ["4", "11", "14", "18", "21"],
                            "patch_extraction_params": {
                                "method": "tile",
                                "patch_size": 300,
                                "patch_overlap_percent": 50
                            }
                        }
                    ],

                    # "training_datasets": [
                    #     {
                    #         "farm_name": "row_spacing",
                    #         "field_name": "brown",
                    #         "mission_date": "2021-06-01",
                    #         "image_names": ["102", "203", "302", "308", "503", "602", "608", "702"],
                    #         "patch_extraction_params": {
                    #             "method": "tile",
                    #             "patch_size": 200,
                    #             "patch_overlap_percent": 50
                    #         }
                    #     }
                    # ],
                    # "validation_datasets": [
                    #     {
                    #         "farm_name": "row_spacing",
                    #         "field_name": "brown",
                    #         "mission_date": "2021-06-01",
                    #         "image_names": ["108", "303", "502", "603", "703"],
                    #         "patch_extraction_params": {
                    #             "method": "tile",
                    #             "patch_size": 200,
                    #             "patch_overlap_percent": 50
                    #         }
                    #     }
                    # ],


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
                    "max_num_epochs": 300,
                    "early_stopping": {
                        "apply": True,
                        "monitor": "validation_loss",
                        "num_epochs_tolerance": 15
                    },
                    "batch_size": 16,

                    "save_method": "best_validation_loss",
                    "percent_of_training_set_used": 100,
                    "percent_of_validation_set_used": 100                    

                }
            ]
        },


        "inference_config": {


            "image_sets": [
                {
                    "farm_name": "UNI",
                    "field_name": "LowN1",
                    "mission_date": "2021-06-07",
                    "training_image_names": ["1", "6", "8", "16", "26", "31", "36", "40"],
                    "validation_image_names": ["108", "303", "502", "603", "703"],
                    "test_image_names": [],
                    "patch_extraction_params": {
                        "method": "tile",
                        "patch_size": 300,
                        "patch_overlap_percent": 50
                    }
                },
                # {
                #     "farm_name": "row_spacing",
                #     "field_name": "brown",
                #     "mission_date": "2021-06-01",
                #     "training_image_names": ["102", "203", "302", "308", "503", "602", "608", "702"],
                #     "validation_image_names": ["108", "303", "502", "603", "703"],
                #     "test_image_names": [],
                #     "patch_extraction_params": {
                #         "method": "tile",
                #         "patch_size": 200,
                #         "patch_overlap_percent": 50
                #     }
                # },
                # {
                #     "farm_name": "row_spacing",
                #     "field_name": "nasser",
                #     "mission_date": "2020-06-08",
                #     "training_image_names": [],
                #     "validation_image_names": [],
                #     "test_image_names": [],
                #     "patch_extraction_params": {
                #         "method": "tile",
                #         "patch_size": 200,
                #         "patch_overlap_percent": 50
                #     }
                # }
                {
                    "farm_name": "BlaineLake",
                    "field_name": "River",
                    "mission_date": "2021-06-09",
                    "training_image_names": [],
                    "validation_image_names": [],
                    "test_image_names": [],
                    "patch_extraction_params": {
                        "method": "tile",
                        "patch_size": 300,
                        "patch_overlap_percent": 50
                    }
                },
            ],

            "patch_border_buffer_percent": 0.10,
            "batch_size": 16,
            "patch_nms_iou_thresh": 0.4,
            "image_nms_iou_thresh": 0.4,
            "score_thresh": 0.5
        }
    }

    print("Running Group")
    group_interface.run_group(config)