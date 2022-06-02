import uuid



def generate_model_names(num):
    return ["model_" + str(i+1) for i in range(num)]


def generate_model_uuids(num):
    return [str(uuid.uuid4()) for _ in range(num)]




def add_models_to_job_config(job_config):

    # if "variation_config" in job_config:
    #     num_models = len(job_config["variation_config"]["param_values"]) * job_config["replications"]
    # else:
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



def fill_job_config(job_config):

    if "arch" not in job_config:
        job_config["arch"] = {}
    if "training" not in job_config:
        job_config["training"] = {}
    if "inference" not in job_config:
        job_config["inference"] = {}


    job_config["arch"].update({
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
    })

    job_config["training"].update({
        "training_sequence": [  

            {

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
                    },
                    # {
                    #     "type": "brightness_contrast",
                    #     "parameters": {
                    #         "probability": 1.0, 
                    #         "brightness_limit": [-0.2, 0.2], 
                    #         "contrast_limit": [-0.2, 0.2]
                    #     }
                    # }
                    # {
                    #     "type": "affine",
                    #     "parameters": {
                    #         "probability": 1.0, 
                    #         "scale": 1.0, 
                    #         "translate_percent": (-0.3, 0.3), 
                    #         "rotate": 0, 
                    #         "shear": 0
                    #     }
                    # }
                ],

                "min_num_epochs": 4, #15,
                "max_num_epochs": 4, #3000,
                "early_stopping": {
                    "apply": True,
                    "monitor": "validation_loss",
                    "num_epochs_tolerance": 30
                },
                "batch_size": 16,

                "save_method": "best_validation_loss",
                "percent_of_training_set_used": 100,
                "percent_of_validation_set_used": 100
            }
        ]
    })

    job_config["inference"].update({
        "batch_size": 16,
        "patch_nms_iou_thresh": 0.4,
        "image_nms_iou_thresh": 0.4,
        "score_thresh": 0.5,
        "predict_on_completed_only": False
    })

    add_models_to_job_config(job_config)
