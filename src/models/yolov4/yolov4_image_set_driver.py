
import os
import shutil
import glob

import tqdm
import logging
import time
import math as m
import numpy as np
import tensorflow as tf
import uuid
import cv2

from mean_average_precision import MetricBuilder

from io_utils import w3c_io, json_io
from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          model_keys, \
                          wbf, \
                          data_augment
from image_set import Image

import image_set_actions as isa
import image_set_aux

import extract_patches as ep

# from image_set_actions import check_restart

from models.yolov4.loss import YOLOv4Loss
from models.yolov4.yolov4 import YOLOv4, YOLOv4Tiny
import models.yolov4.data_load as data_load
from models.yolov4.encode import Decoder
from models.yolov4.yolov4_driver import post_process_sample

VALIDATION_IMPROVEMENT_TOLERANCE = 10
TRAINING_TIME_SESSION_CEILING = 5000            # number of seconds before current session is stopped in order to give others a chance

def create_default_config():


    config = {
        "model_name": "model_1",
        "model_uuid": str(uuid.uuid4()),
        "arch": {
            "model_type": "yolov4_tiny", #"yolov4", #"yolov4_tiny",
            "backbone_config": {
                "backbone_type": "csp_darknet53_tiny" #"csp_darknet53" #"csp_darknet53_tiny"
            },
            "neck_config": {
                "neck_type": "yolov4_tiny_deconv" #"spp_pan" #"yolov4_tiny_deconv"
            },
            "max_detections": 50,
            "input_image_shape": [416, 416, 3],
            "class_map": {"plant": 0}
        },

        "training": {

            "learning_rate_schedule": {
                "schedule_type": "constant",
                "learning_rate": 0.0001
            },


            "data_augmentations": [

                # {
                #     "type": "CLAHE",
                #     "parameters": {
                #         "probability": 1.0,
                #     }
                # }
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
            "batch_size": 16, #8, #16, #256, #128, #16, #64, #64, #1024, #32, # 16,

            #"save_method": "best_validation_loss",
            "percent_of_training_set_used": 100,
            "percent_of_validation_set_used": 100
        },
        "inference": {
            "batch_size": 64, #256, #32, #16,
            "patch_nms_iou_thresh": 0.4,
            "image_nms_iou_thresh": 0.4,
            "score_thresh": 0.25,
            #"predict_on_completed_only": False    

        }
    }

    return config


def update_loss_record(loss_record, key, cur_loss):

    loss_record[key]["values"].append(cur_loss)

    best = loss_record[key]["best"]
    #if cur_loss < best:
    epsilon = 1e-5
    if best - cur_loss > epsilon:
        loss_record[key]["best"] = cur_loss
        loss_record[key]["epochs_since_improvement"] = 0
        return True
    else:
        loss_record[key]["epochs_since_improvement"] += 1
        return False



# def select_baseline(farm_name, field_name, mission_date):


#     logger = logging.getLogger(__name__)

#     image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
#     model_dir = os.path.join(image_set_dir, "model")
#     images_dir = os.path.join(image_set_dir, "images")
#     weights_dir = os.path.join(model_dir, "weights")

#     cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")
#     best_weights_path = os.path.join(weights_dir, "best_weights.h5")

#     annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
#     annotations_json = json_io.load_json(annotations_path)
#     annotations = w3c_io.convert_json_annotations(annotations_json, {"plant": 0})


#     baseline_weights_dir = os.path.join("usr", "data", "baselines")
    
#     config = create_default_config()
#     model_keys.add_general_keys(config)
#     model_keys.add_specialized_keys(config)

#     # config["training"]["active"] = {}
#     # for k in config["training"]:
#     #     config["training"]["active"][k] = config["training"][k]

#     weight_paths = glob.glob(os.path.join(baseline_weights_dir, "*"))
#     #weight_paths.append("random")

#     if len(weight_paths) == 0:
#         yolov4 = YOLOv4Tiny(config)
#         yolov4.save_weights(filepath=best_weights_path, save_format="h5")
#         return
#     if len(weight_paths) == 1:
#         shutil.copyfile(weight_paths[0], cur_weights_path)
#         shutil.copyfile(weight_paths[0], best_weights_path)
#         return


#     decoder = Decoder(config)
#     #loss_fn = YOLOv4Loss(config)

#     tf_record_path = os.path.join(image_set_dir, "model", "init", "patches-record.tfrec")
#     # loader_is_preloaded, data_loader = data_load.get_data_loader([tf_record_path], config, shuffle=False, augment=False)
    

#     # tf_dataset, tf_dataset_size = data_loader.create_batched_dataset(
#     #                               take_percent=config["training"]["active"]["percent_of_validation_set_used"])



#     data_loader = data_load.InferenceDataLoader(tf_record_path, config) #InferenceDataLoader(tf_record_path, config)
#     tf_dataset, tf_dataset_size = data_loader.create_dataset()
    
#     scores = {}
#     for weight_path in weight_paths:
#         print("weight_path", weight_path)
#         yolov4 = YOLOv4Tiny(config)
#         #input_shape = (config["training"]["active"]["batch_size"], *(data_loader.get_model_input_shape()))
#         input_shape = (config["inference"]["batch_size"], *(data_loader.get_model_input_shape()))
#         yolov4.build(input_shape=input_shape)

#         if weight_path != "random":
#             yolov4.load_weights(weight_path, by_name=False)

#         weight_name = os.path.basename(weight_path)
#         scores[weight_name] = 0

#         steps = np.sum([1 for _ in tf_dataset])


#         # bar = tqdm.tqdm(tf_dataset, total=steps)

#         # loss_value = 0
#         # for batch_data in bar:
#         #     batch_images, batch_labels = data_loader.read_batch_data(batch_data)
#         #     conv = yolov4(batch_images, training=False)
#         #     loss_value = loss_fn(batch_labels, conv)
#         #     loss_value += sum(yolov4.losses)

#         #     scores[weight_name] += loss_value.numpy()




#         logger.info("{} ('{}'): Running inference on {} images.".format(config["arch"]["model_type"], 
#                                                                         config["model_name"], 
#                                                                         tf_dataset_size))

#         # debug_out_dir = os.path.join("usr", "data", "tmp_patch_data_ex", 
#         #                     weight_name)
#         # if os.path.exists(debug_out_dir):
#         #     shutil.rmtree(debug_out_dir)
#         # os.makedirs(debug_out_dir)


#         for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

#             batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, True) #False) #is_annotated)
#             batch_size = batch_images.shape[0]

#             pred = yolov4(batch_images, training=False)
#             detections = decoder(pred)

#             batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

#             batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

#             for i in range(batch_size):

#                 pred_bbox = batch_pred_bbox[i]
#                 ratio = batch_ratios[i]

#                 patch_info = batch_info[i]

#                 image_path = bytes.decode((patch_info["image_path"]).numpy())
#                 patch_path = bytes.decode((patch_info["patch_path"]).numpy())
#                 image_name = os.path.basename(image_path)[:-4]
#                 #patch_name = bytes.decode((patch_info["patch_name"]).numpy())
#                 patch_name = os.path.basename(patch_path)[:-4]
#                 patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

#                 #if is_annotated:
#                 patch_abs_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy()
#                 patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy()


#                 pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
#                         post_process_sample(pred_bbox, ratio, patch_coords, config, score_threshold=0.50) #config["inference"]["score_thresh"])


#                 # pred_abs_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
#                 # pred_classes = np.array(image_predictions[image_name]["pred_classes"])
#                 # pred_scores = np.array(image_predictions[image_name]["pred_scores"])
#                 # print("pred_patch_abs_boxes", (pred_patch_abs_boxes.shape))
#                 # print("pred_patch_scores", (pred_patch_scores.shape))
#                 # print("pred_patch_classes", (pred_patch_classes.shape))


#                 # pred_for_mAP, true_for_mAP = inference_metrics.get_pred_and_true_for_mAP(pred_patch_abs_boxes, pred_patch_classes, pred_patch_scores,
#                 #                                                     patch_abs_boxes, patch_classes)

#                 # image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
#                 # image_metric_fn.add(pred_for_mAP, true_for_mAP)
#                 # #pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
#                 # coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

#                 # #relative_detection_accuracy = 1.0 - abs((pred_patch_classes.size - patch_classes.size) / pred_patch_classes.size)

#                 # print("coco_mAP: {} | gt_num: {} | pred_num: {}".format(coco_mAP, patch_classes.size, pred_patch_classes.size))
#                 # scores[weight_name] += coco_mAP
#                 # #scores[weight_name] += relative_detection_accuracy

#                 # patch_height = patch_coords[2] - patch_coords[0]
#                 # patch_width = patch_coords[3] - patch_coords[1]

#                 # accept_bottom = round(patch_height / 4)
#                 # accept_left = round(patch_width / 4)
#                 # accept_top = patch_height - round(patch_height / 4)
#                 # accept_right = patch_width - round(patch_width / 4)


#                 # box_centres = (patch_abs_boxes[..., :2] + patch_abs_boxes[..., 2:]) / 2.0
#                 # mask = np.logical_and(
#                 #     np.logical_and(box_centres[:,0] >= accept_bottom, box_centres[:,0] < accept_top),
#                 #     np.logical_and(box_centres[:,1] >= accept_left, box_centres[:,1] < accept_right)
#                 # )

#                 # #patch_abs_boxes = patch_abs_boxes[mask]

#                 # pred_box_centres = (pred_patch_abs_boxes[..., :2] + pred_patch_abs_boxes[..., 2:]) / 2.0
#                 # pred_mask = np.logical_and(
#                 #     np.logical_and(pred_box_centres[:,0] >= accept_bottom, pred_box_centres[:,0] < accept_top),
#                 #     np.logical_and(pred_box_centres[:,1] >= accept_left, pred_box_centres[:,1] < accept_right)
#                 # )

#                 # #pred_box_centres = pred_box_centres[mask]                


#                 # num_actual = int(np.sum(mask))
#                 # num_pred = int(np.sum(pred_mask))               

#                 # patch = (cv2.cvtColor(cv2.imread(patch_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
#                 # output_patch(patch, patch_abs_boxes, pred_boxes=pred_patch_abs_boxes, 
#                 # pred_classes=pred_patch_classes, pred_scores=pred_patch_scores, 
#                 #             out_path=os.path.join(debug_out_dir, patch_name + ".png"))
                
#                 #scores[weight_name] += abs(num_pred - num_actual)
#                 scores[weight_name] += abs(pred_patch_classes.size - patch_classes.size)

#                 # if pred_patch_abs_boxes.size > 0:
#                 #     #iou = box_utils.compute_iou_np(patch_abs_boxes, pred_patch_abs_boxes)
#                 #     iou = box_utils.compute_iou_np(pred_patch_abs_boxes, patch_abs_boxes)
                    
#                 #     if iou.shape[1] > 0:
#                 #         for i in range(iou.shape[0]):
                            
#                 #             max_iou_ind = np.argmax(iou[i, :])
#                 #             max_iou = iou[i, max_iou_ind]
#                 #             #max_iou_score = pred_patch_scores[max_iou_ind]

#                 #             scores[weight_name] += max_iou # (max_iou * max_iou_score)

        
#     print("scores:")
#     json_io.print_json(scores)

#     best_score_weight_name = min(scores, key=scores.get)
#     print("best weights name", best_score_weight_name)
    
#     shutil.copyfile(os.path.join(baseline_weights_dir, best_score_weight_name),
#                     cur_weights_path)
#     shutil.copyfile(os.path.join(baseline_weights_dir, best_score_weight_name),
#                     best_weights_path)


def predict(sch_ctx, image_set_dir, image_names, save_result):

    start_time = time.time()

    logger = logging.getLogger(__name__)

    pieces = image_set_dir.split("/")
    username = pieces[-5]
    farm_name = pieces[-3]
    field_name = pieces[-2]
    mission_date = pieces[-1]


    model_dir = os.path.join(image_set_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    predictions_dir = os.path.join(model_dir, "prediction")
    results_dir = os.path.join(model_dir, "results")
    
    excess_green_record_path = os.path.join(image_set_dir, "excess_green", "record.json")
    excess_green_record = json_io.load_json(excess_green_record_path)


    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations_json = json_io.load_json(annotations_path)
    annotations = w3c_io.convert_json_annotations(annotations_json, {"plant": 0})

    training_image_names = []
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed_for_training":
            training_image_names.append(image_name) 



    config = create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    if config["arch"]["model_type"] == "yolov4":
        yolov4 = YOLOv4(config)
    elif config["arch"]["model_type"] == "yolov4_tiny":
        yolov4 = YOLOv4Tiny(config)

    decoder = Decoder(config)

    image_predictions = {}

    # transform_types = ["nop"]

    best_weights_path = os.path.join(weights_dir, "best_weights.h5")
    if not os.path.exists(best_weights_path):
        raise RuntimeError("Model weights could not be located.")
        # status_path = os.path.join(model_dir, "status.json")
        # status = json_io.load_json(status_path)
        # model_name = status["model_name"]
        # model_creator = status["model_creator"]
        # model_path = os.path.join("usr", "data", model_creator, "models")
        # public_weights_path = os.path.join(model_path, "available", "public", model_name, "weights.h5")
        # private_weights_path = os.path.join(model_path, "available", "private", model_name, "weights.h5")

        # print("public_weights_path: {}".format(public_weights_path))
        # print("private_weights_path: {}".format(private_weights_path))

        # if os.path.exists(public_weights_path):
        #     best_weights_path = public_weights_path
        # elif os.path.exists(private_weights_path):
        #     best_weights_path = private_weights_path
        # else:
        #     raise RuntimeError("Model weights could not be located.")

    input_shape = (config["inference"]["batch_size"], *(config["arch"]["input_image_shape"]))
    yolov4.build(input_shape=input_shape)
    yolov4.load_weights(best_weights_path, by_name=False)


    loss_record_path = os.path.join(image_set_dir, "model", "training", "loss_record.json")
    loss_record = json_io.load_json(loss_record_path)
    # if len(loss_record["training_loss"]["values"]) == 0:
    # updated_patch_size = ep.get_updated_patch_size(annotations, training_image_names)

    status_path = os.path.join(model_dir, "status.json")
    status = json_io.load_json(status_path)
    model_patch_size = status["patch_size"]


    for image_index, image_name in enumerate(image_names):

        image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
        image = Image(image_path)
        
        patch_records = ep.extract_patch_records_from_image_tiled(
                image, 
                model_patch_size,
                image_annotations=None,
                patch_overlap_percent=50, 
                include_patch_arrays=True)

        data_loader = data_load.InferenceDataLoader(patch_records, config)
        tf_dataset = data_loader.create_dataset()

        # input_shape = (config["inference"]["batch_size"], *(data_loader.get_model_input_shape()))
        # yolov4.build(input_shape=input_shape)

        # best_weights_path = os.path.join(weights_dir, "best_weights.h5")
        # yolov4.load_weights(best_weights_path, by_name=False)

        #steps = np.sum([1 for _ in tf_dataset])

        # logger.info("{} ('{}'): Running inference on {} patches.".format(config["arch"]["model_type"], 
        #                                                                 config["model_name"], 
        #                                                                 tf_dataset_size))

        logger.info("Running inference for image {} ({}/{})".format(image_name, image_index+1, len(image_names)))

        for batch_data in tf_dataset: #tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):
            
            if sch_ctx["switch_queue"].size() > 0:
                affected = drain_switch_queue(sch_ctx, cur_image_set_dir=image_set_dir)
                if affected:
                    end_time = int(time.time())
                    return True, end_time
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.PREDICTING,
                                         extra_items={"num_processed": image_index, "num_images": len(image_names)})   


            batch_images, batch_ratios, batch_indices = data_loader.read_batch_data(batch_data)
            batch_size = batch_images.shape[0]
            
            # batch_images = data_augment.apply_inference_transform(batch_images, transform_type)


            pred = yolov4(batch_images, training=False)
            detections = decoder(pred)

            batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

            batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

            for i in range(batch_size):

                pred_bbox = batch_pred_bbox[i]
                ratio = batch_ratios[i]

                patch_index = batch_indices[i]

                # image_path = bytes.decode((patch_info["image_path"]).numpy())
                image_path = patch_records[patch_index]["image_path"]
                # patch_path = bytes.decode((patch_info["patch_path"]).numpy())
                # image_name = os.path.basename(image_path)[:-4]
                #patch_name = os.path.basename(patch_path)[:-4]
                patch_coords = patch_records[patch_index]["patch_coords"] #tf.sparse.to_dense(patch_records[patch_index]["patch_coords"]).numpy().astype(np.int32)


                # pred_bbox = data_augment.undo_inference_transform(pred_bbox, transform_type)

                #if is_annotated:
                #    patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                #    patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()
                # pred_bbox = \
                #     data_augment.undo_inference_transform(batch_images[i].numpy(), pred_bbox, 
                #                                           transform_type)


                pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
                        post_process_sample(pred_bbox, ratio, patch_coords, config, score_threshold=config["inference"]["score_thresh"])

                # resize to input size


                if image_name not in image_predictions:
                    image_predictions[image_name] = {
                            "image_path": image_path,
                            "pred_image_abs_boxes": [],
                            "pred_classes": [],
                            "pred_scores": [],
                            #"patch_coords": []               
                    }
                    
                    

                pred_image_abs_boxes, pred_image_scores, pred_image_classes = \
                    driver_utils.get_image_detections(pred_patch_abs_boxes, 
                                                    pred_patch_scores, 
                                                    pred_patch_classes, 
                                                    patch_coords, 
                                                    image_path, 
                                                    trim=True) #False) #True)
                                                    
                # print("pred_image_abs_boxes", pred_image_abs_boxes)
                image_predictions[image_name]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
                image_predictions[image_name]["pred_scores"].extend(pred_image_scores.tolist())
                image_predictions[image_name]["pred_classes"].extend(pred_image_classes.tolist())
                #image_predictions[transform_type][image_name]["patch_coords"].append(patch_coords.tolist())


        isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.PREDICTING,
                                 extra_items={"num_processed": image_index+1, "num_images": len(image_names)})            

    driver_utils.clip_image_boxes(image_predictions)

    driver_utils.apply_nms_to_image_boxes(image_predictions, 
                                            iou_thresh=config["inference"]["image_nms_iou_thresh"])


    # for image_name in image_predictions.keys():
    #     image_predictions[image_name] = image_predictions[image_name][transform_types[0]]




    for image_name in image_names:
        image_predictions_dir = os.path.join(predictions_dir, "images", image_name)
        os.makedirs(image_predictions_dir, exist_ok=True)
        predictions_w3c_path = os.path.join(image_predictions_dir, "predictions_w3c.json")
        # metrics_path = os.path.join(image_predictions_dir, "metrics.json")
        w3c_io.save_predictions(predictions_w3c_path, {image_name: image_predictions[image_name]}, config)
        # if image_name in metrics:
        #     json_io.save_json(metrics_path, {image_name: metrics[image_name]})


    end_time = int(time.time())
    if save_result:
        metrics = inference_metrics.collect_image_set_metrics(image_predictions, annotations, config)
        
        image_set_results_dir = os.path.join(results_dir, str(end_time))
        os.makedirs(image_set_results_dir)
        annotations_path = os.path.join(image_set_results_dir, "annotations_w3c.json")
        json_io.save_json(annotations_path, annotations_json)
        excess_green_record_path = os.path.join(image_set_results_dir, "excess_green_record.json")
        json_io.save_json(excess_green_record_path, excess_green_record)
        image_predictions_path = os.path.join(image_set_results_dir, "predictions_w3c.json")
        w3c_io.save_predictions(image_predictions_path, image_predictions, config)
        metrics_path = os.path.join(image_set_results_dir, "metrics.json")
        json_io.save_json(metrics_path, metrics)
        # report_path = os.path.join(image_set_results_dir, "results.csv") #xlsx")
        # inference_metrics.prepare_report(report_path, farm_name, field_name, mission_date, 
        #                                  image_predictions, annotations, excess_green_record, metrics)
    elapsed_time = end_time - start_time
    logger.info("Finished predicting. Elapsed time: {}".format(elapsed_time))

    return False, end_time # image_predictions








def predict_org(image_set_dir, annotations_json, annotations, image_names, save_result, save_image_predictions=True):

    logger = logging.getLogger(__name__)

    #image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    predictions_dir = os.path.join(model_dir, "prediction")
    results_dir = os.path.join(model_dir, "results")

    #patches_dir = os.path.join(image_set_dir, "patches")
    #patch_data_path = os.path.join(patches_dir, "patch_data.json")
    #patch_data = json_io.load_json(patch_data_path)

    # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    # annotations_json = json_io.load_json(annotations_path)
    # annotations = w3c_io.convert_json_annotations(annotations_json, {"plant": 0})

    excess_green_record_path = os.path.join(image_set_dir, "excess_green", "record.json")
    excess_green_record = json_io.load_json(excess_green_record_path)


    config = create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    if config["arch"]["model_type"] == "yolov4":
        yolov4 = YOLOv4(config)
    elif config["arch"]["model_type"] == "yolov4_tiny":
        yolov4 = YOLOv4Tiny(config)

    decoder = Decoder(config)

    image_predictions = {}

    transform_types = ["nop"] #"nop"] #, "flip_horizontal", "flip_vertical"]

    for image_name in image_names:

        image_predictions_dir = os.path.join(predictions_dir, "images", image_name)
        tf_record_path = os.path.join(image_predictions_dir, "patches-record.tfrec")

        data_loader = data_load.InferenceDataLoaderOrg(tf_record_path, config)
        tf_dataset, tf_dataset_size = data_loader.create_dataset()

        input_shape = (config["inference"]["batch_size"], *(data_loader.get_model_input_shape()))
        #print("input_shape", input_shape)
        yolov4.build(input_shape=input_shape)

        best_weights_path = os.path.join(weights_dir, "best_weights.h5")
        yolov4.load_weights(best_weights_path, by_name=False)

        steps = np.sum([1 for _ in tf_dataset])

        logger.info("{} ('{}'): Running inference on {} patches.".format(config["arch"]["model_type"], 
                                                                        config["model_name"], 
                                                                        tf_dataset_size))


        # ISSUE: problem if image annotation status is changed 

        #is_annotated = annotations[image_name]["status"] == "completed"

        #image_status = patch_data[image_name]["status"]
        #is_annotated = image_status == "completed_for_training" or image_status == "completed_for_testing"


        for transform_type in transform_types:
            #image_predictions[transform_type] = {}

            for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

                # if check_restart():
                #     return False

                batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, False) #is_annotated)
                batch_size = batch_images.shape[0]

                #print("batch_images.shape", batch_images.shape)
                batch_images = data_augment.apply_inference_transform(batch_images, transform_type)


                pred = yolov4(batch_images, training=False)
                detections = decoder(pred)

                batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

                batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

                for i in range(batch_size):

                    pred_bbox = batch_pred_bbox[i]
                    ratio = batch_ratios[i]

                    patch_info = batch_info[i]

                    image_path = bytes.decode((patch_info["image_path"]).numpy())
                    patch_path = bytes.decode((patch_info["patch_path"]).numpy())
                    image_name = os.path.basename(image_path)[:-4]
                    #patch_name = os.path.basename(patch_path)[:-4]
                    patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)


                    # pred_bbox = data_augment.undo_inference_transform(pred_bbox, transform_type)

                    #if is_annotated:
                    #    patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                    #    patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()
                    # pred_bbox = \
                    #     data_augment.undo_inference_transform(batch_images[i].numpy(), pred_bbox, 
                    #                                           transform_type)


                    pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
                            post_process_sample(pred_bbox, ratio, patch_coords, config, score_threshold=config["inference"]["score_thresh"])

                    # resize to input size

                    pred_patch_abs_boxes, pred_patch_classes, pred_patch_scores = \
                        data_augment.undo_inference_transform(patch_path, pred_patch_abs_boxes, pred_patch_classes, pred_patch_scores, 
                                                              transform_type)

                    if image_name not in image_predictions:
                        image_predictions[image_name] = {}
                    if transform_type not in image_predictions[image_name]:
                        image_predictions[image_name][transform_type] = {
                                "image_path": image_path,
                                "pred_image_abs_boxes": [],
                                "pred_classes": [],
                                "pred_scores": [],
                                #"patch_coords": []               
                        }
                        
                        

                    pred_image_abs_boxes, pred_image_scores, pred_image_classes = \
                        driver_utils.get_image_detections(pred_patch_abs_boxes, 
                                                        pred_patch_scores, 
                                                        pred_patch_classes, 
                                                        patch_coords, 
                                                        image_path, 
                                                        trim=True) #False) #True)
                                                        
                    # print("pred_image_abs_boxes", pred_image_abs_boxes)
                    image_predictions[image_name][transform_type]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
                    image_predictions[image_name][transform_type]["pred_scores"].extend(pred_image_scores.tolist())
                    image_predictions[image_name][transform_type]["pred_classes"].extend(pred_image_classes.tolist())
                    #image_predictions[transform_type][image_name]["patch_coords"].append(patch_coords.tolist())
                        

    driver_utils.clip_image_boxes(image_predictions)

    driver_utils.apply_nms_to_image_boxes(image_predictions, 
                                            iou_thresh=config["inference"]["image_nms_iou_thresh"])


    if len(transform_types) == 1:

        for image_name in image_predictions.keys():
            image_predictions[image_name] = image_predictions[image_name][transform_types[0]]
    else:
        for image_name in image_predictions.keys():
            all_boxes = []
            all_scores = []
            all_classes = []
            #weights = []

            for transform_type in image_predictions[image_name].keys():

                width, height = Image(image_predictions[image_name][transform_type]["image_path"]).get_wh()
                boxes = np.array(image_predictions[image_name][transform_type]["pred_image_abs_boxes"])
                # print("boxes", boxes)
                image_normalized_boxes = np.stack([
                    boxes[:, 0] / height,
                    boxes[:, 1] / width,
                    boxes[:, 2] / height,
                    boxes[:, 3] / width
                ], axis=-1).tolist()


                all_boxes.append(image_normalized_boxes) #image_predictions[image_name][transform_type]["pred_image_abs_boxes"])
                all_scores.append(image_predictions[image_name][transform_type]["pred_scores"])
                all_classes.append(image_predictions[image_name][transform_type]["pred_classes"])
                #weights.append(1)

            iou_thr = config["inference"]["image_nms_iou_thresh"] #0.5
            skip_box_thr = 0.0001
            

            pred_image_normalized_boxes, pred_scores, pred_classes = wbf.weighted_boxes_fusion(
                all_boxes, 
                all_scores, 
                all_classes,
                iou_thr=iou_thr, 
                skip_box_thr=skip_box_thr)

            pred_image_abs_boxes = np.rint(np.stack([
                    pred_image_normalized_boxes[:, 0] * height,
                    pred_image_normalized_boxes[:, 1] * width,
                    pred_image_normalized_boxes[:, 2] * height,
                    pred_image_normalized_boxes[:, 3] * width
                ], axis=-1)).astype(np.int64)

            image_predictions[image_name] = {
                "pred_image_abs_boxes": pred_image_abs_boxes.tolist(),
                "pred_scores": pred_scores.tolist(),
                "pred_classes": pred_classes.tolist()
            }

        #driver_utils.add_class_detections(image_predictions, config)

        #metrics = driver_utils.create_metrics_skeleton(dataset)

        #all_image_names = predictions["image_predictions"].keys()
        #inference_metrics.collect_statistics(all_image_names, metrics, predictions, config, inference_times=inference_times)
        #inference_metrics.collect_metrics(all_image_names, metrics, predictions, dataset, config)
    

    if save_image_predictions:
        for image_name in image_names:
            image_predictions_dir = os.path.join(predictions_dir, "images", image_name)
            predictions_w3c_path = os.path.join(image_predictions_dir, "predictions_w3c.json")
            # metrics_path = os.path.join(image_predictions_dir, "metrics.json")
            w3c_io.save_predictions(predictions_w3c_path, {image_name: image_predictions[image_name]}, config)
            # if image_name in metrics:
            #     json_io.save_json(metrics_path, {image_name: metrics[image_name]})


    end_time = int(time.time())
    if save_result:
        metrics = inference_metrics.collect_image_set_metrics(image_predictions, annotations, config)
        
        image_set_results_dir = os.path.join(results_dir, str(end_time))
        os.makedirs(image_set_results_dir)
        annotations_path = os.path.join(image_set_results_dir, "annotations_w3c.json")
        json_io.save_json(annotations_path, annotations_json)
        excess_green_record_path = os.path.join(image_set_results_dir, "excess_green_record.json")
        json_io.save_json(excess_green_record_path, excess_green_record)
        image_predictions_path = os.path.join(image_set_results_dir, "predictions_w3c.json")
        w3c_io.save_predictions(image_predictions_path, image_predictions, config)
        metrics_path = os.path.join(image_set_results_dir, "metrics.json")
        json_io.save_json(metrics_path, metrics)
        # report_path = os.path.join(image_set_results_dir, "results.csv") #xlsx")
        # inference_metrics.prepare_report(report_path, farm_name, field_name, mission_date, 
        #                                  image_predictions, annotations, excess_green_record, metrics)
        
        
    return end_time, image_predictions



# def train_baseline(request):

#     logger = logging.getLogger(__name__)

#     tf.keras.backend.clear_session()

#     baseline_name = request["baseline_name"]

#     #os.makedirs(baseline_name)

#     patch_records = ep.extract_patch_records_from_image_tiled(
#         image, 
#         updated_patch_size,
#         image_annotations=None,
#         patch_overlap_percent=50, 
#         include_patch_arrays=False)



def train_baseline(root_dir, sch_ctx):
    
    logger = logging.getLogger(__name__)

    pieces = root_dir.split("/")
    username = pieces[-4]

    tf.keras.backend.clear_session()

    start_time = int(time.time())

    model_dir = os.path.join(root_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    training_dir = os.path.join(model_dir, "training")

    training_tf_record_paths = [os.path.join(training_dir, "training-patches-record.tfrec")]
    validation_tf_record_paths = [os.path.join(training_dir, "validation-patches-record.tfrec")]

    config = create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    config["training"]["active"] = {}
    for k in config["training"]:
        config["training"]["active"][k] = config["training"][k]

    # train_loader_is_preloaded, train_data_loader = data_load.get_data_loader(training_tf_record_paths, config, shuffle=True, augment=True)
    # val_loader_is_preloaded, val_data_loader = data_load.get_data_loader(validation_tf_record_paths, config, shuffle=False, augment=False)
    
    train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
    val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)


    # logger.info("Data loaders created. Train loader is preloaded?: {}. Validation loader is preloaded?: {}.".format(
    #     train_loader_is_preloaded, val_loader_is_preloaded
    # ))


    train_dataset, num_train_patches = train_data_loader.create_batched_dataset(
                                      take_percent=config["training"]["active"]["percent_of_training_set_used"])
    # if check_restart():
    #     return False
    val_dataset, num_val_patches = val_data_loader.create_batched_dataset(
                                  take_percent=config["training"]["active"]["percent_of_validation_set_used"])


    logger.info("Building model...")


    if config["arch"]["model_type"] == "yolov4":
        yolov4 = YOLOv4(config)
    elif config["arch"]["model_type"] == "yolov4_tiny":
        yolov4 = YOLOv4Tiny(config)

    loss_fn = YOLOv4Loss(config)


    input_shape = (config["training"]["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
    yolov4.build(input_shape=input_shape)

    logger.info("Model built.")


    cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")
    best_weights_path = os.path.join(weights_dir, "best_weights.h5")
    if os.path.exists(cur_weights_path):
        logger.info("Loading weights...")
        yolov4.load_weights(cur_weights_path, by_name=False)
        logger.info("Weights loaded.")
    else:
        logger.info("No initial weights found.")


    optimizer = tf.optimizers.Adam()


    train_loss_metric = tf.metrics.Mean()
    val_loss_metric = tf.metrics.Mean()

    @tf.function
    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            conv = yolov4(batch_images, training=True)
            loss_value = loss_fn(batch_labels, conv)
            loss_value += sum(yolov4.losses)

        #if np.isnan(loss_value):
        #    raise RuntimeError("NaN loss has occurred.")
        gradients = tape.gradient(target=loss_value, sources=yolov4.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, yolov4.trainable_variables))
        train_loss_metric.update_state(values=loss_value)

    train_steps_per_epoch = np.sum([1 for _ in train_dataset])
    val_steps_per_epoch = np.sum([1 for _ in val_dataset])

    logger.info("{} ('{}'): Starting to train with {} training patches and {} validation patches.".format(
                    config["arch"]["model_type"], config["model_name"], num_train_patches, num_val_patches))



    while True:


        loss_record_path = os.path.join(training_dir, "loss_record.json")
        loss_record = json_io.load_json(loss_record_path)

        logger.info("Epochs since validation loss improvement: {}".format(loss_record["validation_loss"]["epochs_since_improvement"]))

        train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
        for batch_data in train_bar:

            # optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))
            optimizer.lr.assign(config["training"]["active"]["learning_rate_schedule"]["learning_rate"])

            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)

            train_step(batch_images, batch_labels)
            if np.isnan(train_loss_metric.result()):
                raise RuntimeError("NaN loss has occurred (training dataset).")
            train_bar.set_description("t. loss: {:.4f} | best: {:.4f}".format(
                                        train_loss_metric.result(), 
                                        loss_record["training_loss"]["best"]))

            if sch_ctx["switch_queue"].size() > 0:
                drain_switch_queue(sch_ctx)
                isa.set_scheduler_status(username, "---", "---", "---", isa.TRAINING)
            # steps_taken += 1


            # if check_restart():
            #     return False

        cur_training_loss = float(train_loss_metric.result())

        update_loss_record(loss_record, "training_loss", cur_training_loss)

        #json_io.save_json(loss_record_path, loss_record)

        train_loss_metric.reset_states()


        val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
        
        for batch_data in val_bar:
            batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
            conv = yolov4(batch_images, training=False)
            loss_value = loss_fn(batch_labels, conv)
            loss_value += sum(yolov4.losses)

            val_loss_metric.update_state(values=loss_value)
            if np.isnan(val_loss_metric.result()):
                raise RuntimeError("NaN loss has occurred (validation dataset).")

            val_bar.set_description("v. loss: {:.4f} | best: {:.4f}".format(
                                    val_loss_metric.result(), 
                                    loss_record["validation_loss"]["best"]))

            if sch_ctx["switch_queue"].size() > 0:
                drain_switch_queue(sch_ctx)
                isa.set_scheduler_status(username, "---", "---", "---", isa.TRAINING)

        
        cur_validation_loss = float(val_loss_metric.result())

        cur_validation_loss_is_best = update_loss_record(loss_record, "validation_loss", cur_validation_loss)
        yolov4.save_weights(filepath=cur_weights_path, save_format="h5")
        if cur_validation_loss_is_best:
            yolov4.save_weights(filepath=best_weights_path, save_format="h5")

            #model_io.save_model_weights(yolov4, config, seq_num, epoch)    


        json_io.save_json(loss_record_path, loss_record)
        
        val_loss_metric.reset_states()


        if loss_record["validation_loss"]["epochs_since_improvement"] >= VALIDATION_IMPROVEMENT_TOLERANCE:
            # shutil.copyfile(best_weights_path, cur_weights_path)
            return True

        #if isa.check_for_predictions():
        if sch_ctx["prediction_queue"].size() > 0: # or sch_ctx["restart_queue"].size() > 0:
            return False

        # if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
        #     return False

        # if os.path.exists(restart_req_path):
        #     return False

        cur_time = int(time.time())
        elapsed_train_time = cur_time - start_time
        if elapsed_train_time > TRAINING_TIME_SESSION_CEILING:
            return False


# def train(image_set_dir, sch_ctx):

#     tf.keras.backend.clear_session()
    
#     images_dir = os.path.join(image_set_dir, "images")
#     model_dir = os.path.join(image_set_dir, "model")
#     weights_dir = os.path.join(model_dir, "weights")
#     training_dir = os.path.join(model_dir, "training")

#     usr_block_path = os.path.join(training_dir, "usr_block.json")
#     sys_block_path = os.path.join(training_dir, "sys_block.json")

#     restart_req_path = os.path.join(training_dir, "restart_request.json")


#     annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
#     annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

#     training_image_names = []
#     for image_name in annotations.keys():
#         if annotations[image_name]["status"] == "completed_for_training":
#             training_image_names.append(image_name)

#     updated_patch_size = ep.get_updated_patch_size(annotations)
#     sample_training_image_path = glob.glob(os.path.join(images_dir, training_image_names[0] + "*"))[0]
#     sample_training_image = Image(sample_training_image_path)
#     image_width, image_height = sample_training_image.get_wh()

#     patch_overlap_percent = 50
#     num_patches_per_image = m.ceil(image_width / (updated_patch_size * (patch_overlap_percent / 100))) * \
#                             m.ceil(image_height / (updated_patch_size * (patch_overlap_percent / 100)))

#     num_patches = num_patches_per_image * len(training_image_names)

#     input_image_shape = config["arch"]["input_image_shape"]

#     space_consumed = num_patches * input_image_shape[0] * input_image_shape[1] * 3
#     available_bytes = psutil.virtual_memory()[1]

#     THRESH = 0.6
#     write_data = (space_consumed / available_bytes) > THRESH
#     patch_records = []
#     if write_data:

#     else:
#         for training_image_name in training_image_names:

#             image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
#             image = Image(image_path)
#             image_patch_records = ep.extract_patch_records_from_image_tiled(
#                 image, 
#                 updated_patch_size,
#                 image_annotations=annotations[image_name],
#                 patch_overlap_percent=patch_overlap_percent, 
#                 include_patch_arrays=True)

#             patch_records.extend(image_patch_records)



def drain_switch_queue(sch_ctx, cur_image_set_dir=None):
    affected = False
    switch_queue_size = sch_ctx["switch_queue"].size()
    while switch_queue_size > 0:
        item = sch_ctx["switch_queue"].dequeue()
        isa.process_switch(item)
        switch_queue_size = sch_ctx["switch_queue"].size()

        if cur_image_set_dir is not None:
            affected_image_set_dir = os.path.join("usr", "data", item["username"],
                                                  "image_sets", 
                                                  item["farm_name"], item["field_name"], item["mission_date"])
            if affected_image_set_dir == cur_image_set_dir:
                affected = True

    return affected


def train(sch_ctx, root_dir): #farm_name, field_name, mission_date):
    
    logger = logging.getLogger(__name__)

    pieces = root_dir.split("/")
    username = pieces[-5]
    farm_name = pieces[-3]
    field_name = pieces[-2]
    mission_date = pieces[-1]

    

    start_time = int(time.time())

    tf.keras.backend.clear_session()
    
    #image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(root_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    training_dir = os.path.join(model_dir, "training")

    usr_block_path = os.path.join(training_dir, "usr_block.json")
    sys_block_path = os.path.join(training_dir, "sys_block.json")



    #training_patch_dir = os.path.join(training_patches_dir, str(seq_num), "training")
    #validation_patch_dir = os.path.join(training_patches_dir, str(seq_num), "validation")

    training_record_dir = os.path.join(training_dir, "training_tf_records")
    validation_record_dir = os.path.join(training_dir, "validation_tf_records")

    training_tf_record_paths = glob.glob(os.path.join(training_record_dir, "*.tfrec")) #[os.path.join(training_dir, "training-patches-record.tfrec")]
    validation_tf_record_paths = glob.glob(os.path.join(validation_record_dir, "*.tfrec")) #[os.path.join(training_dir, "validation-patches-record.tfrec")]

    config = create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    config["training"]["active"] = {}
    for k in config["training"]:
        config["training"]["active"][k] = config["training"][k]


    # train_loader_is_preloaded, train_data_loader = data_load.get_data_loader(training_tf_record_paths, config, shuffle=True, augment=True)
    # val_loader_is_preloaded, val_data_loader = data_load.get_data_loader(validation_tf_record_paths, config, shuffle=False, augment=False)
    

    # logger.info("Data loaders created. Train loader is preloaded?: {}. Validation loader is preloaded?: {}.".format(
    #     train_loader_is_preloaded, val_loader_is_preloaded
    # ))

    train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
    val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

    train_dataset, num_train_patches = train_data_loader.create_batched_dataset(
                                      take_percent=config["training"]["active"]["percent_of_training_set_used"])
    # if check_restart():
    #     return False
    val_dataset, num_val_patches = val_data_loader.create_batched_dataset(
                                  take_percent=config["training"]["active"]["percent_of_validation_set_used"])


    logger.info("Building model...")


    if config["arch"]["model_type"] == "yolov4":
        yolov4 = YOLOv4(config)
    elif config["arch"]["model_type"] == "yolov4_tiny":
        yolov4 = YOLOv4Tiny(config)

    loss_fn = YOLOv4Loss(config)


    # TODO: change model input shape from static to get_patch_size() rounded to multiple of 32 
    # (with reasonable upper and lower limits)

    input_shape = (config["training"]["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
    yolov4.build(input_shape=input_shape)

    logger.info("Model built.")


    cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")
    best_weights_path = os.path.join(weights_dir, "best_weights.h5")
    # if os.path.exists(cur_weights_path):

    # else:



    cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")
    if not os.path.exists(cur_weights_path):
        raise RuntimeError("Model weights could not be located.")
    # if not os.path.exists(cur_weights_path):
    #     logger.info("No fine-tuned weights found .. checking status file.")
    #     status_path = os.path.join(model_dir, "status.json")
    #     status = json_io.load_json(status_path)
    #     model_name = status["model_name"]
    #     model_creator = status["model_creator"]
    #     model_path = os.path.join("usr", "data", model_creator, "models")
    #     public_weights_path = os.path.join(model_path, "available", "public", model_name, "weights.h5")
    #     private_weights_path = os.path.join(model_path, "available", "private", model_name, "weights.h5")

    #     if os.path.exists(public_weights_path):
    #         shutil.move(public_weights_path, cur_weights_path)
    #         shutil.move(public_weights_path, best_weights_path)
    #     elif os.path.exists(private_weights_path):
    #         shutil.move(private_weights_path, cur_weights_path)
    #         shutil.move(private_weights_path, best_weights_path)
    #     else:
    #         raise RuntimeError("Model weights could not be located.")

    logger.info("Loading weights...")
    yolov4.load_weights(cur_weights_path, by_name=False)
    logger.info("Weights loaded.")

    optimizer = tf.optimizers.Adam()


    train_loss_metric = tf.metrics.Mean()
    val_loss_metric = tf.metrics.Mean()

    @tf.function
    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            conv = yolov4(batch_images, training=True)
            loss_value = loss_fn(batch_labels, conv)
            loss_value += sum(yolov4.losses)

        #if np.isnan(loss_value):
        #    raise RuntimeError("NaN loss has occurred.")
        gradients = tape.gradient(target=loss_value, sources=yolov4.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, yolov4.trainable_variables))
        train_loss_metric.update_state(values=loss_value)




    # steps_taken = 0

    while True:
        train_steps_per_epoch = np.sum([1 for _ in train_dataset])
        val_steps_per_epoch = np.sum([1 for _ in val_dataset])

        logger.info("{} ('{}'): Starting to train with {} training patches and {} validation patches.".format(
                        config["arch"]["model_type"], config["model_name"], num_train_patches, num_val_patches))


        loss_record_path = os.path.join(training_dir, "loss_record.json")
        loss_record = json_io.load_json(loss_record_path)

        logger.info("Epochs since validation loss improvement: {}".format(loss_record["validation_loss"]["epochs_since_improvement"]))

        train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
        for batch_data in train_bar:

            # optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))
            optimizer.lr.assign(config["training"]["active"]["learning_rate_schedule"]["learning_rate"])

            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)

            train_step(batch_images, batch_labels)
            if np.isnan(train_loss_metric.result()):
                raise RuntimeError("NaN loss has occurred (training dataset).")
            train_bar.set_description("t. loss: {:.4f} | best: {:.4f}".format(
                                        train_loss_metric.result(), 
                                        loss_record["training_loss"]["best"]))
            # steps_taken += 1


            # if check_restart():
            #     return False

            if sch_ctx["switch_queue"].size() > 0:
                affected = drain_switch_queue(sch_ctx, cur_image_set_dir=root_dir)
                if affected:
                    return (False, False)
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINE_TUNING)


        cur_training_loss = float(train_loss_metric.result())

        update_loss_record(loss_record, "training_loss", cur_training_loss)

        #json_io.save_json(loss_record_path, loss_record)

        train_loss_metric.reset_states()


        val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
        
        for batch_data in val_bar:
            batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
            conv = yolov4(batch_images, training=False)
            loss_value = loss_fn(batch_labels, conv)
            loss_value += sum(yolov4.losses)

            val_loss_metric.update_state(values=loss_value)
            if np.isnan(val_loss_metric.result()):
                raise RuntimeError("NaN loss has occurred (validation dataset).")

            val_bar.set_description("v. loss: {:.4f} | best: {:.4f}".format(
                                    val_loss_metric.result(), 
                                    loss_record["validation_loss"]["best"]))

            if sch_ctx["switch_queue"].size() > 0:
                affected = drain_switch_queue(sch_ctx, cur_image_set_dir=root_dir)
                if affected:
                    return (False, False)
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINE_TUNING)

        
        cur_validation_loss = float(val_loss_metric.result())

        cur_validation_loss_is_best = update_loss_record(loss_record, "validation_loss", cur_validation_loss)
        yolov4.save_weights(filepath=cur_weights_path, save_format="h5")
        if cur_validation_loss_is_best:
            yolov4.save_weights(filepath=best_weights_path, save_format="h5")

            #model_io.save_model_weights(yolov4, config, seq_num, epoch)    


        json_io.save_json(loss_record_path, loss_record)
        
        val_loss_metric.reset_states()


        # annotations_read_time = int(time.time())

        annotations_path = os.path.join(root_dir, "annotations", "annotations_w3c.json")
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
        training_image_names = []
        for image_name in annotations.keys():
            if annotations[image_name]["status"] == "completed_for_training":
                training_image_names.append(image_name)

        # updated_patch_size = ep.get_updated_patch_size(annotations, training_image_names)
        updated_patch_size = ep.update_model_patch_size(annotations, training_image_names, root_dir)
        changed_training_image_names = ep.update_patches(root_dir, annotations, training_image_names, updated_patch_size)
        if len(changed_training_image_names) > 0:

            image_set_aux.update_training_tf_records(root_dir, changed_training_image_names, annotations)
            image_set_aux.reset_loss_record(root_dir)

            training_tf_record_paths = glob.glob(os.path.join(training_record_dir, "*.tfrec"))
            validation_tf_record_paths = glob.glob(os.path.join(validation_record_dir, "*.tfrec"))

            train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
            val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

            train_dataset, num_train_patches = train_data_loader.create_batched_dataset(
                                            take_percent=config["training"]["active"]["percent_of_training_set_used"])

            val_dataset, num_val_patches = val_data_loader.create_batched_dataset(
                                        take_percent=config["training"]["active"]["percent_of_validation_set_used"])




        if loss_record["validation_loss"]["epochs_since_improvement"] >= VALIDATION_IMPROVEMENT_TOLERANCE:
            shutil.copyfile(best_weights_path, cur_weights_path)
            return (True, False)

        #if isa.check_for_predictions():
        if sch_ctx["prediction_queue"].size() > 0: # or sch_ctx["switch_queue"].size() > 0:
            return (False, True)

        if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
            return (False, False)

        # if os.path.exists(restart_req_path):
        #     return False

        cur_time = int(time.time())
        elapsed_train_time = cur_time - start_time
        if elapsed_train_time > TRAINING_TIME_SESSION_CEILING:
            return (False, True)





def output_patch(patch, gt_boxes, pred_boxes, pred_classes, pred_scores, out_path):
    from models.common import model_vis

    out_array = model_vis.draw_boxes_on_image(patch,
                      pred_boxes,
                      pred_classes,
                      pred_scores,
                      class_map={"plant": 0},
                      gt_boxes=gt_boxes, #None,
                      patch_coords=None,
                      display_class=False,
                      display_score=False)
    cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))