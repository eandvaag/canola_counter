import tensorflow as tf
import numpy as np
import time
import sys
import os
import shutil
import tqdm
import logging
import time
import random

from image_set import DataSet

from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          inference_record_io

from models.yolov4.loss import YOLOv4Loss
from models.yolov4.yolov4 import YOLOv4, YOLOv4Tiny
import models.yolov4.data_load as data_load
from models.yolov4.encode import Decoder


from io_utils import json_io, tf_record_io, w3c_io



# def post_process_sample(detections, resize_ratios, index):

#     num_detections = detections.valid_detections[index]
#     classes = detections.nmsed_classes[index][:num_detections]
#     scores = detections.nmsed_scores[index][:num_detections]
#     boxes = detections.nmsed_boxes[index][:num_detections]

#     boxes = boxes.numpy()
#     scores = scores.numpy()
#     classes = classes.numpy()

#     boxes = box_utils.swap_xy_np(boxes)
#     boxes = np.stack([
#         boxes[:, 0] * resize_ratios[index][0],
#         boxes[:, 1] * resize_ratios[index][1],
#         boxes[:, 2] * resize_ratios[index][0],
#         boxes[:, 3] * resize_ratios[index][1]
#     ], axis=-1)

#     boxes = np.rint(boxes).astype(np.int32)
#     scores = scores.astype(np.float32)
#     classes = classes.astype(np.int32)

#     return boxes, scores, classes

def post_process_sample(detections, resize_ratio, patch_coords, config, apply_nms=True):

    detections = np.array(detections)

    pred_xywh = detections[:, 0:4]
    pred_conf = detections[:, 4]
    pred_prob = detections[:, 5:]

    pred_boxes = (box_utils.swap_xy_tf(box_utils.convert_to_corners_tf(pred_xywh))).numpy()

    pred_boxes = np.stack([
            pred_boxes[:, 0] * resize_ratio[0],
            pred_boxes[:, 1] * resize_ratio[1],
            pred_boxes[:, 2] * resize_ratio[0],
            pred_boxes[:, 3] * resize_ratio[1]
    ], axis=-1)


    invalid_mask = np.logical_or((pred_boxes[:, 0] > pred_boxes[:, 2]), (pred_boxes[:, 1] > pred_boxes[:, 3]))
    pred_boxes[invalid_mask] = 0

    # # (4) discard some invalid boxes
    #bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    #scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))


    score_threshold = 0.5
    pred_classes = np.argmax(pred_prob, axis=-1)
    pred_scores = pred_conf * pred_prob[np.arange(len(pred_boxes)), pred_classes]
    score_mask = pred_scores > score_threshold
    #mask = np.logical_and(scale_mask, score_mask)
    mask = score_mask
    pred_boxes, pred_scores, pred_classes = pred_boxes[mask], pred_scores[mask], pred_classes[mask]

    #print("coors: ", coors)
    #print("scores: ", scores)
    #print("classes", classes)
    pred_boxes = np.rint(pred_boxes).astype(np.int32)
    pred_scores = pred_scores.astype(np.float32)
    pred_classes = pred_classes.astype(np.int32)

    if apply_nms:
        pred_boxes, pred_classes, pred_scores = box_utils.non_max_suppression_with_classes(
            pred_boxes,
            pred_classes,
            pred_scores,
            iou_thresh=config.inference["patch_nms_iou_thresh"])


    # # TODO: need to fix to not do this when patches are on the edge of the image
    # if "patch_border_buffer" in config.inference["active"]:
    #     patch_wh = (patch_coords[2] - patch_coords[1])
    #     pct_buf = config.inference["active"]["patch_border_buffer_percent"]
    #     pix_buf = (pct_buf / 100 ) * patch_wh
                    

    #     #mask = np.logical_or(
    #     #            pred_boxes[:, 0] < patch_coords[0] + buf,
    #     #            pred_boxes[:, 1] < patch_coords[1] + buf,
    #     #            pred_boxes[:, 2] > patch_coords[2] - buf,
    #     #            pred_boxes[:, 3] > patch_coords[3] - buf)

    #     mask = np.logical_and(
    #                 pred_boxes[:, 0] > patch_coords[0] + buf,
    #                 pred_boxes[:, 1] > patch_coords[1] + buf,
    #                 pred_boxes[:, 2] < patch_coords[2] - buf,
    #                 pred_boxes[:, 3] < patch_coords[3] - buf)        
    #     pred_boxes = pred_boxes[mask]
    #     pred_scores = pred_scores[mask]
    #     pred_classes = pred_classes[mask]

    return pred_boxes, pred_scores, pred_classes



def find_optimal_iou(img_set, dataset, config):

    logger = logging.getLogger(__name__)
    
    driver_utils.set_active_inference_params(config, 0)
    dataset_name = "training"
    patch_dir, _ = driver_utils.create_patches(config.inference["active"]["inference_patch_extraction_params"], 
                                                        img_set, dataset_name)

    tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

    data_loader = data_load.InferenceDataLoader(tf_record_path, config)
    tf_dataset, tf_dataset_size = data_loader.create_dataset()

    if config.arch["model_type"] == "yolov4":
        yolov4 = YOLOv4(config)
    elif config.arch["model_type"] == "yolov4_tiny":
        yolov4 = YOLOv4Tiny(config)
    decoder = Decoder(config)

    input_shape = (config.inference["active"]["batch_size"], *(data_loader.get_model_input_shape()))
    # *(config.arch["input_img_shape"]))
    yolov4.build(input_shape=input_shape)

    model_io.load_all_weights(yolov4, config)

    predictions = {"image_predictions": {}, "patch_predictions": {}}
    steps = np.sum([1 for i in tf_dataset])


    logger.info("{} ('{}'): Running inference on {} images.".format(config.arch["model_type"], 
                                                                    config.arch["model_name"], 
                                                                    tf_dataset_size))

    inference_times = []
    for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

        batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, dataset.is_annotated)
        batch_size = batch_images.shape[0]


        start_inference_time = time.time()
        pred = yolov4(batch_images, training=False)
        detections = decoder(pred)
        #print("detections shape", [tf.shape(x) for x in detections])
        pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]
        #print("(1) pred_bbox shape", [tf.shape(x) for x in pred_bbox])

        pred_bbox = tf.concat(pred_bbox, axis=1)

        #print("(2) pred_bbox shape", tf.shape(pred_bbox))
        #detections = decoder(batch_images, pred)
        end_inference_time = time.time()

        inference_times.append(end_inference_time - start_inference_time)

        for i in range(batch_size):

            pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = post_process_sample(
                pred_bbox, batch_ratios, i, config, apply_nms=False)

            patch_info = batch_info[i]

            img_path = bytes.decode((patch_info["img_path"]).numpy())
            patch_path = bytes.decode((patch_info["patch_path"]).numpy())
            img_name = os.path.basename(img_path)[:-4]
            patch_name = os.path.basename(patch_path)[:-4]
            patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

            if dataset.is_annotated:
                patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()

            if pred_patch_abs_boxes.size == 0:
                pred_img_abs_boxes = np.array([], dtype=np.int32)
            else:
                pred_img_abs_boxes = (np.array(pred_patch_abs_boxes) + \
                                      np.tile(patch_coords[:2], 2)).astype(np.int32)

            predictions["patch_predictions"][patch_name] = {
                "img_name": img_name,
                "img_path": img_path,
                "patch_coords": patch_coords.tolist(),
                "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
                "pred_scores": pred_patch_scores.tolist(),
                "pred_classes": pred_patch_classes.tolist()
            }
            #if dataset.is_annotated:
            predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
            predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes


    best_thresh = search_for_iou(predictions, config)
    #predictions[""]

def search_for_iou(predictions, config):
    thresh_vals = np.arange(0, 1.05, 0.05)

    best = float("inf")
    best_thresh = 0.5



    for thresh_val in thresh_vals:
        annotated_counts = []
        pred_counts = []
        for patch_name, patch_data in predictions["patch_predictions"].items():
            #print(patch_data["pred_patch_abs_boxes"])
            #print(patch_data["pred_classes"])
            boxes = patch_data["pred_patch_abs_boxes"]
            classes = patch_data["pred_classes"]
            scores = patch_data["pred_scores"]

            if len(classes) > 0:
                boxes, classes, scores = box_utils.non_max_suppression_with_classes(
                    np.array(boxes),
                    np.array(classes),
                    np.array(scores),
                    iou_thresh=float(thresh_val))

            annotated_counts.append(len(patch_data["patch_classes"]))
            pred_counts.append(len(classes))

        mean_abs_diff = inference_metrics.mean_abs_DiC(np.array(annotated_counts), np.array(pred_counts))
        print("{}: mean_abs_diff: {}".format(thresh_val, mean_abs_diff))
        if mean_abs_diff < best:
            best = mean_abs_diff
            best_thresh = thresh_val
    print("---")
    print("Best thresh is {}".format(best_thresh))
    return best_thresh


# def generate_predictions(config):

#     logger = logging.getLogger(__name__)

#     if config.inference["shared_default"]["patch_nms_iou_thresh"] == "find_optimal":
#         img_set_conf = config.inference["image_sets"][0]
#         img_set = ImgSet(img_set_conf["farm_name"], img_set_conf["field_name"], 
#                          img_set_conf["mission_date"])
#         find_optimal_iou(img_set, img_set.datasets["training"], config)
#         exit()
    

#     dataset_index = 0
#     for img_set_num, img_set_conf in enumerate(config.inference["image_sets"]):

#         farm_name = img_set_conf["farm_name"]
#         field_name = img_set_conf["field_name"]
#         mission_date = img_set_conf["mission_date"] 
#         img_set = ImgSet(farm_name, field_name, mission_date)

#         driver_utils.set_active_inference_params(config, img_set_num)

#         for dataset_name in img_set_conf["datasets"]:

#             dataset = img_set.datasets[dataset_name]

#             patch_dir, _ = driver_utils.create_patches(img_set_conf["inference_patch_extraction_params"], 
#                                                         img_set, dataset_name)

#             tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

#             data_loader = data_load.InferenceDataLoader(tf_record_path, config)
#             tf_dataset, tf_dataset_size = data_loader.create_dataset()

#             if config.arch["model_type"] == "yolov4":
#                 yolov4 = YOLOv4(config)
#             elif config.arch["model_type"] == "yolov4_tiny":
#                 yolov4 = YOLOv4Tiny(config)
#             decoder = Decoder(config)

#             input_shape = (config.inference["active"]["batch_size"], *(data_loader.get_model_input_shape()))
#             # *(config.arch["input_img_shape"]))
#             yolov4.build(input_shape=input_shape)

#             model_io.load_all_weights(yolov4, config)

#             predictions = driver_utils.create_predictions_skeleton(img_set, dataset)


#             if "patch_border_buffer_percent" in config.inference["active"]:
#                 buffer_pct = config.inference["active"]["patch_border_buffer_percent"]
#             else:
#                 buffer_pct = None

#             steps = np.sum([1 for i in tf_dataset])


#             logger.info("{} ('{}'): Running inference on {} images.".format(config.arch["model_type"], 
#                                                                             config.arch["model_name"], 
#                                                                             tf_dataset_size))

#             inference_times = []
#             for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

#                 batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, dataset.is_annotated)
#                 batch_size = batch_images.shape[0]


#                 start_inference_time = time.time()
#                 pred = yolov4(batch_images, training=False)
#                 detections = decoder(pred)
#                 #print("detections shape", [tf.shape(x) for x in detections])
#                 batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]
#                 #print("(1) pred_bbox shape", [tf.shape(x) for x in pred_bbox])

#                 batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

#                 #print("(2) pred_bbox shape", tf.shape(pred_bbox))
#                 #detections = decoder(batch_images, pred)
#                 end_inference_time = time.time()

#                 inference_times.append(end_inference_time - start_inference_time)

#                 for i in range(batch_size):

#                     pred_bbox = batch_pred_bbox[i]
#                     ratio = batch_ratios[i]

#                     patch_info = batch_info[i]

#                     img_path = bytes.decode((patch_info["img_path"]).numpy())
#                     patch_path = bytes.decode((patch_info["patch_path"]).numpy())
#                     img_name = os.path.basename(img_path)[:-4]
#                     patch_name = os.path.basename(patch_path)[:-4]
#                     patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

#                     if dataset.is_annotated:
#                         patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
#                         patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()


#                     pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
#                             post_process_sample(pred_bbox, ratio, patch_coords, config)




#                     # if pred_patch_abs_boxes.size == 0:
#                     #     pred_img_abs_boxes = np.array([], dtype=np.int32)
#                     # else:
#                     #     pred_img_abs_boxes = (np.array(pred_patch_abs_boxes) + \
#                     #                           np.tile(patch_coords[:2], 2)).astype(np.int32)

#                     predictions["patch_predictions"][patch_name] = {
#                         "img_name": img_name,
#                         "img_path": img_path,
#                         "patch_coords": patch_coords.tolist(),
#                         "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
#                         "pred_scores": pred_patch_scores.tolist(),
#                         "pred_classes": pred_patch_classes.tolist()
#                     }
#                     if dataset.is_annotated:
#                         predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
#                         predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

#                     if img_name not in predictions["image_predictions"]:
#                         predictions["image_predictions"][img_name] = {
#                             "img_path": img_path,
#                             "pred_img_abs_boxes": [],
#                             "pred_classes": [],
#                             "pred_scores": [],
#                             "patch_coords": []
#                         }

#                     pred_img_abs_boxes, pred_img_scores, pred_img_classes = \
#                         driver_utils.get_img_detections(pred_patch_abs_boxes, 
#                                                         pred_patch_scores, 
#                                                         pred_patch_classes, 
#                                                         patch_coords, 
#                                                         img_path, 
#                                                         buffer_pct=buffer_pct)


#                     predictions["image_predictions"][img_name]["pred_img_abs_boxes"].extend(pred_img_abs_boxes.tolist())
#                     predictions["image_predictions"][img_name]["pred_scores"].extend(pred_img_scores.tolist())
#                     predictions["image_predictions"][img_name]["pred_classes"].extend(pred_img_classes.tolist())
#                     predictions["image_predictions"][img_name]["patch_coords"].append(patch_coords.tolist())


#             driver_utils.clip_img_boxes(predictions["image_predictions"])

#             driver_utils.apply_nms_to_img_boxes(predictions["image_predictions"], 
#                                                 iou_thresh=config.inference["active"]["image_nms_iou_thresh"])
#             driver_utils.add_class_detections(predictions["image_predictions"], img_set)

#             inference_metrics.collect_statistics(predictions, img_set, dataset,
#                                                  inference_times=inference_times)
#             if dataset.is_annotated:
#                 inference_metrics.collect_metrics(predictions, img_set, dataset)

#             pred_dirname = os.path.basename(patch_dir)
#             pred_dir = os.path.join(config.model_dir, "predictions", pred_dirname)
#             os.makedirs(pred_dir)
#             pred_path = os.path.join(pred_dir, "predictions.json")
#             json_io.save_json(pred_path, predictions)

#             excel_path = os.path.join(pred_dir, "results.xlsx")
#             driver_utils.output_excel(excel_path, predictions, img_set, dataset_name)

#             inference_entry = {
#                 "farm_name": farm_name,
#                 "field_name": field_name,
#                 "mission_date": mission_date,
#                 "dataset_name": dataset_name,
#                 "model_uuid": config.arch["model_uuid"],
#                 "value": {
#                     "job_uuid": config.arch["job_uuid"],
#                     "job_name": config.arch["job_name"],
#                     "model_uuid": config.arch["model_uuid"],
#                     "model_name": config.arch["model_name"],
#                     "prediction_dirname": pred_dirname,
#                     "metrics": predictions["metrics"]
#                 }
#             }
#             inference_record_io.add_entry_to_inference_record(inference_entry)

#             dataset_index += 1



# def train_dep(config):

#     tf.keras.backend.clear_session()

#     logger = logging.getLogger(__name__)



#     for seq_num in range(len(config.training["training_sequence"])):

#         training_tf_record_paths = []
#         validation_tf_record_paths = []
#         training_tf_record_paths_obj = []
#         training_tf_record_paths_bg = []
#         validation_tf_record_paths_obj = []
#         validation_tf_record_paths_bg = []

#         driver_utils.set_active_training_params(config, seq_num)

#         for img_set_conf in config.training["training_sequence"][seq_num]["image_sets"]:

            
#             img_set = ImgSet(img_set_conf["farm_name"], img_set_conf["field_name"], img_set_conf["mission_date"])
#             training_patch_dir, _ = driver_utils.create_patches(
#                 img_set_conf["training_patch_extraction_params"], img_set, "training")
#             validation_patch_dir, _ = driver_utils.create_patches(
#                 img_set_conf["validation_patch_extraction_params"], img_set, "validation")

#             training_tf_record_paths_obj.append(os.path.join(training_patch_dir, "patches-with-boxes-record.tfrec"))
#             validation_tf_record_paths_obj.append(os.path.join(validation_patch_dir, "patches-with-boxes-record.tfrec"))
#             training_tf_record_paths_bg.append(os.path.join(training_patch_dir, "patches-with-no-boxes-record.tfrec"))
#             validation_tf_record_paths_bg.append(os.path.join(validation_patch_dir, "patches-with-no-boxes-record.tfrec"))
#             training_tf_record_paths.append(os.path.join(training_patch_dir, "patches-record.tfrec"))
#             validation_tf_record_paths.append(os.path.join(validation_patch_dir, "patches-record.tfrec"))

#         data_loader_type = config.training["active"]["data_loader"]["type"]
#         if data_loader_type == "default":
#             train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
#             val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)
#         elif data_loader_type == "split":
#             train_data_loader = data_load.SplitDataLoader(training_tf_record_paths_obj, training_tf_record_paths_bg,
#                                                           config, shuffle=True, augment=True)
#             val_data_loader = data_load.SplitDataLoader(validation_tf_record_paths_obj, validation_tf_record_paths_bg,
#                                                           config, shuffle=False, augment=False)
#         else:
#             raise RuntimeError("Unrecognized data loader type: {}".format(data_loader_type))


#         train_dataset, num_train_images = train_data_loader.create_batched_dataset(
#                                                 take_percent=config.training["active"]["percent_of_training_set_used"])

#         val_dataset, num_val_images = val_data_loader.create_batched_dataset(
#                                                 take_percent=config.training["active"]["percent_of_validation_set_used"])

#         if config.arch["model_type"] == "yolov4":
#             yolov4 = YOLOv4(config)
#         elif config.arch["model_type"] == "yolov4_tiny":
#             yolov4 = YOLOv4Tiny(config)

#         loss_fn = YOLOv4Loss(config)


#         input_shape = (config.training["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
#         yolov4.build(input_shape=input_shape)


#         if seq_num == 0:
#             layer_lookup = yolov4.get_layer_lookup()
#             layer_lookup_path = os.path.join(config.weights_dir, "layer_lookup.json")
#             json_io.save_json(layer_lookup_path, layer_lookup)

#         else:
#             model_io.load_select_weights(yolov4, config)

#         optimizer = tf.optimizers.Adam()


#         train_loss_metric = tf.metrics.Mean()
#         val_loss_metric = tf.metrics.Mean()

#         #@tf.function
#         def train_step(batch_images, batch_labels):
#             with tf.GradientTape() as tape:
#                 conv = yolov4(batch_images, training=True)
#                 loss_value = loss_fn(batch_labels, conv)

#             if np.isnan(loss_value):
#                 raise RuntimeError("NaN loss has occurred.")
#             gradients = tape.gradient(target=loss_value, sources=yolov4.trainable_variables)
#             optimizer.apply_gradients(grads_and_vars=zip(gradients, yolov4.trainable_variables))
#             train_loss_metric.update_state(values=loss_value)



#         train_steps_per_epoch = np.sum([1 for i in train_dataset])
#         val_steps_per_epoch = np.sum([1 for i in val_dataset])
#         best_val_loss = float("inf")
#         epochs_since_improvement = 0
#         loss_record = {
#             "training_loss": { "values": [],
#                                "best": {"epoch": -1, "value": float("inf")},
#                                "epochs_since_improvement": 0},
#             "validation_loss": {"values": [],
#                                 "best": {"epoch": -1, "value": float("inf")},
#                                 "epochs_since_improvement": 0}
#         }

#         logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
#                      config.arch["model_type"], config.arch["model_name"], num_train_images, num_val_images))


#         max_num_epochs = config.training["active"]["max_num_epochs"]
#         steps_taken = 0
#         #with tf.profiler.experimental.Profile('logdir'):
#         for epoch in range(max_num_epochs):


#             train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
#             for batch_data in train_bar:
#             #for step in tqdm.trange(train_steps_per_epoch):

#                 #with tf.profiler.experimental.Trace('train', step_num=steps_taken, _r=1):

#                 #batch_data = next(iter(train_dataset))

#                 optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))

#                 batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)

#                 train_step(batch_images, batch_labels)
#                 train_bar.set_description("Epoch: {}/{} | t. loss: {:.4f} | best: {:.4f} (ep. {})".format(
#                                           epoch, max_num_epochs-1, train_loss_metric.result(), 
#                                           loss_record["training_loss"]["best"]["value"],
#                                           loss_record["training_loss"]["best"]["epoch"]))
#                 steps_taken += 1


#             val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
            
#             for batch_data in val_bar:
#                 batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
#                 conv = yolov4(batch_images, training=False)
#                 loss_value = loss_fn(batch_labels, conv)
#                 if np.isnan(loss_value):
#                     raise RuntimeError("NaN loss has occurred.")

#                 val_loss_metric.update_state(values=loss_value)

#                 val_bar.set_description("Epoch: {}/{} | v. loss: {:.4f} | best: {:.4f} (ep. {})".format(
#                                         epoch, max_num_epochs-1, val_loss_metric.result(), 
#                                         loss_record["validation_loss"]["best"]["value"],
#                                         loss_record["validation_loss"]["best"]["epoch"]))

#             cur_training_loss = float(train_loss_metric.result())
#             cur_validation_loss = float(val_loss_metric.result())

#             cur_training_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "training_loss", cur_training_loss, epoch)
#             if cur_training_loss_is_best and config.training["active"]["save_method"] == "best_training_loss":
#                 model_io.save_model_weights(yolov4, config, seq_num, epoch)

#             cur_validation_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "validation_loss", cur_validation_loss, epoch)
#             if cur_validation_loss_is_best and config.training["active"]["save_method"] == "best_validation_loss":
#                 model_io.save_model_weights(yolov4, config, seq_num, epoch)    

#             if driver_utils.stop_early(config, loss_record):
#                 break


#             train_loss_metric.reset_states()
#             val_loss_metric.reset_states()


#         loss_record_path = os.path.join(config.loss_records_dir, str(seq_num) + ".json")
#         json_io.save_json(loss_record_path, loss_record)




def generate_predictions(config):

    logger = logging.getLogger(__name__)

    # patches_dir = os.path.join(config.model_dir, "patches")
    # if not os.path.exists(patches_dir):
    #     os.makedirs(patches_dir)




    #data_loader = data_load.InferenceDataLoader(tf_record_path, config)
    target_patches_dir = os.path.join(config.model_dir, "target_patches") #config.inference["inference_patch_dir"]

    for dataset_conf in config.inference["datasets"]:

        #driver_utils.set_active_inference_params(config)
        #image_set = ImageSet(image_set_conf)
        dataset = DataSet(dataset_conf)

        if config.arch["model_type"] == "yolov4":
            yolov4 = YOLOv4(config)
        elif config.arch["model_type"] == "yolov4_tiny":
            yolov4 = YOLOv4Tiny(config)
        decoder = Decoder(config)

        #patch_dir = driver_utils.extract_patches(image_set.all_dataset, config)
        tf_record_names = ["annotated-patches-record.tfrec", "unannotated-patches-record.tfrec"]

        predictions = driver_utils.create_predictions_skeleton(dataset)


        for k, tf_record_name in enumerate(tf_record_names):
            is_annotated = k == 0
            tf_record_path = os.path.join(target_patches_dir, tf_record_name)
            data_loader = data_load.InferenceDataLoader(tf_record_path, config)
            tf_dataset, tf_dataset_size = data_loader.create_dataset()

            input_shape = (config.inference["batch_size"], *(data_loader.get_model_input_shape()))
            yolov4.build(input_shape=input_shape)
            model_io.load_all_weights(yolov4, config)

            if "patch_border_buffer_percent" in config.inference:
                buffer_pct = config.inference["patch_border_buffer_percent"]
            else:
                buffer_pct = None


            steps = np.sum([1 for i in tf_dataset])

            logger.info("{} ('{}'): Running inference on {} images.".format(config.arch["model_type"], 
                                                                            config.arch["model_name"], 
                                                                            tf_dataset_size))

            inference_times = []
            for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

                batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, is_annotated)
                batch_size = batch_images.shape[0]


                start_inference_time = time.time()
                pred = yolov4(batch_images, training=False)
                detections = decoder(pred)
                #print("detections shape", [tf.shape(x) for x in detections])
                batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]
                #print("(1) pred_bbox shape", [tf.shape(x) for x in pred_bbox])

                batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

                #print("(2) pred_bbox shape", tf.shape(pred_bbox))
                #detections = decoder(batch_images, pred)
                end_inference_time = time.time()

                inference_times.append(end_inference_time - start_inference_time)

                for i in range(batch_size):

                    pred_bbox = batch_pred_bbox[i]
                    ratio = batch_ratios[i]

                    patch_info = batch_info[i]

                    image_path = bytes.decode((patch_info["image_path"]).numpy())
                    patch_path = bytes.decode((patch_info["patch_path"]).numpy())
                    image_name = os.path.basename(image_path)[:-4]
                    patch_name = os.path.basename(patch_path)[:-4]
                    patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

                    #if dataset.is_annotated:
                    if is_annotated:
                        patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                        patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()


                    pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
                            post_process_sample(pred_bbox, ratio, patch_coords, config)


                    predictions["patch_predictions"][patch_name] = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "patch_coords": patch_coords.tolist(),
                        "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
                        "pred_scores": pred_patch_scores.tolist(),
                        "pred_classes": pred_patch_classes.tolist()
                    }
                    if is_annotated:
                        predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
                        predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

                    if image_name not in predictions["image_predictions"]:
                        predictions["image_predictions"][image_name] = {
                            "image_path": image_path,
                            "pred_image_abs_boxes": [],
                            "pred_classes": [],
                            "pred_scores": [],
                            "patch_coords": []
                        }

                    pred_image_abs_boxes, pred_image_scores, pred_image_classes = \
                        driver_utils.get_image_detections(pred_patch_abs_boxes, 
                                                        pred_patch_scores, 
                                                        pred_patch_classes, 
                                                        patch_coords, 
                                                        image_path, 
                                                        buffer_pct=buffer_pct)


                    predictions["image_predictions"][image_name]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
                    predictions["image_predictions"][image_name]["pred_scores"].extend(pred_image_scores.tolist())
                    predictions["image_predictions"][image_name]["pred_classes"].extend(pred_image_classes.tolist())
                    predictions["image_predictions"][image_name]["patch_coords"].append(patch_coords.tolist())


        driver_utils.clip_image_boxes(predictions["image_predictions"])

        driver_utils.apply_nms_to_image_boxes(predictions["image_predictions"], 
                                            iou_thresh=config.inference["image_nms_iou_thresh"])
        driver_utils.add_class_detections(predictions["image_predictions"], config)

        inference_metrics.collect_statistics(predictions, dataset, config, inference_times=inference_times)

        inference_metrics.collect_metrics(predictions, dataset, config)

        results_dir = os.path.join("usr", "data", "results",
                                   dataset.farm_name, dataset.field_name, dataset.mission_date,
                                   config.arch["job_uuid"],
                                   config.arch["model_uuid"])
        os.makedirs(results_dir, exist_ok=True)

        #pred_dirname = os.path.basename(patch_dir)
        #pred_dir = os.path.join(config.model_dir, "predictions", pred_dirname)
        #os.makedirs(pred_dir)
        pred_path = os.path.join(results_dir, "predictions.json")
        json_io.save_json(pred_path, predictions)

        annotations_path = os.path.join(results_dir, "annotations.json")
        w3c_io.save_annotations(annotations_path, predictions, config)

        excel_path = os.path.join(results_dir, "results.xlsx")
        driver_utils.output_excel(excel_path, predictions, dataset, config)

        # inference_entry = {
        #     "farm_name": image_set.farm_name,
        #     "field_name": image_set.field_name,
        #     "mission_date": image_set.mission_date,
        #     #"dataset_name": dataset_name,
        #     "model_uuid": config.arch["model_uuid"],
        #     "value": {
        #         "job_uuid": config.arch["job_uuid"],
        #         "job_name": config.arch["job_name"],
        #         "model_uuid": config.arch["model_uuid"],
        #         "model_name": config.arch["model_name"],
        #         "prediction_dirname": pred_dirname,
        #         "metrics": predictions["metrics"]
        #     }
        # }
        # inference_record_io.add_entry_to_inference_record(inference_entry)

        # for dataset_name in img_set_conf["datasets"]:

        #     dataset = img_set.datasets[dataset_name]

        #     patch_dir, _ = driver_utils.create_patches(img_set_conf["inference_patch_extraction_params"], 
        #                                                 img_set, dataset_name)

        #     tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

        #     data_loader = data_load.InferenceDataLoader(tf_record_path, config)
        #     tf_dataset, tf_dataset_size = data_loader.create_dataset()

        #     if config.arch["model_type"] == "yolov4":
        #         yolov4 = YOLOv4(config)
        #     elif config.arch["model_type"] == "yolov4_tiny":
        #         yolov4 = YOLOv4Tiny(config)
        #     decoder = Decoder(config)

        #     input_shape = (config.inference["active"]["batch_size"], *(data_loader.get_model_input_shape()))
        #     # *(config.arch["input_img_shape"]))
        #     yolov4.build(input_shape=input_shape)

        #     model_io.load_all_weights(yolov4, config)

        #     predictions = driver_utils.create_predictions_skeleton(img_set, dataset)


        #     if "patch_border_buffer_percent" in config.inference["active"]:
        #         buffer_pct = config.inference["active"]["patch_border_buffer_percent"]
        #     else:
        #         buffer_pct = None

        #     steps = np.sum([1 for i in tf_dataset])


        #     logger.info("{} ('{}'): Running inference on {} images.".format(config.arch["model_type"], 
        #                                                                     config.arch["model_name"], 
        #                                                                     tf_dataset_size))

        #     inference_times = []
        #     for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

        #         batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, dataset.is_annotated)
        #         batch_size = batch_images.shape[0]


        #         start_inference_time = time.time()
        #         pred = yolov4(batch_images, training=False)
        #         detections = decoder(pred)
        #         #print("detections shape", [tf.shape(x) for x in detections])
        #         batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]
        #         #print("(1) pred_bbox shape", [tf.shape(x) for x in pred_bbox])

        #         batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

        #         #print("(2) pred_bbox shape", tf.shape(pred_bbox))
        #         #detections = decoder(batch_images, pred)
        #         end_inference_time = time.time()

        #         inference_times.append(end_inference_time - start_inference_time)

        #         for i in range(batch_size):

        #             pred_bbox = batch_pred_bbox[i]
        #             ratio = batch_ratios[i]

        #             patch_info = batch_info[i]

        #             img_path = bytes.decode((patch_info["img_path"]).numpy())
        #             patch_path = bytes.decode((patch_info["patch_path"]).numpy())
        #             img_name = os.path.basename(img_path)[:-4]
        #             patch_name = os.path.basename(patch_path)[:-4]
        #             patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

        #             if dataset.is_annotated:
        #                 patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
        #                 patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()


        #             pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
        #                     post_process_sample(pred_bbox, ratio, patch_coords, config)




        #             # if pred_patch_abs_boxes.size == 0:
        #             #     pred_img_abs_boxes = np.array([], dtype=np.int32)
        #             # else:
        #             #     pred_img_abs_boxes = (np.array(pred_patch_abs_boxes) + \
        #             #                           np.tile(patch_coords[:2], 2)).astype(np.int32)

        #             predictions["patch_predictions"][patch_name] = {
        #                 "img_name": img_name,
        #                 "img_path": img_path,
        #                 "patch_coords": patch_coords.tolist(),
        #                 "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
        #                 "pred_scores": pred_patch_scores.tolist(),
        #                 "pred_classes": pred_patch_classes.tolist()
        #             }
        #             if dataset.is_annotated:
        #                 predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
        #                 predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

        #             if img_name not in predictions["image_predictions"]:
        #                 predictions["image_predictions"][img_name] = {
        #                     "img_path": img_path,
        #                     "pred_img_abs_boxes": [],
        #                     "pred_classes": [],
        #                     "pred_scores": [],
        #                     "patch_coords": []
        #                 }

        #             pred_img_abs_boxes, pred_img_scores, pred_img_classes = \
        #                 driver_utils.get_img_detections(pred_patch_abs_boxes, 
        #                                                 pred_patch_scores, 
        #                                                 pred_patch_classes, 
        #                                                 patch_coords, 
        #                                                 img_path, 
        #                                                 buffer_pct=buffer_pct)


        #             predictions["image_predictions"][img_name]["pred_img_abs_boxes"].extend(pred_img_abs_boxes.tolist())
        #             predictions["image_predictions"][img_name]["pred_scores"].extend(pred_img_scores.tolist())
        #             predictions["image_predictions"][img_name]["pred_classes"].extend(pred_img_classes.tolist())
        #             predictions["image_predictions"][img_name]["patch_coords"].append(patch_coords.tolist())


        #     driver_utils.clip_img_boxes(predictions["image_predictions"])

        #     driver_utils.apply_nms_to_img_boxes(predictions["image_predictions"], 
        #                                         iou_thresh=config.inference["active"]["image_nms_iou_thresh"])
        #     driver_utils.add_class_detections(predictions["image_predictions"], img_set)

        #     inference_metrics.collect_statistics(predictions, img_set, dataset,
        #                                          inference_times=inference_times)
        #     if dataset.is_annotated:
        #         inference_metrics.collect_metrics(predictions, img_set, dataset)

        #     pred_dirname = os.path.basename(patch_dir)
        #     pred_dir = os.path.join(config.model_dir, "predictions", pred_dirname)
        #     os.makedirs(pred_dir)
        #     pred_path = os.path.join(pred_dir, "predictions.json")
        #     json_io.save_json(pred_path, predictions)

        #     excel_path = os.path.join(pred_dir, "results.xlsx")
        #     driver_utils.output_excel(excel_path, predictions, img_set, dataset_name)

        #     inference_entry = {
        #         "farm_name": farm_name,
        #         "field_name": field_name,
        #         "mission_date": mission_date,
        #         "dataset_name": dataset_name,
        #         "model_uuid": config.arch["model_uuid"],
        #         "value": {
        #             "job_uuid": config.arch["job_uuid"],
        #             "job_name": config.arch["job_name"],
        #             "model_uuid": config.arch["model_uuid"],
        #             "model_name": config.arch["model_name"],
        #             "prediction_dirname": pred_dirname,
        #             "metrics": predictions["metrics"]
        #         }
        #     }
        #     inference_record_io.add_entry_to_inference_record(inference_entry)


    shutil.rmtree(target_patches_dir)

def train(config):

    tf.keras.backend.clear_session()

    logger = logging.getLogger(__name__)

    # patches_dir = os.path.join(config.model_dir, "patches")
    # if not os.path.exists(patches_dir):
    #     os.makedirs(patches_dir)
    source_patches_dir = os.path.join(config.model_dir, "source_patches")

    for seq_num in range(len(config.training["training_sequence"])):

        #training_tf_record_paths = config.training["training_sequence"][seq_num]["training_tf_record_paths"]
        #validation_tf_record_paths = config.training["training_sequence"][seq_num]["validation_tf_record_paths"]       
        training_patch_dir = os.path.join(source_patches_dir, str(seq_num), "training")
        validation_patch_dir = os.path.join(source_patches_dir, str(seq_num), "validation")

        training_tf_record_paths = [os.path.join(training_patch_dir, "annotated-patches-record.tfrec")]
        validation_tf_record_paths = [os.path.join(validation_patch_dir, "annotated-patches-record.tfrec")]


        # training_tf_record_paths = []
        # validation_tf_record_paths = []
        # training_tf_record_paths_obj = []
        # training_tf_record_paths_bg = []
        # validation_tf_record_paths_obj = []
        # validation_tf_record_paths_bg = []

        driver_utils.set_active_training_params(config, seq_num)

        # training_tf_record_paths = []
        # for dataset_conf in config.training["training_sequence"][seq_num]["training_datasets"]:
        #     #training_datasets.append(MyDataset(dataset_conf))
        #     ds = DataSet(dataset_conf)
        #     training_patch_dir = driver_utils.extract_patches(ds, config)
        #     training_tf_record_paths.append(os.path.join(training_patch_dir, "annotated-patches-record.tfrec"))
        #     training_tf_record_paths_obj.append(os.path.join(training_patch_dir, "annotated-patches-with-boxes-record.tfrec"))
        #     training_tf_record_paths_bg.append(os.path.join(training_patch_dir, "annotated-patches-with-no-boxes-record.tfrec"))


        # validation_tf_record_paths = []
        # for dataset_conf in config.training["training_sequence"][seq_num]["validation_datasets"]:
        #     #training_datasets.append(MyDataset(dataset_conf))
        #     ds = DataSet(dataset_conf)
        #     validation_patch_dir = driver_utils.extract_patches(ds, config)
        #     validation_tf_record_paths.append(os.path.join(validation_patch_dir, "annotated-patches-record.tfrec"))
        #     validation_tf_record_paths_obj.append(os.path.join(validation_patch_dir, "annotated-patches-with-boxes-record.tfrec"))
        #     validation_tf_record_paths_bg.append(os.path.join(validation_patch_dir, "annotated-patches-with-no-boxes-record.tfrec"))

      

        # print("got patches")
        # exit()
        # train_data_loader = data_load.MyTrainDataLoader(training_patches, config, shuffle=True, augment=True)
        # val_data_loader = data_load.MyTrainDataLoader(validation_patches, config, shuffle=False, augment=False)
        

        # for img_set_conf in config.training["training_sequence"][seq_num]["image_sets"]:

            
        #     img_set = ImgSet(img_set_conf["farm_name"], img_set_conf["field_name"], img_set_conf["mission_date"])
        #     training_patch_dir, _ = driver_utils.create_patches(
        #         img_set_conf["training_patch_extraction_params"], img_set, "training")
        #     validation_patch_dir, _ = driver_utils.create_patches(
        #         img_set_conf["validation_patch_extraction_params"], img_set, "validation")

        #     training_tf_record_paths_obj.append(os.path.join(training_patch_dir, "patches-with-boxes-record.tfrec"))
        #     validation_tf_record_paths_obj.append(os.path.join(validation_patch_dir, "patches-with-boxes-record.tfrec"))
        #     training_tf_record_paths_bg.append(os.path.join(training_patch_dir, "patches-with-no-boxes-record.tfrec"))
        #     validation_tf_record_paths_bg.append(os.path.join(validation_patch_dir, "patches-with-no-boxes-record.tfrec"))
        #     training_tf_record_paths.append(os.path.join(training_patch_dir, "patches-record.tfrec"))
        #     validation_tf_record_paths.append(os.path.join(validation_patch_dir, "patches-record.tfrec"))

        data_loader_type = config.training["active"]["data_loader"]["type"]
        if data_loader_type == "default":
            #train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
            #val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)
            train_data_loader = data_load.PreLoadedTrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
            val_data_loader = data_load.PreLoadedTrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)
            
        # elif data_loader_type == "split":
        #     train_data_loader = data_load.SplitDataLoader(training_tf_record_paths_obj, training_tf_record_paths_bg,
        #                                                   config, shuffle=True, augment=True)
        #     val_data_loader = data_load.SplitDataLoader(validation_tf_record_paths_obj, validation_tf_record_paths_bg,
        #                                                   config, shuffle=False, augment=False)
        else:
            raise RuntimeError("Unrecognized data loader type: {}".format(data_loader_type))


        train_dataset, num_train_images = train_data_loader.create_batched_dataset(
                                                take_percent=config.training["active"]["percent_of_training_set_used"])

        val_dataset, num_val_images = val_data_loader.create_batched_dataset(
                                                take_percent=config.training["active"]["percent_of_validation_set_used"])

        if config.arch["model_type"] == "yolov4":
            yolov4 = YOLOv4(config)
        elif config.arch["model_type"] == "yolov4_tiny":
            yolov4 = YOLOv4Tiny(config)

        loss_fn = YOLOv4Loss(config)


        input_shape = (config.training["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
        yolov4.build(input_shape=input_shape)


        if seq_num == 0:
            layer_lookup = yolov4.get_layer_lookup()
            layer_lookup_path = os.path.join(config.weights_dir, "layer_lookup.json")
            json_io.save_json(layer_lookup_path, layer_lookup)

        else:
            model_io.load_select_weights(yolov4, config)

        optimizer = tf.optimizers.Adam()


        train_loss_metric = tf.metrics.Mean()
        val_loss_metric = tf.metrics.Mean()

        @tf.function
        def train_step(batch_images, batch_labels):
            with tf.GradientTape() as tape:
                conv = yolov4(batch_images, training=True)
                loss_value = loss_fn(batch_labels, conv)

            #if np.isnan(loss_value):
            #    raise RuntimeError("NaN loss has occurred.")
            gradients = tape.gradient(target=loss_value, sources=yolov4.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, yolov4.trainable_variables))
            train_loss_metric.update_state(values=loss_value)



        train_steps_per_epoch = np.sum([1 for i in train_dataset])
        val_steps_per_epoch = np.sum([1 for i in val_dataset])
        loss_record = {
            "training_loss": { "values": [],
                               "best": {"epoch": -1, "value": sys.float_info.max},
                               "epochs_since_improvement": 0}, 
            "validation_loss": {"values": [],
                                "best": {"epoch": -1, "value": sys.float_info.max},
                                "epochs_since_improvement": 0}
        }

        logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
                     config.arch["model_type"], config.arch["model_name"], num_train_images, num_val_images))


        loss_record_path = os.path.join(config.loss_records_dir, str(seq_num) + ".json")

        max_num_epochs = config.training["active"]["max_num_epochs"]
        steps_taken = 0
        #with tf.profiler.experimental.Profile('logdir'):
        for epoch in range(max_num_epochs):
            if epoch == 0:
                disp_training_best = float("inf")
                disp_validation_best = float("inf")
            else:
                disp_training_best = loss_record["training_loss"]["best"]["value"]
                disp_validation_best = loss_record["validation_loss"]["best"]["value"]


            train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
            for batch_data in train_bar:
            #for step in tqdm.trange(train_steps_per_epoch):

                #with tf.profiler.experimental.Trace('train', step_num=steps_taken, _r=1):

                #batch_data = next(iter(train_dataset))

                optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))

                batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)

                train_step(batch_images, batch_labels)
                if np.isnan(train_loss_metric.result()):
                    raise RuntimeError("NaN loss has occurred (training dataset).")
                train_bar.set_description("Epoch: {}/{} | t. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                          epoch, max_num_epochs-1, train_loss_metric.result(), 
                                          disp_training_best,
                                          loss_record["training_loss"]["best"]["epoch"]))
                steps_taken += 1


            cur_training_loss = float(train_loss_metric.result())

            cur_training_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "training_loss", cur_training_loss, epoch)
            if cur_training_loss_is_best and config.training["active"]["save_method"] == "best_training_loss":
                model_io.save_model_weights(yolov4, config, seq_num, epoch)

            json_io.save_json(loss_record_path, loss_record)

            train_loss_metric.reset_states()


            val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
            
            for batch_data in val_bar:
                batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
                conv = yolov4(batch_images, training=False)
                loss_value = loss_fn(batch_labels, conv)

                val_loss_metric.update_state(values=loss_value)
                if np.isnan(val_loss_metric.result()):
                    raise RuntimeError("NaN loss has occurred (validation dataset).")

                val_bar.set_description("Epoch: {}/{} | v. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                        epoch, max_num_epochs-1, val_loss_metric.result(), 
                                        disp_validation_best,
                                        loss_record["validation_loss"]["best"]["epoch"]))

            
            cur_validation_loss = float(val_loss_metric.result())

            cur_validation_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "validation_loss", cur_validation_loss, epoch)
            if cur_validation_loss_is_best and config.training["active"]["save_method"] == "best_validation_loss":
                model_io.save_model_weights(yolov4, config, seq_num, epoch)    


            json_io.save_json(loss_record_path, loss_record)
            
            val_loss_metric.reset_states()

            if driver_utils.stop_early(config, loss_record):
                break




    shutil.rmtree(source_patches_dir)