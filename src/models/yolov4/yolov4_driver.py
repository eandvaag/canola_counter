import tensorflow as tf
import numpy as np
import time
import sys
import os
import glob
import shutil
import tqdm
import logging
import time
import random
import cv2

from image_set import DataSet, Image
import build_datasets

from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          inference_record_io, \
                          model_vis

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

    pred_boxes = np.rint(pred_boxes).astype(np.int32)
    pred_scores = pred_scores.astype(np.float32)
    pred_classes = pred_classes.astype(np.int32)

    if apply_nms:
        pred_boxes, pred_classes, pred_scores = box_utils.non_max_suppression_with_classes(
            pred_boxes,
            pred_classes,
            pred_scores,
            iou_thresh=config.inference["patch_nms_iou_thresh"])

    return pred_boxes, pred_scores, pred_classes



def generate_predictions(config):

    logger = logging.getLogger(__name__)

    # patches_dir = os.path.join(config.model_dir, "patches")
    # if not os.path.exists(patches_dir):
    #     os.makedirs(patches_dir)

    similarity_analysis_debug(config)
    #optimal_train_val_thresh, optimal_train_val_diff = calculate_optimal_score_thresh(config)
    #print("got optimal thresh val: {}".format(optimal_thresh_val))




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



    #data_loader = data_load.InferenceDataLoader(tf_record_path, config)
    all_targets_dir = os.path.join(config.model_dir, "target_patches") #config.inference["inference_patch_dir"]



    #for dataset_conf in config.inference["datasets"]:
    for target in targets:

        #driver_utils.set_active_inference_params(config)
        #image_set = ImageSet(image_set_conf)
        #dataset = DataSet(dataset_conf)
        dataset = DataSet(target)
        target_patches_dir = os.path.join(all_targets_dir, 
                                            target["farm_name"] + "_" + target["field_name"] + "_" + target["mission_date"])

        if config.arch["model_type"] == "yolov4":
            yolov4 = YOLOv4(config)
        elif config.arch["model_type"] == "yolov4_tiny":
            yolov4 = YOLOv4Tiny(config)
        decoder = Decoder(config)

        #patch_dir = driver_utils.extract_patches(image_set.all_dataset, config)
        tf_record_names = ["annotated-patches-record.tfrec", "unannotated-patches-record.tfrec"]

        predictions = driver_utils.create_predictions_skeleton(dataset)
        inference_times = []

        for k, tf_record_name in enumerate(tf_record_names):
            is_annotated = k == 0
            tf_record_path = os.path.join(target_patches_dir, tf_record_name)
            data_loader = data_load.InferenceDataLoader(tf_record_path, config) #, mask_border_boxes=True)
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
                            "patch_coords": [],
                            # "boxes_added_per_patch": []
                            # "nt_pred_image_abs_boxes": [],
                            # "nt_pred_classes": [],
                            # "nt_pred_scores": []                     
                        }

                    pred_image_abs_boxes, pred_image_scores, pred_image_classes = \
                        driver_utils.get_image_detections(pred_patch_abs_boxes, 
                                                        pred_patch_scores, 
                                                        pred_patch_classes, 
                                                        patch_coords, 
                                                        image_path, 
                                                        trim=True)
                                                        #config.inference["rm_edge_boxes"],
                                                        #patch_border_buffer_percent=config.inference["patch_border_buffer_percent"]) #True) #buffer_pct=buffer_pct)

                    # nt_pred_image_abs_boxes, nt_pred_image_scores, nt_pred_image_classes = \
                    #     driver_utils.get_image_detections(pred_patch_abs_boxes, 
                    #                                     pred_patch_scores, 
                    #                                     pred_patch_classes, 
                    #                                     patch_coords, 
                    #                                     image_path, 
                    #                                     trim=False)

                    # predictions["image_predictions"][image_name]["boxes_added_per_patch"].append(pred_image_abs_boxes.shape[0])
                    predictions["image_predictions"][image_name]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
                    predictions["image_predictions"][image_name]["pred_scores"].extend(pred_image_scores.tolist())
                    predictions["image_predictions"][image_name]["pred_classes"].extend(pred_image_classes.tolist())
                    predictions["image_predictions"][image_name]["patch_coords"].append(patch_coords.tolist())
                    
                    # predictions["image_predictions"][image_name]["nt_pred_image_abs_boxes"].extend(nt_pred_image_abs_boxes.tolist())
                    # predictions["image_predictions"][image_name]["nt_pred_scores"].extend(nt_pred_image_scores.tolist())
                    # predictions["image_predictions"][image_name]["nt_pred_classes"].extend(nt_pred_image_classes.tolist())




        driver_utils.clip_image_boxes(predictions["image_predictions"])

        
        #out_dir = os.path.join(config.model_dir, "image_pre_nms")
        #nt_out_dir = os.path.join(config.model_dir, "nt_image_pre_nms")
        #f not os.path.exists(out_dir):
        #    os.makedirs(out_dir)
        #    os.makedirs(nt_out_dir)
        #output_image_predictions(predictions["image_predictions"], out_dir)
        #output_image_predictions(predictions["image_predictions"], nt_out_dir, nt=True)

        driver_utils.apply_nms_to_image_boxes(predictions["image_predictions"], 
                                            iou_thresh=config.inference["image_nms_iou_thresh"])

        #driver_utils.apply_careful_nms_to_image_boxes(predictions["image_predictions"], iou_thresh=0.1)


        #out_dir = os.path.join(config.model_dir, "image_post_nms")
        #nt_out_dir = os.path.join(config.model_dir, "nt_image_post_nms")
        #if not os.path.exists(out_dir):
        #    os.makedirs(out_dir)
            #os.makedirs(nt_out_dir)
        #output_image_predictions(predictions["image_predictions"], out_dir)
        #output_image_predictions(predictions["image_predictions"], nt_out_dir, nt=True)


        annotations = w3c_io.load_annotations(dataset.annotations_path, config.arch["class_map"])
        #optimal_train_val_thresh, optimal_train_val_diff = inference_metrics.calculate_optimal_score_threshold(annotations, predictions, config.arch["training_validation_images"])
        #logger.info("Optimal train/val thresh is {}".format(optimal_train_val_thresh))

        driver_utils.add_class_detections(predictions["image_predictions"], config) #, optimal_train_val_thresh)

        metrics = driver_utils.create_metrics_skeleton(dataset)

        #metrics["point"]["train_val_optimal_score_threshold"] = optimal_train_val_thresh
        #metrics["point"]["train_val_optimal_score_threshold"] = {}
        #metrics["point"]["train_val_optimal_score_threshold"]["threshold_value"] = optimal_train_val_thresh
        #metrics["point"]["train_val_optimal_score_threshold"]["mean_absolute_difference"] = optimal_train_val_diff


        all_image_names = predictions["image_predictions"].keys()
        inference_metrics.collect_statistics(all_image_names, metrics, predictions, config, inference_times=inference_times)
        inference_metrics.collect_metrics(all_image_names, metrics, predictions, dataset, config)
        #inference_metrics.similarity_analysis(config, predictions)


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

        metrics_path = os.path.join(results_dir, "metrics.json")
        json_io.save_json(metrics_path, metrics)

        # test_metrics_path = os.path.join(results_dir, "test_metrics.json")
        # json_io.save_json(test_metrics_path, test_metrics)

        annotations_path = os.path.join(results_dir, "annotations.json")
        w3c_io.save_annotations(annotations_path, predictions, config)

        excel_path = os.path.join(results_dir, "results.xlsx")
        driver_utils.output_excel(excel_path, predictions, dataset, config)



    source_patches_dir = os.path.join(config.model_dir, "source_patches")
    shutil.rmtree(source_patches_dir)

    shutil.rmtree(all_targets_dir)

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

        #data_loader_type = config.training["active"]["data_loader"]["type"]
        #if data_loader_type == "default":
            #train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
            #val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

        # elif data_loader_type == "split":
        #     train_data_loader = data_load.SplitDataLoader(training_tf_record_paths_obj, training_tf_record_paths_bg,
        #                                                   config, shuffle=True, augment=True)
        #     val_data_loader = data_load.SplitDataLoader(validation_tf_record_paths_obj, validation_tf_record_paths_bg,
        #                                                   config, shuffle=False, augment=False)
        #else:
        #    raise RuntimeError("Unrecognized data loader type: {}".format(data_loader_type))
        try:
            train_data_loader = data_load.PreLoadedTrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
            val_data_loader = data_load.PreLoadedTrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

            train_dataset, num_train_images = train_data_loader.create_batched_dataset(
                                                    take_percent=config.training["active"]["percent_of_training_set_used"])

            val_dataset, num_val_images = val_data_loader.create_batched_dataset(
                                                    take_percent=config.training["active"]["percent_of_validation_set_used"])
        
        except RuntimeError:
            logger.info("Switching to non-preloaded data loader due to high memory usage.")
            train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
            val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

            train_dataset, num_train_images = train_data_loader.create_batched_dataset(
                                                    take_percent=config.training["active"]["percent_of_training_set_used"])

            val_dataset, num_val_images = val_data_loader.create_batched_dataset(
                                                    take_percent=config.training["active"]["percent_of_validation_set_used"])


        logger.info("Building model...")


        if config.arch["model_type"] == "yolov4":
            yolov4 = YOLOv4(config)
        elif config.arch["model_type"] == "yolov4_tiny":
            yolov4 = YOLOv4Tiny(config)

        loss_fn = YOLOv4Loss(config)


        input_shape = (config.training["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
        yolov4.build(input_shape=input_shape)

        logger.info("Model built.")


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
                loss_value += sum(yolov4.losses)

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
                loss_value += sum(yolov4.losses)

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

    #shutil.rmtree(source_patches_dir)



#def calculate_optimal_score_thresh(config):
def similarity_analysis_debug(config):


    logger = logging.getLogger(__name__)

    logger.info("Starting to calculate optimal score threshold.")
    image_names = config.arch["training_validation_images"]
    # if len(image_names) == 0:
    #     return 0.5

    patch_results = {}
    
    
    patches_dir = os.path.join(config.model_dir, "DEBUG_MATCH")
    #source_patches_dir = os.path.join(config.model_dir, "source_patches")

    # debug_out_dir = os.path.join(config.model_dir, "debug_out")
    # if not os.path.exists(debug_out_dir):
    #     os.makedirs(debug_out_dir)

    # build_datasets.build_train_val_thresh_dataset(config)

    #patches_dir = os.path.join(config.model_dir, "training_validation_opt_patches")

    #tf_record_paths = os.path.join(patches_dir, "annotated-patches-record.tfrec")

    #seq_source_patches_dir = sorted(glob.glob(os.path.join(source_patches_dir, "*")))[-1]
    # #seq_num = len(config.training["training_sequence"]) - 1
    
    #training_patch_dir = os.path.join(seq_source_patches_dir, "training")
    # validation_patch_dir = os.path.join(seq_source_patches_dir, "validation")

    training_tf_record_paths = [os.path.join(patches_dir, "annotated-patches-record.tfrec")]
    # validation_tf_record_paths = [os.path.join(validation_patch_dir, "annotated-patches-record.tfrec")]

    data_loader = data_load.InferenceDataLoader(training_tf_record_paths, config, mask_border_boxes=False) #True)
    tf_dataset, tf_dataset_size = data_loader.create_dataset()

    if config.arch["model_type"] == "yolov4":
        yolov4 = YOLOv4(config)
    elif config.arch["model_type"] == "yolov4_tiny":
        yolov4 = YOLOv4Tiny(config)
    decoder = Decoder(config)

    #driver_utils.set_active_training_params(config, seq_num)

    input_shape = (config.inference["batch_size"], *(data_loader.get_model_input_shape()))
    yolov4.build(input_shape=input_shape)

    model_io.load_all_weights(yolov4, config)

    #buffer_pct = None
    steps = np.sum([1 for i in tf_dataset])

    # if "patch_border_buffer_percent" in config.inference:
    #     buffer_pct = config.inference["patch_border_buffer_percent"]
    # else:
    #     buffer_pct = None

    #dataset = DataSet({"farm_name":  })
    #driver_utils.create_predictions_skeleton(dataset)

    predictions = {
        "patch_predictions": {}
    }
    #predictions["image_predictions"] = {}
    for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, 
                                      desc="Generating predictions for calculating optimal score threshold")):

        batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, True)
        batch_size = batch_images.shape[0]

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
            patch_name = os.path.basename(patch_path)[:-4]
            patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

            #patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy()

            pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
                    post_process_sample(pred_bbox, ratio, patch_coords, config)


            #patch = cv2.cvtColor(cv2.imread(patch_path), cv2.COLOR_BGR2RGB)
            #ratio = np.array(image.shape[:2]) / np.array(self.input_image_shape[:2])

            #new_h = round(ratio[0] * config.arch["input_image_shape"][0])
            #new_w = round(ratio[1] * config.arch["input_image_shape"][1])
            #resized_patch = tf.image.resize(batch_images[i], size=np.array([new_h, new_w], dtype=np.int32)).numpy().astype(np.uint8)

            #output_patch_prediction(resized_patch, pred_patch_abs_boxes,
            #    pred_patch_classes, pred_patch_scores, os.path.join(debug_out_dir, patch_name) + ".png")

            #patch_results[patch_name] = {
            # predictions["patch_predictions"][patch_name] = {
            #     "pred_scores": pred_patch_scores,
            #     "num_boxes": patch_info["masked_box_count"] #patch_boxes.shape[0]
            # }




            #if dataset.is_annotated:
            #if is_annotated:
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

            predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
            predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes


    #         if image_name not in predictions["image_predictions"]:
    #             predictions["image_predictions"][image_name] = {
    #                 "image_path": image_path,
    #                 "pred_image_abs_boxes": [],
    #                 "pred_classes": [],
    #                 "pred_scores": [],
    #                 "patch_coords": []
    #             }

    #         pred_image_abs_boxes, pred_image_scores, pred_image_classes = \
    #             driver_utils.get_image_detections(pred_patch_abs_boxes, 
    #                                             pred_patch_scores, 
    #                                             pred_patch_classes, 
    #                                             patch_coords, 
    #                                             image_path, 
    #                                             buffer_pct=buffer_pct)


    #         predictions["image_predictions"][image_name]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
    #         predictions["image_predictions"][image_name]["pred_scores"].extend(pred_image_scores.tolist())
    #         predictions["image_predictions"][image_name]["pred_classes"].extend(pred_image_classes.tolist())
    #         predictions["image_predictions"][image_name]["patch_coords"].append(patch_coords.tolist())


    # driver_utils.clip_image_boxes(predictions["image_predictions"])

    # driver_utils.apply_nms_to_image_boxes(predictions["image_predictions"], 
    #                                       iou_thresh=config.inference["image_nms_iou_thresh"])


    #annotations_path = os.path.join("usr", "data", "image_sets",
    #                                config.arch["target_farm_name"], 
    #                                config.arch["target_field_name"],
    #                                config.arch["target_mission_date"],
    #                                "annotations", "annotations_w3c.json")
    #annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
    #optimal_thresh_val, _ = inference_metrics.calculate_optimal_score_threshold(annotations, predictions, image_names)
    inference_metrics.similarity_analysis(config, predictions)

    return

    optimal_thresh_val = None
    optimal_mean_abs_diff = np.inf
    prev_mean_abs_diff = np.inf
    thresh_vals = np.arange(0.0, 1.0, 0.01)
    #print_thresh_vals = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for thresh_val in tqdm.tqdm(thresh_vals, desc="Calculating optimal threshold value"):
        abs_diffs = []

        for patch_name in patch_results.keys():
            num_boxes = patch_results[patch_name]["num_boxes"]
            num_pred_boxes = (np.where(patch_results[patch_name]["pred_scores"] >= thresh_val)[0]).size

            abs_diffs.append(abs(num_boxes - num_pred_boxes))
        mean_abs_diff = float(np.mean(abs_diffs))

        logger.info("thresh_val: {}. sum_abs_diff: {}. best_thresh_val: {}. optimal_sum_abs_diff: {}.".format(
            thresh_val, mean_abs_diff, optimal_thresh_val, optimal_mean_abs_diff
        ))
        if prev_mean_abs_diff < mean_abs_diff:
            logger.info("breaking at thresh_val {}".format(thresh_val))
            break

        if mean_abs_diff <= optimal_mean_abs_diff:
            optimal_mean_abs_diff = mean_abs_diff
            optimal_thresh_val = thresh_val

        prev_mean_abs_diff = mean_abs_diff

    logger.info("Optimal score threshold for training patches is: {}".format(optimal_thresh_val))
    # #optimal_thresh_record_path = os.path.join(config.model_dir, "optimal_score_threshold.json")
    # #optimal_thresh_record = {
    # #    "optimal_thresh": optimal_thresh_val
    # #}
    # #json_io.save_json(optimal_thresh_record_path, optimal_thresh_record)

    #shutil.rmtree(patches_dir)

    return optimal_thresh_val, optimal_mean_abs_diff





def output_image_predictions(image_predictions, out_dir, nt=False):
    for image_name in image_predictions.keys():
        image_path = image_predictions[image_name]["image_path"]
        image_array = Image(image_path).load_image_array()

        boxes_key = "nt_pred_image_abs_boxes" if nt else "pred_image_abs_boxes"
        classes_key = "nt_pred_classes" if nt else "pred_classes"
        scores_key = "nt_pred_scores" if nt else "pred_scores"

        out_array = model_vis.draw_boxes_on_image(image_array,
                                                  image_predictions[image_name][boxes_key], #"pred_image_abs_boxes"],
                                                  image_predictions[image_name][classes_key], #"pred_classes"],
                                                  image_predictions[image_name][scores_key], #"pred_scores"],
                                                  class_map={"plant": 0},
                                                  gt_boxes=None,
                                                  patch_coords=image_predictions[image_name]["patch_coords"], #None,
                                                  display_class=False,
                                                  display_score=True
                                                  )
        out_path = os.path.join(out_dir, image_name + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))


def output_patch_prediction(patch, pred_boxes, pred_classes, pred_scores, out_path):

    out_array = model_vis.draw_boxes_on_image(patch,
                      pred_boxes,
                      pred_classes,
                      pred_scores,
                      class_map={"plant": 0},
                      gt_boxes=None,
                      patch_coords=None,
                      display_class=False,
                      display_score=False)
    cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))