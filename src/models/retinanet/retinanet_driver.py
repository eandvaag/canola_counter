import tensorflow as tf
import numpy as np
import time
import os
import shutil
import tqdm
import logging
import time

from image_set import ImgSet

from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          inference_record_io

from models.retinanet.loss import RetinaNetLoss
from models.retinanet.retinanet import RetinaNet
import models.retinanet.data_load as data_load
from models.retinanet.encode import Decoder


from io_utils import json_io
from io_utils import tf_record_io



def post_process_sample(detections, resize_ratios, index):

    num_detections = detections.valid_detections[index]
    classes = detections.nmsed_classes[index][:num_detections]
    scores = detections.nmsed_scores[index][:num_detections]
    boxes = detections.nmsed_boxes[index][:num_detections]

    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()

    boxes = box_utils.swap_xy_np(boxes)
    boxes = np.stack([
        boxes[:, 0] * resize_ratios[index][0],
        boxes[:, 1] * resize_ratios[index][1],
        boxes[:, 2] * resize_ratios[index][0],
        boxes[:, 3] * resize_ratios[index][1]
    ], axis=-1)

    boxes = np.rint(boxes).astype(np.int32)
    scores = scores.astype(np.float32)
    classes = classes.astype(np.int32)

    return boxes, scores, classes


def generate_predictions(config):

    logger = logging.getLogger(__name__)

    dataset_index = 0
    for img_set_num, img_set_conf in enumerate(config.inference["image_sets"]):

        farm_name = img_set_conf["farm_name"]
        field_name = img_set_conf["field_name"]
        mission_date = img_set_conf["mission_date"] 
        img_set = ImgSet(farm_name, field_name, mission_date)

        driver_utils.set_active_inference_params(config, img_set_num)

        for dataset_name in img_set_conf["datasets"]:

            dataset = img_set.datasets[dataset_name]

            patch_dir, _ = driver_utils.create_patches(config.inference["active"]["inference_patch_extraction_params"], 
                                                        img_set, dataset_name)


            tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

            data_loader = data_load.InferenceDataLoader(tf_record_path, config)
            tf_dataset, tf_dataset_size = data_loader.create_dataset()

            retinanet = RetinaNet(config)
            decoder = Decoder(config)

            input_shape = (config.inference["active"]["batch_size"], *(data_loader.get_model_input_shape()))
            # *(config.arch["input_img_shape"]))
            retinanet.build(input_shape=input_shape)

            model_io.load_all_weights(retinanet, config)

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
                pred = retinanet(batch_images, training=False)
                detections = decoder(batch_images, pred)
                end_inference_time = time.time()

                inference_times.append(end_inference_time - start_inference_time)

                for i in range(batch_size):

                    pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = post_process_sample(detections, batch_ratios, i)

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
                    if dataset.is_annotated:
                        predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
                        predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

                    if img_name not in predictions["image_predictions"]:
                        predictions["image_predictions"][img_name] = {
                            "img_path": img_path,
                            "pred_img_abs_boxes": [],
                            "pred_classes": [],
                            "pred_scores": [],
                            "patch_coords": []
                        }

                    predictions["image_predictions"][img_name]["pred_img_abs_boxes"].extend(pred_img_abs_boxes.tolist())
                    predictions["image_predictions"][img_name]["pred_scores"].extend(pred_patch_scores.tolist())
                    predictions["image_predictions"][img_name]["pred_classes"].extend(pred_patch_classes.tolist())
                    predictions["image_predictions"][img_name]["patch_coords"].append(patch_coords.tolist())


            driver_utils.clip_img_boxes(predictions["image_predictions"])
            driver_utils.apply_nms_to_img_boxes(predictions["image_predictions"], 
                                                iou_thresh=config.inference["active"]["image_nms_iou_thresh"])
            driver_utils.add_class_detections(predictions["image_predictions"], config)

            # for img_name in predictions["image_predictions"].keys():
            #     if len(predictions["image_predictions"][img_name]["pred_img_abs_boxes"]) > 0:
            #         pred_img_abs_boxes = np.array(predictions["image_predictions"][img_name]["pred_img_abs_boxes"])
            #         pred_img_abs_boxes = box_utils.clip_boxes_np(pred_img_abs_boxes, img_set.img_width, img_set.img_height)
            #         predictions["image_predictions"][img_name]["pred_img_abs_boxes"] = pred_img_abs_boxes.tolist()
            #         nms_boxes, nms_classes, nms_scores = box_utils.non_max_suppression(
            #                                                 pred_img_abs_boxes,
            #                                                 np.array(predictions["image_predictions"][img_name]["pred_classes"]),
            #                                                 np.array(predictions["image_predictions"][img_name]["pred_scores"]),
            #                                                 iou_thresh=config.inference["active"]["image_nms_iou_thresh"])
            #     else:
            #         nms_boxes = np.array([])
            #         nms_classes = np.array([])
            #         nms_scores = np.array([])

            #     predictions["image_predictions"][img_name]["nms_pred_img_abs_boxes"] = nms_boxes.tolist()
            #     predictions["image_predictions"][img_name]["nms_pred_classes"] = nms_classes.tolist()
            #     predictions["image_predictions"][img_name]["nms_pred_scores"] = nms_scores.tolist()
            #     unique, counts = np.unique(nms_classes, return_counts=True)
            #     class_num_to_count = dict(zip(unique, counts))
            #     pred_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
            #     pred_class_boxes = {}
            #     for class_num in class_num_to_count.keys():
            #         class_name = config.arch["reverse_class_map"][class_num]
            #         pred_class_counts[class_name] = int(class_num_to_count[class_num])
            #         pred_class_boxes[class_name] = (nms_boxes[class_num == nms_classes]).tolist()


            #     predictions["image_predictions"][img_name]["pred_class_counts"] = pred_class_counts
            #     predictions["image_predictions"][img_name]["pred_class_boxes"] = pred_class_boxes


            total_inference_time = float(np.sum(inference_times))
            per_patch_inference_time = float(total_inference_time / tf_dataset_size)
            per_image_inference_time = float(total_inference_time / len(predictions["image_predictions"]))

            predictions["metrics"] = {}
            predictions["metrics"]["Total Inference Time (s)"] = {}
            predictions["metrics"]["Total Inference Time (s)"]["---"] = total_inference_time
            predictions["metrics"]["Per-Image Inference Time (s)"] = {}
            predictions["metrics"]["Per-Image Inference Time (s)"]["---"] = per_image_inference_time
            predictions["metrics"]["Per-Patch Inference Time (s)"] = {}
            predictions["metrics"]["Per-Patch Inference Time (s)"]["---"] = per_patch_inference_time

            if dataset.is_annotated:
                inference_metrics.collect_metrics(predictions, dataset, config)

            pred_dirname = os.path.basename(patch_dir)
            pred_dir = os.path.join(config.model_dir, "predictions", pred_dirname)
            os.makedirs(pred_dir)
            pred_path = os.path.join(pred_dir, "predictions.json")
            json_io.save_json(pred_path, predictions)

            excel_path = os.path.join(pred_dir, "results.xlsx")
            driver_utils.output_excel(excel_path, predictions, img_set, dataset_name, config)

            inference_entry = {
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "dataset_name": dataset_name,
                "model_uuid": config.arch["model_uuid"],
                "value": {
                    "group_uuid": config.arch["group_uuid"],
                    "group_name": config.arch["group_name"],
                    "model_uuid": config.arch["model_uuid"],
                    "model_name": config.arch["model_name"],
                    "prediction_dirname": pred_dirname,
                    "metrics": predictions["metrics"]
                }
            }
            inference_record_io.add_entry_to_inference_record(inference_entry)

            dataset_index += 1



def train(config):

    logger = logging.getLogger(__name__)


    for seq_num in range(len(config.training["training_sequence"])):

        training_tf_record_paths = []
        validation_tf_record_paths = []

        driver_utils.set_active_training_params(config, seq_num)

        for img_set_conf in config.training["training_sequence"][seq_num]["image_sets"]:
            img_set = ImgSet(img_set_conf["farm_name"], img_set_conf["field_name"], img_set_conf["mission_date"])
            training_patch_dir, _ = driver_utils.create_patches(
                config.training["active"]["training_patch_extraction_params"], img_set, "training")
            validation_patch_dir, _ = driver_utils.create_patches(
                config.training["active"]["validation_patch_extraction_params"], img_set, "validation")

            training_tf_record_paths.append(os.path.join(training_patch_dir, "patches-with-boxes-record.tfrec"))
            validation_tf_record_paths.append(os.path.join(validation_patch_dir, "patches-with-boxes-record.tfrec"))


        train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
        train_dataset, train_dataset_size = train_data_loader.create_batched_dataset(
                                                take_percent=config.training["active"]["percent_of_training_set_used"])

        val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)
        #val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=True, augment=True)
        val_dataset, val_dataset_size = val_data_loader.create_batched_dataset()


        
        retinanet = RetinaNet(config)
        loss_fn = RetinaNetLoss(config)

        input_shape = (config.training["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
        retinanet.build(input_shape=input_shape)


        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
        #                                                             decay_steps=steps_per_epoch * Config.learning_rate_decay_epochs,
        #                                                             decay_rate=0.96)
        if seq_num == 0:
            layer_lookup = retinanet.get_layer_lookup()
            layer_lookup_path = os.path.join(config.weights_dir, "layer_lookup.json")
            json_io.save_json(layer_lookup_path, layer_lookup)

        else:
            model_io.load_select_weights(retinanet, config)

        optimizer = tf.optimizers.Adam()


        train_loss_metric = tf.metrics.Mean()
        val_loss_metric = tf.metrics.Mean()

        def train_step(batch_images, batch_labels):
            with tf.GradientTape() as tape:
                pred = retinanet(batch_images, training=True)
                loss_value = loss_fn(y_true=batch_labels, y_pred=pred)

            gradients = tape.gradient(target=loss_value, sources=retinanet.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, retinanet.trainable_variables))
            train_loss_metric.update_state(values=loss_value)



        train_steps_per_epoch = np.sum([1 for i in train_dataset])
        val_steps_per_epoch = np.sum([1 for i in val_dataset])
        best_val_loss = float("inf")
        epochs_since_improvement = 0
        loss_record = {
            "training_loss": { "values": [],
                               "best": {"epoch": -1, "value": float("inf")},
                               "epochs_since_improvement": 0},
            "validation_loss": {"values": [],
                                "best": {"epoch": -1, "value": float("inf")},
                                "epochs_since_improvement": 0}
        }

        logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
                     config.arch["model_type"], config.arch["model_name"], train_dataset_size, val_dataset_size))


        max_num_epochs = config.training["active"]["max_num_epochs"]
        steps_taken = 0
        for epoch in range(max_num_epochs):

            train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
            for batch_data in train_bar:

                optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))

                batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
                #print("num batch labels", (batch_labels.numpy()[:,:,4]).size)
                #print("num -1", np.sum(batch_labels.numpy()[:,:,4] == -1))
                #print("num not -1", np.sum(batch_labels.numpy()[:,:,4] != -1))
                train_step(batch_images, batch_labels)
                train_bar.set_description("Epoch: {}/{} | t. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                          epoch, max_num_epochs-1, train_loss_metric.result(), 
                                          loss_record["training_loss"]["best"]["value"],
                                          loss_record["training_loss"]["best"]["epoch"]))
                steps_taken += 1


            val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
            #v_i = 0
            for batch_data in val_bar:
                batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)

                #pred = retinanet(batch_images, training=True)
                pred = retinanet(batch_images, training=False)
                loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
                prev_val_loss = float(val_loss_metric.result())
                val_loss_metric.update_state(values=loss_value)
                # if v_i > 0:
                #     if val_loss_metric.result() - prev_val_loss > 1:
                #         print("sudden upward spike")
                #         print("updated_state_value", float(val_loss_metric.result()))
                #         print("loss_value", loss_value)
                #         print("prev_val_loss", prev_val_loss)
                #         print("batch_images", batch_images)
                #         print("batch_labels", batch_labels)
                #         print("pred", pred)
                #         print("num batch labels", (batch_labels.numpy()[:,:,4]).size)
                #         print("num -1", np.sum(batch_labels.numpy()[:,:,4] == -1))
                #         print("num not -1", np.sum(batch_labels.numpy()[:,:,4] != -1))
                #         exit()
                # v_i += 1

                val_bar.set_description("Epoch: {}/{} | v. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                        epoch, max_num_epochs-1, val_loss_metric.result(), 
                                        loss_record["validation_loss"]["best"]["value"],
                                        loss_record["validation_loss"]["best"]["epoch"]))

            cur_training_loss = float(train_loss_metric.result())
            cur_validation_loss = float(val_loss_metric.result())

            cur_training_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "training_loss", cur_training_loss, epoch)
            if cur_training_loss_is_best and config.training["active"]["save_method"] == "best_training_loss":
                model_io.save_model_weights(retinanet, config, seq_num, epoch)

            cur_validation_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "validation_loss", cur_validation_loss, epoch)
            if cur_validation_loss_is_best and config.training["active"]["save_method"] == "best_validation_loss":
                model_io.save_model_weights(retinanet, config, seq_num, epoch)    

            if driver_utils.stop_early(config, loss_record):
                break


            train_loss_metric.reset_states()
            val_loss_metric.reset_states()


        loss_record_path = os.path.join(config.loss_records_dir, str(seq_num) + ".json")
        json_io.save_json(loss_record_path, loss_record)