import tensorflow as tf
import numpy as np
import time
import os
import shutil
import tqdm
import logging
import time

import glob
import cv2

import extract_patches as ep

import models.common.box_utils as box_utils
import models.common.model_load as model_load
import models.common.inference_metrics as inference_metrics

from models.centernet.loss import CenterNetLoss
from models.centernet.centernet import CenterNet
import models.centernet.data_load as data_load
from models.centernet.encode import Decoder


from io_utils import json_io
from io_utils import tf_record_io



def post_process_sample(detections, resize_ratios, index):

    num_detections = detections[0][index]
    boxes = detections[1][index][:num_detections]
    scores = detections[2][index][:num_detections]
    classes = detections[3][index][:num_detections]

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


def generate_predictions(patch_dir, pred_dir, img_dataset, config):

    logger = logging.getLogger(__name__)

    if len(os.listdir(config.weights_dir)) == 0:
        raise RuntimeError("Weights directory for '{}' is empty. Did you forget to train the model?".format(config.instance_name))

    tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

    data_loader = data_load.InferenceDataLoader(tf_record_path, config)
    dataset, dataset_size = data_loader.create_dataset()

    centernet = CenterNet(config)
    decoder = Decoder(config)

    weights_path = model_load.get_weights_path(config)
    centernet.load_weights(weights_path)


    predictions = {"image_predictions": {}, "patch_predictions": {}}
    steps = np.sum([1 for i in dataset])


    logger.info("{} ('{}'): Running inference on {} images.".format(config.model_type, 
                                                                    config.instance_name, 
                                                                    dataset_size))

    inference_times = []
    for step, batch_data in enumerate(tqdm.tqdm(dataset, total=steps, desc="Generating predictions")):

        batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, img_dataset.is_annotated)
        batch_size = batch_images.shape[0]


        start_inference_time = time.time()
        pred = centernet(batch_images)
        detections = decoder(pred)
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

            if img_dataset.is_annotated:
                patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()

            if pred_patch_abs_boxes.size == 0:
                pred_img_abs_boxes = np.array([], dtype=np.int32)
            else:
                pred_img_abs_boxes = (np.array(pred_patch_abs_boxes) + \
                                      np.tile(patch_coords[:2], 2)).astype(np.int32)

            predictions["patch_predictions"][patch_name] = {
                "img_name": img_name,
                "patch_coords": patch_coords.tolist(),
                "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
                "pred_scores": pred_patch_scores.tolist(),
                "pred_classes": pred_patch_classes.tolist()
            }
            if img_dataset.is_annotated:
                predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
                predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

            if img_name not in predictions["image_predictions"]:
                predictions["image_predictions"][img_name] = {
                    "pred_img_abs_boxes": [],
                    "pred_classes": [],
                    "pred_scores": [],
                    "patch_coords": []
                }

            predictions["image_predictions"][img_name]["pred_img_abs_boxes"].extend(pred_img_abs_boxes.tolist())
            predictions["image_predictions"][img_name]["pred_scores"].extend(pred_patch_scores.tolist())
            predictions["image_predictions"][img_name]["pred_classes"].extend(pred_patch_classes.tolist())
            predictions["image_predictions"][img_name]["patch_coords"].append(patch_coords.tolist())


    for img_name in predictions["image_predictions"].keys():
        if len(predictions["image_predictions"][img_name]["pred_img_abs_boxes"]) > 0:
            nms_boxes, nms_classes, nms_scores = box_utils.non_max_suppression(
                                                    np.array(predictions["image_predictions"][img_name]["pred_img_abs_boxes"]),
                                                    np.array(predictions["image_predictions"][img_name]["pred_classes"]),
                                                    np.array(predictions["image_predictions"][img_name]["pred_scores"]),
                                                    iou_thresh=config.img_nms_iou_thresh)
        else:
            nms_boxes = np.array([])
            nms_classes = np.array([])
            nms_scores = np.array([])

        predictions["image_predictions"][img_name]["nms_pred_img_abs_boxes"] = nms_boxes.tolist()
        predictions["image_predictions"][img_name]["nms_pred_classes"] = nms_classes.tolist()
        predictions["image_predictions"][img_name]["nms_pred_scores"] = nms_scores.tolist()
        unique, counts = np.unique(nms_classes, return_counts=True)
        class_num_to_count = dict(zip(unique, counts))
        pred_class_counts = {k: 0 for k in config.class_map.keys()}
        pred_class_boxes = {}
        for class_num in class_num_to_count.keys():
            class_name = config.reverse_class_map[class_num]
            pred_class_counts[class_name] = int(class_num_to_count[class_num])
            pred_class_boxes[class_name] = (nms_boxes[class_num == nms_classes]).tolist()


        predictions["image_predictions"][img_name]["pred_class_counts"] = pred_class_counts
        predictions["image_predictions"][img_name]["pred_class_boxes"] = pred_class_boxes
        #predictions["image_predictions"][img_name]["pred_count"] = nms_boxes.shape[0]


    total_inference_time = float(np.sum(inference_times))
    per_patch_inference_time = float(total_inference_time / dataset_size)
    per_image_inference_time = float(total_inference_time / len(predictions["image_predictions"]))

    predictions["metrics"] = {}
    predictions["metrics"]["Total Inference Time (s)"] = {}
    predictions["metrics"]["Total Inference Time (s)"]["---"] = total_inference_time
    predictions["metrics"]["Per-Image Inference Time (s)"] = {}
    predictions["metrics"]["Per-Image Inference Time (s)"]["---"] = per_image_inference_time
    predictions["metrics"]["Per-Patch Inference Time (s)"] = {}
    predictions["metrics"]["Per-Patch Inference Time (s)"]["---"] = per_patch_inference_time

    if img_dataset.is_annotated:
        inference_metrics.collect_metrics(predictions, img_dataset, config)

    pred_path = os.path.join(pred_dir, "predictions.json")
    json_io.save_json(pred_path, predictions)




def train(training_patch_dirs, validation_patch_dirs, config):

    logger = logging.getLogger(__name__)

    training_tf_record_paths = []
    for training_patch_dir in training_patch_dirs:
        training_tf_record_paths.append(os.path.join(training_patch_dir, "patches-with-boxes-record.tfrec"))
    validation_tf_record_paths = []
    for validation_patch_dir in validation_patch_dirs:
        validation_tf_record_paths.append(os.path.join(validation_patch_dir, "patches-with-boxes-record.tfrec"))

    train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
    train_dataset, train_dataset_size = train_data_loader.create_batched_dataset(take_pct=config.pct_of_training_set_used)

    val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)
    val_dataset, val_dataset_size = val_data_loader.create_batched_dataset()


    
    centernet = CenterNet(config)

    loss_fn = CenterNetLoss(config)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
    #                                                             decay_steps=steps_per_epoch * Config.learning_rate_decay_epochs,
    #                                                             decay_rate=0.96)

    optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)


    train_loss_metric = tf.metrics.Mean()
    val_loss_metric = tf.metrics.Mean()

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = centernet(batch_images, training=True)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)

        gradients = tape.gradient(target=loss_value, sources=centernet.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, centernet.trainable_variables))
        train_loss_metric.update_state(values=loss_value)



    train_steps_per_epoch = np.sum([1 for i in train_dataset]) #int(round((np.sum([1 for i in train_dataset]) * (config.pct_of_training_set_used / 100))))
    val_steps_per_epoch = np.sum([1 for i in val_dataset])
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    loss_record = {
        "training_loss": { "values": [],
                           "best": float("inf"),
                           "epochs_since_improvement": 0},
        "validation_loss": {"values": [],
                            "best": float("inf"),
                            "epochs_since_improvement": 0}
    }

    logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
                 config.model_type, config.instance_name, train_dataset_size, val_dataset_size))

    for epoch in range(config.num_epochs):

        train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
        for batch_data in train_bar:
            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
            train_step(batch_images, batch_labels)
            train_bar.set_description("Epoch: {}/{} | training loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs-1,
                                                                                    train_loss_metric.result()))

        val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
        for batch_data in val_bar:
            batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
            pred = centernet(batch_images, training=False)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
            val_loss_metric.update_state(values=loss_value)
            val_bar.set_description("Epoch: {}/{} | validation loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs-1,
                                                                                    val_loss_metric.result()))

        cur_training_loss = float(train_loss_metric.result())
        cur_validation_loss = float(val_loss_metric.result())

        cur_training_loss_is_best = update_loss_tracker_entry(loss_record, "training_loss", cur_training_loss)
        if cur_training_loss_is_best and config.save_method == "best_training_loss":
            update_weights_dir(centernet, config, epoch)

        cur_validation_loss_is_best = update_loss_tracker_entry(loss_record, "validation_loss", cur_validation_loss)
        if cur_validation_loss_is_best and config.save_method == "best_validation_loss":
            update_weights_dir(centernet, config, epoch)    

        if stop_early(config, loss_record):
            break


        train_loss_metric.reset_states()
        val_loss_metric.reset_states()


    loss_record_path = os.path.join(config.model_dir, "loss_record.json")
    json_io.save_json(loss_record_path, loss_record)



def update_loss_tracker_entry(loss_tracker, key, cur_loss):

    loss_tracker[key]["values"].append(cur_loss)

    best = loss_tracker[key]["best"]
    if cur_loss < best:
        loss_tracker[key]["best"] = cur_loss
        loss_tracker[key]["epochs_since_improvement"] = 0
        return True
    else:
        loss_tracker[key]["epochs_since_improvement"] += 1
        return False


def update_weights_dir(model, config, epoch):
    shutil.rmtree(config.weights_dir)
    os.makedirs(config.weights_dir)
    model.save_weights(filepath=os.path.join(config.weights_dir, "epoch-{}".format(epoch)), save_format="tf")


def stop_early(config, loss_tracker):
    if config.early_stopping["apply"]:
        key = config.early_stopping["monitor"]
        if loss_tracker[key]["epochs_since_improvement"] >= config.early_stopping["num_epochs_tolerance"]:
            return True

    return False



# def draw_boxes_on_image(image, boxes, scores, classes):
#     idx2class_dict = {0: "plant"}
#     num_boxes = boxes.shape[0]
#     for i in range(num_boxes):
#         class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
#         cv2.rectangle(img=image, pt1=(boxes[i, 1], boxes[i, 0]), pt2=(boxes[i, 3], boxes[i, 2]), color=(250, 206, 135), thickness=2)

#         text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
#         text_width, text_height = text_size[0][0], text_size[0][1]
#         cv2.rectangle(img=image, pt1=(boxes[i, 1], boxes[i, 0]), pt2=(boxes[i, 1] + text_width, boxes[i, 0] - text_height), color=(203, 192, 255), thickness=-1)
#         cv2.putText(img=image, text=class_and_score, org=(boxes[i, 1], boxes[i, 0] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
#     return image




# def test_single_picture(picture_dir, model, config):

#     image_array = cv2.imread(picture_dir)
#     ratio = np.array(image_array.shape[:2]) / np.array(config.input_img_shape[:2])
#     img = tf.image.resize(images=image_array, size=config.input_img_shape)

#     img = tf.expand_dims(input=img, axis=0)
#     decoder = Decoder(config)

#     pred = model(img, training=False)
#     batch_boxes, batch_scores, batch_classes = decoder(pred)
#     for i in range(batch_boxes.shape[0]):
#         boxes, scores, classes = post_process_sample(batch_boxes[i], batch_scores[i], batch_classes[i], ratio, config)

#         image_with_boxes = draw_boxes_on_image(image_array, boxes, scores, classes)
#         return image_with_boxes



# def test_centernet(model, patches_dir, output_dir, config):

#     for i, img_path in enumerate(sorted(glob.glob(os.path.join(patches_dir, "*.png")))):
#         if i > 25:
#             break

#         image = test_single_picture(picture_dir=img_path, model=model, config=config)
#         cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), image)