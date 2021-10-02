import tensorflow as tf
import numpy as np
import time
import os
import shutil
import tqdm
import logging

import glob
import cv2

import extract_patches as ep

import models.detectors.common.box_utils as box_utils
import models.detectors.common.model_load as model_load

from models.detectors.centernet.loss import CenterNetLoss
from models.detectors.centernet.centernet import CenterNet
import models.detectors.centernet.data_load as data_load
from models.detectors.centernet.encode import Decoder


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

    return boxes, scores, classes


def generate_predictions(patch_dir, config, skip_if_found=True):

    logger = logging.getLogger(__name__)

    if len(os.listdir(config.weights_dir)) == 0:
        raise RuntimeError("Weights directory for '{}' is empty. Did you forget to train the model?".format(config.instance_name))

    tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")
    patch_metadata = ep.parse_patch_dir(patch_dir)
    is_annotated = patch_metadata["is_annotated"]

    data_loader = data_load.InferenceDataLoader(tf_record_path, config)
    dataset, dataset_size = data_loader.create_dataset()

    pred_dir = os.path.join(config.model_dir, os.path.basename(patch_dir))
    pred_path = os.path.join(pred_dir, "predictions.json")

    if os.path.exists(pred_path) and skip_if_found:
        return

    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir)

    centernet = CenterNet(config)
    decoder = Decoder(config)

    weights_path = model_load.get_weights_path(config)
    centernet.load_weights(weights_path)


    prediction_data = {"predictions": []}
    steps = np.sum([1 for i in dataset])


    logger.info("{} ('{}'): Running inference on {} images.".format(config.model_name, 
                                                                    config.instance_name, 
                                                                    dataset_size))

    for step, batch_data in enumerate(tqdm.tqdm(dataset, total=steps, desc="Generating predictions")):

        batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, is_annotated)
        batch_size = batch_images.shape[0]

        pred = centernet(batch_images)
        detections = decoder(pred)

        for i in range(batch_size):

            patch_abs_boxes, patch_scores, patch_classes = post_process_sample(detections, batch_ratios, i)

            patch_info = batch_info[i]

            prediction = {
                "img_path": bytes.decode((patch_info["img_path"]).numpy()),
                "patch_path": bytes.decode((patch_info["patch_path"]).numpy()),
                "patch_coords": tf.sparse.to_dense(patch_info["patch_coords"]).numpy().tolist(),
                "pred_patch_abs_boxes": patch_abs_boxes.tolist(),
                "pred_classes": patch_classes.flatten().tolist(),
                "pred_scores": patch_scores.flatten().tolist()
            }
            if is_annotated:
                # add annotated boxes to facilitate comparison with predictions
                prediction.update({
                    "patch_normalized_boxes": tf.reshape(tf.sparse.to_dense(patch_info["patch_normalized_boxes"]), shape=(-1, 4)).numpy().tolist(),
                    "patch_abs_boxes": tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist(),
                    "img_abs_boxes": tf.reshape(tf.sparse.to_dense(patch_info["img_abs_boxes"]), shape=(-1, 4)).numpy().tolist(),
                    "patch_classes": tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()
                })

            prediction_data["predictions"].append(prediction)


    json_io.save_json(pred_path, prediction_data)




def train(train_patches_dir, val_patches_dir, config):

    logger = logging.getLogger(__name__)

    train_tf_record_path = os.path.join(train_patches_dir, "patches-with-boxes-record.tfrec")
    val_tf_record_path = os.path.join(val_patches_dir, "patches-with-boxes-record.tfrec")

    train_data_loader = data_load.TrainDataLoader(train_tf_record_path, config, shuffle=True, augment=True)
    train_dataset, train_dataset_size = train_data_loader.create_batched_dataset()

    val_data_loader = data_load.TrainDataLoader(val_tf_record_path, config, shuffle=False, augment=False)
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



    train_steps_per_epoch = np.sum([1 for i in train_dataset])
    val_steps_per_epoch = np.sum([1 for i in val_dataset])
    best_val_loss = float("inf")
    epochs_since_improvement = 0


    loss_record = {}

    logger.info("{} ('{}'): Starting to train with {} train images and {} validation images.".format(
                 config.model_name, config.instance_name, train_dataset_size, val_dataset_size))

    for epoch in range(1, config.num_epochs + 1):

        loss_record[str(epoch)] = {}

        train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
        for batch_data in train_bar:
            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
            train_step(batch_images, batch_labels)
            train_bar.set_description("Epoch: {}/{} | training loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs,
                                                                                    train_loss_metric.result()))
        
        loss_record[str(epoch)]["train_loss"] = float(train_loss_metric.result())


        val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
        for batch_data in val_bar:
            batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
            pred = centernet(batch_images, training=False)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
            val_loss_metric.update_state(values=loss_value)
            val_bar.set_description("Epoch: {}/{} | validation loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs,
                                                                                    val_loss_metric.result()))

        loss_record[str(epoch)]["val_loss"] = float(val_loss_metric.result())


        centernet.save_weights(filepath=os.path.join(config.weights_dir, "epoch-{}".format(epoch)), save_format="tf")


        if val_loss_metric.result() < best_val_loss:
            best_val_loss = val_loss_metric.result()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
       
        if config.early_stopping and epochs_since_improvement >= 3:
            break

        train_loss_metric.reset_states()
        val_loss_metric.reset_states()


    loss_record_path = os.path.join(config.model_dir, "loss_record.json")
    json_io.save_json(loss_record_path, loss_record)





def draw_boxes_on_image(image, boxes, scores, classes):
    idx2class_dict = {0: "plant"}
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        cv2.rectangle(img=image, pt1=(boxes[i, 1], boxes[i, 0]), pt2=(boxes[i, 3], boxes[i, 2]), color=(250, 206, 135), thickness=2)

        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 1], boxes[i, 0]), pt2=(boxes[i, 1] + text_width, boxes[i, 0] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 1], boxes[i, 0] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    return image




def test_single_picture(picture_dir, model, config):

    image_array = cv2.imread(picture_dir)
    ratio = np.array(image_array.shape[:2]) / np.array(config.input_img_shape[:2])
    img = tf.image.resize(images=image_array, size=config.input_img_shape)

    img = tf.expand_dims(input=img, axis=0)
    decoder = Decoder(config)

    pred = model(img, training=False)
    batch_boxes, batch_scores, batch_classes = decoder(pred)
    for i in range(batch_boxes.shape[0]):
        boxes, scores, classes = post_process_sample(batch_boxes[i], batch_scores[i], batch_classes[i], ratio, config)

        image_with_boxes = draw_boxes_on_image(image_array, boxes, scores, classes)
        return image_with_boxes



def test_centernet(model, patches_dir, output_dir, config):

    for i, img_path in enumerate(sorted(glob.glob(os.path.join(patches_dir, "*.png")))):
        if i > 25:
            break

        image = test_single_picture(picture_dir=img_path, model=model, config=config)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), image)