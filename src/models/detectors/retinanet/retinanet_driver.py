import tensorflow as tf
import numpy as np
import time
import os
import shutil
import tqdm
import logging

import extract_patches as ep

import models.detectors.common.box_utils as box_utils
import models.detectors.common.model_load as model_load

from models.detectors.retinanet.loss import RetinaNetLoss
from models.detectors.retinanet.retinanet import RetinaNet
import models.detectors.retinanet.data_load as data_load
from models.detectors.retinanet.encode import Decoder


from io_utils import json_io
from io_utils import tf_record_io





def post_process_sample(detections, resize_ratios, sample_index):

    num_detections = detections.valid_detections[sample_index]
    classes = detections.nmsed_classes[sample_index][:num_detections]
    scores = detections.nmsed_scores[sample_index][:num_detections]
    boxes = detections.nmsed_boxes[sample_index][:num_detections]

    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()

    boxes = box_utils.swap_xy_np(boxes)
    boxes = np.stack([
        boxes[:, 0] * resize_ratios[sample_index][0],
        boxes[:, 1] * resize_ratios[sample_index][1],
        boxes[:, 2] * resize_ratios[sample_index][0],
        boxes[:, 3] * resize_ratios[sample_index][1]
    ], axis=-1)

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

    retinanet = RetinaNet(config)
    decoder = Decoder(config)

    weights_path = model_load.get_weights_path(config)
    retinanet.load_weights(weights_path)


    prediction_data = {"predictions": []}
    steps = np.sum([1 for i in dataset])


    logger.info("{} ('{}'): Running inference on {} images.".format(config.model_name, 
                                                                    config.instance_name, 
                                                                    dataset_size))

    for step, batch_data in enumerate(tqdm.tqdm(dataset, total=steps, desc="Generating predictions")):

        batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, is_annotated)
        batch_size = batch_images.shape[0]

        pred = retinanet(batch_images)
        detections = decoder(batch_images, pred)

        for i in range(batch_size):

            patch_abs_boxes, patch_scores, patch_classes = post_process_sample(detections, batch_ratios, i)

            patch_info = batch_info[i]

            prediction = {
                "img_path": bytes.decode((patch_info["img_path"]).numpy()),
                "patch_path": bytes.decode((patch_info["patch_path"]).numpy()),
                "patch_coords": tf.sparse.to_dense(patch_info["patch_coords"]).numpy().tolist(),
                "pred_patch_abs_boxes": patch_abs_boxes.tolist(),
                "pred_classes": patch_classes.tolist(),
                "pred_scores": patch_scores.tolist()
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




def train_with_keras(train_patches_dir, val_patches_dir, config):
    
    train_tf_record_path = os.path.join(train_patches_dir, "patches-with-boxes-record.tfrec")
    val_tf_record_path = os.path.join(val_patches_dir, "patches-with-boxes-record.tfrec")

    retinanet = RetinaNet(config)

    weights_dir = os.path.join(model_dir, "weights")

    loss_fn = RetinaNetLoss(config.num_classes)
    optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)

    train_dataset, train_dataset_size = data_load.load_all_data(train_tf_record_path, config)
    val_dataset, val_dataset_size = data_load.load_all_data(val_tf_record_path, config)

    train_dataset = data_load.prepare_dataset_for_training(train_dataset, config, shuffle=True, augment=True)
    val_dataset = data_load.prepare_dataset_for_training(val_dataset, config, shuffle=False, augment=False)

    train_steps_per_epoch = train_dataset_size // config.train_batch_size
    val_steps_per_epoch = val_dataset_size // config.train_batch_size

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(weights_dir, "epoch-{epoch}"),
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=3, restore_best_weights=True)
    ]

    retinanet.compile(loss=loss_fn, optimizer=optimizer)
    retinanet.fit(train_dataset,
              validation_data=val_dataset,
              batch_size=config.train_batch_size,
              epochs=config.num_epochs,
              callbacks=callbacks,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=val_steps_per_epoch,
              verbose=1)



def train(train_patches_dir, val_patches_dir, config):

    logger = logging.getLogger(__name__)

    train_tf_record_path = os.path.join(train_patches_dir, "patches-with-boxes-record.tfrec")
    val_tf_record_path = os.path.join(val_patches_dir, "patches-with-boxes-record.tfrec")

    train_data_loader = data_load.TrainDataLoader(train_tf_record_path, config, shuffle=True, augment=True)
    train_dataset, train_dataset_size = train_data_loader.create_batched_dataset()

    val_data_loader = data_load.TrainDataLoader(val_tf_record_path, config, shuffle=False, augment=False)
    val_dataset, val_dataset_size = val_data_loader.create_batched_dataset()

    retinanet = RetinaNet(config)

    loss_fn = RetinaNetLoss(config.num_classes)
    optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)


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
            pred = retinanet(batch_images, training=False)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
            val_loss_metric.update_state(values=loss_value)
            val_bar.set_description("Epoch: {}/{} | validation loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs,
                                                                                    val_loss_metric.result()))

        loss_record[str(epoch)]["val_loss"] = float(val_loss_metric.result())


        retinanet.save_weights(filepath=os.path.join(config.weights_dir, "epoch-{}".format(epoch)), save_format="tf")


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