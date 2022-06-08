
import os
import shutil
import glob
import tqdm
import logging
import numpy as np
import tensorflow as tf
import uuid


from io_utils import w3c_io, json_io
from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          model_keys

from models.yolov4.loss import YOLOv4Loss
from models.yolov4.yolov4 import YOLOv4, YOLOv4Tiny
import models.yolov4.data_load as data_load
from models.yolov4.encode import Decoder
from models.yolov4.yolov4_driver import post_process_sample

VALIDATION_IMPROVEMENT_TOLERANCE = 10

def create_default_config():


    config = {
        "model_name": "model_1",
        "model_uuid": str(uuid.uuid4()),
        "arch": {
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

        "training": {

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

            #"save_method": "best_validation_loss",
            "percent_of_training_set_used": 100,
            "percent_of_validation_set_used": 100
        },
        "inference": {
            "batch_size": 16,
            "patch_nms_iou_thresh": 0.4,
            "image_nms_iou_thresh": 0.4,
            "score_thresh": 0.5,
            #"predict_on_completed_only": False    

        }
    }

    return config


def update_loss_record(loss_record, key, cur_loss):

    loss_record[key]["values"].append(cur_loss)

    best = loss_record[key]["best"]
    if cur_loss < best:
        loss_record[key]["best"] = cur_loss
        loss_record[key]["epochs_since_improvement"] = 0
        return True
    else:
        loss_record[key]["epochs_since_improvement"] += 1
        return False






def predict(farm_name, field_name, mission_date, image_names):

    logger = logging.getLogger(__name__)

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    predictions_dir = os.path.join(model_dir, "prediction")

    #annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    #annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})


    config = create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    yolov4 = YOLOv4Tiny(config)
    decoder = Decoder(config)

    image_predictions = {}

    for image_name in image_names:

        image_predictions_dir = os.path.join(predictions_dir, "images", image_name)
        tf_record_path = os.path.join(image_predictions_dir, "patches-record.tfrec")

        data_loader = data_load.InferenceDataLoader(tf_record_path, config)
        tf_dataset, tf_dataset_size = data_loader.create_dataset()

        input_shape = (config["inference"]["batch_size"], *(data_loader.get_model_input_shape()))
        print("input_shape", input_shape)
        yolov4.build(input_shape=input_shape)

        best_weights_path = os.path.join(weights_dir, "best_weights.h5")
        yolov4.load_weights(best_weights_path, by_name=False)



        steps = np.sum([1 for _ in tf_dataset])

        logger.info("{} ('{}'): Running inference on {} images.".format(config["arch"]["model_type"], 
                                                                        config["model_name"], 
                                                                        tf_dataset_size))


        # ISSUE: problem if image annotation status is changed 

        #is_annotated = annotations[image_name]["status"] == "completed"




        for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

            batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, False) #is_annotated)
            batch_size = batch_images.shape[0]

            #print("batch_images.shape", batch_images.shape)


            pred = yolov4(batch_images, training=False)
            detections = decoder(pred)

            batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

            batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

            for i in range(batch_size):

                pred_bbox = batch_pred_bbox[i]
                ratio = batch_ratios[i]

                patch_info = batch_info[i]

                image_path = bytes.decode((patch_info["image_path"]).numpy())
                #patch_path = bytes.decode((patch_info["patch_path"]).numpy())
                image_name = os.path.basename(image_path)[:-4]
                #patch_name = os.path.basename(patch_path)[:-4]
                patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

                #if is_annotated:
                #    patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                #    patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()


                pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
                        post_process_sample(pred_bbox, ratio, patch_coords, config)


                if image_name not in image_predictions:
                    image_predictions[image_name] = {
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
                                                    trim=True)
                                                    

                image_predictions[image_name]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
                image_predictions[image_name]["pred_scores"].extend(pred_image_scores.tolist())
                image_predictions[image_name]["pred_classes"].extend(pred_image_classes.tolist())
                image_predictions[image_name]["patch_coords"].append(patch_coords.tolist())
                    

        driver_utils.clip_image_boxes(image_predictions)

        driver_utils.apply_nms_to_image_boxes(image_predictions, 
                                             iou_thresh=config["inference"]["image_nms_iou_thresh"])

        driver_utils.add_class_detections(image_predictions, config)

        #metrics = driver_utils.create_metrics_skeleton(dataset)

        #all_image_names = predictions["image_predictions"].keys()
        #inference_metrics.collect_statistics(all_image_names, metrics, predictions, config, inference_times=inference_times)
        #inference_metrics.collect_metrics(all_image_names, metrics, predictions, dataset, config)

        for image_name in image_names:
            image_predictions_dir = os.path.join(predictions_dir, "images", image_name)
            predictions_w3c_path = os.path.join(image_predictions_dir, "predictions_w3c.json")
            w3c_io.save_annotations(predictions_w3c_path, {image_name: image_predictions[image_name]}, config)





def train(farm_name, field_name, mission_date):
    
    logger = logging.getLogger(__name__)

    tf.keras.backend.clear_session()

    image_set_dir = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    weights_dir = os.path.join(model_dir, "weights")
    training_dir = os.path.join(model_dir, "training")


    #training_patch_dir = os.path.join(training_patches_dir, str(seq_num), "training")
    #validation_patch_dir = os.path.join(training_patches_dir, str(seq_num), "validation")

    training_tf_record_paths = [os.path.join(training_dir, "training-patches-record.tfrec")]
    validation_tf_record_paths = [os.path.join(training_dir, "validation-patches-record.tfrec")]

    config = create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    config["training"]["active"] = {}
    for k in config["training"]:
        config["training"]["active"][k] = config["training"][k]


    train_loader_is_preloaded, train_data_loader = data_load.get_data_loader(training_tf_record_paths, config, shuffle=True, augment=True)
    val_loader_is_preloaded, val_data_loader = data_load.get_data_loader(validation_tf_record_paths, config, shuffle=False, augment=False)
    

    logger.info("Data loaders created. Train loader is preloaded?: {}. Validation loader is preloaded?: {}.".format(
        train_loader_is_preloaded, val_loader_is_preloaded
    ))

    train_dataset, num_train_images = train_data_loader.create_batched_dataset(
                                      take_percent=config["training"]["active"]["percent_of_training_set_used"])

    val_dataset, num_val_images = val_data_loader.create_batched_dataset(
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
    yolov4.load_weights(cur_weights_path, by_name=False)


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

    logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
                    config["arch"]["model_type"], config["model_name"], num_train_images, num_val_images))


    loss_record_path = os.path.join(training_dir, "loss_record.json")
    loss_record = json_io.load_json(loss_record_path)


    steps_taken = 0
    while True:
        logger.info("Epochs since validation loss improvement: {}".format(loss_record["validation_loss"]["epochs_since_improvement"]))

        train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
        for batch_data in train_bar:

            optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))

            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)

            train_step(batch_images, batch_labels)
            if np.isnan(train_loss_metric.result()):
                raise RuntimeError("NaN loss has occurred (training dataset).")
            train_bar.set_description("t. loss: {:.4f} | best: {:.4f}".format(
                                        train_loss_metric.result(), 
                                        loss_record["training_loss"]["best"]))
            steps_taken += 1


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

        
        cur_validation_loss = float(val_loss_metric.result())

        cur_validation_loss_is_best = update_loss_record(loss_record, "validation_loss", cur_validation_loss)
        yolov4.save_weights(filepath=cur_weights_path, save_format="h5")
        if cur_validation_loss_is_best:
            yolov4.save_weights(filepath=best_weights_path, save_format="h5")

            #model_io.save_model_weights(yolov4, config, seq_num, epoch)    


        json_io.save_json(loss_record_path, loss_record)
        
        val_loss_metric.reset_states()

        if loss_record["validation_loss"]["epochs_since_improvement"] >= VALIDATION_IMPROVEMENT_TOLERANCE:
            shutil.copyfile(best_weights_path, cur_weights_path)
            return True

        prediction_requests_dir = os.path.join("usr", "data", "prediction_requests")
        prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))

        if len(prediction_request_paths) > 0:
            return False



