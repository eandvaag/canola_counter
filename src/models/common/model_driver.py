
class DetectionModel(ABC):

    def get_train_dataloader():
        pass

    def get_inference_dataloader():
        pass

    def get_model():
        pass

    def get_loss():
        pass

    def get_decoder():
        pass




def generate_predictions(patch_dir, pred_dir, is_annotated, config):

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

        batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, is_annotated)
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
        predictions["image_predictions"][img_name]["pred_count"] = nms_boxes.shape[0]


    total_inference_time = np.sum(inference_times)
    per_patch_inference_time = total_inference_time / dataset_size
    per_image_inference_time = total_inference_time / len(predictions["image_predictions"])

    predictions["metrics"] = {}
    predictions["metrics"]["total_inference_time"] = total_inference_time
    predictions["metrics"]["per_patch_inference_time"] = per_patch_inference_time
    predictions["metrics"]["per_image_inference_time"] = per_image_inference_time

    if is_annotated:
        inference_metrics.collect_metrics(predictions, config)

    pred_path = os.path.join(pred_dir, "predictions.json")
    json_io.save_json(pred_path, predictions)




def train(train_patches_dir, val_patches_dir, config):

    logger = logging.getLogger(__name__)

    train_tf_record_path = os.path.join(train_patches_dir, "patches-with-boxes-record.tfrec")
    val_tf_record_path = os.path.join(val_patches_dir, "patches-with-boxes-record.tfrec")

    train_data_loader = data_load.TrainDataLoader(train_tf_record_path, config, shuffle=True, augment=True)
    train_dataset, train_dataset_size = train_data_loader.create_batched_dataset(take_pct=config.pct_of_training_set_used)

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

    for epoch in range(1, config.num_epochs + 1):

        train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
        for batch_data in train_bar:
            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
            train_step(batch_images, batch_labels)
            train_bar.set_description("Epoch: {}/{} | training loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs,
                                                                                    train_loss_metric.result()))

        val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
        for batch_data in val_bar:
            batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
            pred = centernet(batch_images, training=False)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
            val_loss_metric.update_state(values=loss_value)
            val_bar.set_description("Epoch: {}/{} | validation loss: {:.4f}".format(epoch,
                                                                                    config.num_epochs,
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