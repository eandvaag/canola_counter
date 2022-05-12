import logging
import os
import glob
import math as m
import numpy as np
import tensorflow as tf

from image_set import DataSet
import extract_patches as ep
from io_utils import w3c_io

from models.yolov4 import yolov4

# def create_reg_model(model_name, reg_type, reg_strength):
#     if reg_type == 'l1':
#         regularizer = regularizers.l1(reg_strength)
#     elif reg_type == 'l2':
#         regularizer = regularizers.l2(reg_strength)
#     else:
#         raise RuntimeError("Invalid regularization type")
#     model = models.Sequential(name=model_name)
#     model.add(layers.Conv2D(6, (5, 5), kernel_regularizer=regularizer, 
#      activation='relu', input_shape=(32, 32, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(16, (5, 5), 
#     kernel_regularizer=regularizer, activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(120, kernel_regularizer=regularizer, 
#     activation='relu'))
#     model.add(layers.Dense(84, kernel_regularizer=regularizer, 
#     activation='relu'))
#     model.add(layers.Dense(10, kernel_regularizer=regularizer, 
#     activation='softmax'))
#     return model



def train_model(config):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    #source_construction_params = config.training["source_construction_params"]
    #method_params = source_construction_params["method_params"]
    #patch_size = method_params["patch_size"]
    patch_size = "image_set_dependent"


    image_set_root = os.path.join("usr", "data", "image_sets")
    total_training_set_size = 10000
    datasets = []
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })
                # annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
                # try:
                #     if patch_size == "image_set_dependent":
                #         image_set_patch_size = w3c_io.get_patch_size(annotations)
                #     else:
                #         image_set_patch_size = patch_size
                    
                #     # patch_size can be determined --> use this dataset
                #     datasets.append(dataset)
                # except RuntimeError:
                #     pass

                if len(dataset.completed_images) > 0:
                    datasets.append(dataset)
                

    all_patches = []
    all_labels = []
    extraction_func = ep.extract_patch_records_with_exg_box_combo
    num_per_dataset = m.ceil(total_training_set_size / len(datasets))
    
    logger.info("Found {} datasets".format(len(datasets)))
    #for i, dataset in enumerate(datasets):
    i = 0
    while len(datasets) > 0:
        i += 1
        dataset = datasets[0]
        num_patches_per_image = m.ceil(num_per_dataset / len(dataset.completed_images))
        logger.info("Processing {} | Num per dataset: {} | Num per image: {}".format(
            dataset.image_set_name, num_per_dataset, num_patches_per_image))
        
        annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
        if patch_size == "image_set_dependent":
            image_set_patch_size = w3c_io.get_patch_size(annotations)
        else:
            image_set_patch_size = patch_size
        image_set_patches = []
        for image in dataset.completed_images:
            patches = extraction_func( #ep.extract_patch_records_with_exg(
                            image, 
                            annotations[image.image_name], 
                            num_patches_per_image, 
                            image_set_patch_size)

            for patch in patches:
                patch_array = tf.convert_to_tensor(patch["patch"], dtype=tf.float32)
                patch_array = tf.image.resize(images=patch_array, size=(224, 224)).numpy()
                image_set_patches.append(patch_array)
           #image_set_patches.extend([patch["patch"] for patch in patches])
        #np.random.shuffle(image_set_patches)
        all_patches.extend(image_set_patches[:num_per_dataset])
        all_labels.extend([i] * num_per_dataset)

        print("len(all_patches)", len(all_patches))
        print("len(all_labels)", len(all_labels))
        del image_set_patches
        del datasets[0]
        del dataset


    weights = 'imagenet'
    model = tf.keras.applications.ResNet50(
        weights=weights, #None, #weights,
        include_top=True, 
        input_shape=(224, 224, 3),
    )

    # for i in range(len(all_patches)):
    #     patch = tf.convert_to_tensor(patches[i], dtype=tf.float32)
    #     patch = tf.image.resize(images=patch, size=input_image_shape[:2])


    all_patches = np.array(all_patches)
    all_labels = np.array(all_labels)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="accuracy")
    model.fit(all_patches, all_labels, epochs=50, validation_split=0.2, shuffle=True, verbose=1)



# def train_general_purpose_model(config):
#     patch_size = "image_set_dependent"


#     image_set_root = os.path.join("usr", "data", "image_sets")
#     total_training_set_size = 10000
#     datasets = []
#     for farm_path in glob.glob(os.path.join(image_set_root, "*")):
#         farm_name = os.path.basename(farm_path)
#         for field_path in glob.glob(os.path.join(farm_path, "*")):
#             field_name = os.path.basename(field_path)
#             for mission_path in glob.glob(os.path.join(field_path, "*")):
#                 mission_date = os.path.basename(mission_path)

#                 dataset = DataSet({
#                     "farm_name": farm_name,
#                     "field_name": field_name,
#                     "mission_date": mission_date
#                 })
#                 if len(dataset.completed_images) > 0:
#                     datasets.append(dataset)
    

#     training_patches = []
#     extraction_func = ep.extract_patch_records_with_exg_box_combo
#     num_per_dataset = m.ceil(total_training_set_size / len(datasets))
    
#     logger.info("Found {} datasets".format(len(datasets)))
#     for i, dataset in enumerate(datasets):
#         num_patches_per_image = m.ceil(num_per_dataset / len(dataset.completed_images))
#         logger.info("Processing {} | Num per dataset: {} | Num per image: {}".format(
#             dataset.image_set_name, num_per_dataset, num_patches_per_image))
        
#         annotations = w3c_io.load_annotations(dataset.annotations_path, {"plant": 0})
#         if patch_size == "image_set_dependent":
#             image_set_patch_size = w3c_io.get_patch_size(annotations)
#         else:
#             image_set_patch_size = patch_size
#         #image_set_patches = []
#         for image in dataset.completed_images:
#             patches = extraction_func( #ep.extract_patch_records_with_exg(
#                             image, 
#                             annotations[image.image_name], 
#                             num_patches_per_image, 
#                             image_set_patch_size)

#             training_patches.extend(patches)

#     usr_data_root = os.path.join("usr", "data")
#     patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
#     training_patches_dir = os.path.join(patches_dir, "training")
#     validation_patches_dir = os.path.join(patches_dir, "validation")
#     os.makedirs(training_patches_dir)
#     os.makedirs(validation_patches_dir)

#     training_patches = np.array(training_patches)

#     training_size = round(training_patches.size * 0.8)
#     training_subset = random.sample(np.arange(training_patches.size).tolist(), training_size)

#     training_patches = training_patches[training_subset]
#     validation_patches = np.delete(training_patches, training_subset)

#     logger.info("Writing patches...")
#     ep.write_annotated_patch_records(training_patches, training_patches_dir)
#     ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
#     logger.info("Finished writing patches.")




def get_model(model_name, config):

    if model_name == "resnet_50":
        weights = 'imagenet'
        model = tf.keras.applications.ResNet50( #101( #ResNet50(
            weights=weights,
            include_top=False, 
            input_shape=[None, None, 3],
            pooling="max"
        )
        #c3_output, c4_output, c5_output = [
        #    model.get_layer(layer_name).output
        #    for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
        #]
        return model
    elif model_name == "YOLOv4TinyBackbone":
        model = yolov4.YOLOv4TinyBackbone(config, max_pool=True)
        input_shape = (256, *(config.arch["input_image_shape"]))
        model.build(input_shape=input_shape)
        model.load_weights(os.path.join("weights", "weights.h5"), by_name=False)
        return model

    elif model_name == "YOLOv4TinyBackboneNoPool":
        model = yolov4.YOLOv4TinyBackbone(config, max_pool=False)
        input_shape = (256, *(config.arch["input_image_shape"]))
        model.build(input_shape=input_shape)
        model.load_weights(os.path.join("weights", "weights.h5"), by_name=False)
        return model    



#def get_YOLOv4Tiny_graph_model(config):
#    backbone = yolov4.build_backbone(config)
#    backbone._name = "yolov4_tiny_backbone"


# class DataLoader:

#     def __init__(self, tf_record_paths, input_image_shape, batch_size):
#         self.tf_record_paths = tf_record_paths
#         self.batch_size = batch_size
#         self.input_image_shape = input_image_shape


#     def create_batched_dataset(self):

#         dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)

#         num_images = np.sum([1 for _ in dataset])

#         dataset = dataset.batch(batch_size=self.batch_size)

#         autotune = tf.data.experimental.AUTOTUNE
#         dataset = dataset.prefetch(autotune)

#         return dataset, num_images

#     def read_batch_data(self, batch_data):

#         batch_images = []

#         for tf_sample in batch_data:
#             sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=False)
#             image = self._preprocess(sample)
#             batch_images.append(image)

#         batch_images = tf.stack(values=batch_images, axis=0)
#         #batch_images = tf.ragged.constant(batch_images)
#         #batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
#         return batch_images

#     def _preprocess(self, sample):
#         image_path = bytes.decode((sample["patch_path"]).numpy())
#         image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
#         image = tf.convert_to_tensor(image, dtype=tf.float32)
#         image = tf.image.resize(images=image, size=self.input_image_shape[:2])
#         return image