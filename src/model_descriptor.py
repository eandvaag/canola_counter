import os
import glob
import random
import math as m
import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

from io_utils import json_io
from models.common import annotation_utils, box_utils
import extract_patches as ep


IMAGE_SET_SAMPLE_SIZE = 1 # 1024


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        norm = 1e-20
    # return v
    return v / norm




def create_model_descriptors(baseline_pending_dir):
    annotations_dir = os.path.join(baseline_pending_dir, "annotations")

    log_path = os.path.join(baseline_pending_dir, "log.json")
    log = json_io.load_json(log_path)

    feature_vectors = []

    model, model_input_shape = get_feature_vector_model()

    # batch_size = IMAGE_SET_SAMPLE_SIZE

    for image_set_index in range(len(log["image_sets"])):

        image_set = log["image_sets"][image_set_index]

        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]

        image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                    farm_name, field_name, mission_date)

        # create_image_set_vector(log["image_sets"][image_set_index])


        annotations_path = os.path.join(annotations_dir, 
                                        username, 
                                        farm_name,
                                        field_name,
                                        mission_date,
                                        "annotations.json")

        annotations = annotation_utils.load_annotations(annotations_path)

        image_set_vectors = create_image_set_vectors(model, model_input_shape, 
        image_set_dir, annotations, sample_rate=0.2) #0.01) #, baseline_pending_dir)

        feature_vectors.extend(image_set_vectors)

    feature_vectors_path = os.path.join(baseline_pending_dir, "feature_vectors.npy")
    np.save(feature_vectors_path, np.array(feature_vectors))
    # json_io.save_json(feature_vectors_path, feature_vectors)


def get_feature_vector_model():
    model_input_shape = np.array([416, 416, 3]) #[150, 150, 3]) #150, 150, 3])
    weights = 'imagenet'
    # model = tf.keras.applications.ResNet50( #101( #ResNet50(
    #     weights=weights,
    #     include_top=False, 
    #     input_shape=[None, None, 3],
    #     pooling="max"
    # )
    b_model = tf.keras.applications.VGG16( #101( #ResNet50(
        weights=weights,
        include_top=False,
        input_shape=[None, None, 3],
        pooling="max" #"max"
    )
    # new_top_layer = tf.keras.layers.GlobalMaxPooling2D()(b_model.get_layer('block5_pool').output)
    # model = tf.keras.models.Model(inputs=b_model.input, outputs=new_top_layer)

    # f = tf.keras.layers.GlobalMaxPooling2D()(f)
    
    return b_model, model_input_shape


def get_sample_box_patches(image_set_dir, annotations, sample_rate=0.01, max_num_samples=None):

    if sample_rate > 1 or sample_rate < 0:
        raise RuntimeError("Illegal sample rate specified")

    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    is_ortho = metadata["is_ortho"] == "yes"

    image_set_candidates = np.array([])
    image_names = list(annotations.keys())
    for image_index, image_name in enumerate(image_names):
        boxes = annotations[image_name]["boxes"]
        regions = []
        for region_key in ["training_regions", "test_regions"]:
            regions.extend(annotations[image_name][region_key])
        # regions = annotations[image_name]["training_regions"] + annotations[image_name]["test_regions"]
        inds = box_utils.get_contained_inds(boxes, regions)
        region_boxes = boxes[inds]
        image_index_array = np.full(inds.size, image_index)

        if region_boxes.size > 0:

            # image_candidates = np.stack([image_index_array, region_boxes], axis=-1)
            image_candidates = np.concatenate([np.expand_dims(image_index_array, axis=1), region_boxes], axis=1)

            if image_set_candidates.size == 0:
                image_set_candidates = image_candidates
            else:
                image_set_candidates = np.concatenate([image_set_candidates, image_candidates])
    
    num_samples = max(1, round(image_set_candidates.shape[0] * sample_rate))
    if max_num_samples is not None:
        if max_num_samples < 1:
            raise RuntimeError("max_num_samples must be greater than or equal to 1.")
        num_samples = min(max_num_samples, num_samples)

    sample_inds = random.sample(range(image_set_candidates.shape[0]), num_samples)
    # min(image_set_candidates.shape[0], IMAGE_SET_SAMPLE_SIZE))

    # for i in range(0, IMAGE_SET_SAMPLE_SIZE):
    sample_candidates = image_set_candidates[sample_inds]
    image_indices = np.unique(sample_candidates[:, 0])

    image_set_box_arrays = []
    for image_index in image_indices:

        image_name = image_names[image_index]
        sample_image_candidate_inds = np.where(sample_candidates[:, 0] == image_index)[0]
        sample_image_candidates = sample_candidates[sample_image_candidate_inds][:, 1:]
        image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]

        # image = Image(image)
        # w, h = image.get_wh()

        patch_size = annotation_utils.get_patch_size(annotations, None)
        # image_box_arrays = ep.extract_random_patches(image_path, 1, patch_size, is_ortho) #sample_image_candidates.size, patch_size, is_ortho)


        image_box_arrays = ep.extract_box_patches(
            image_path, 
            sample_image_candidates, 
            patch_size,
            is_ortho
        )

        image_set_box_arrays.extend(image_box_arrays)

    return image_set_box_arrays


def create_image_set_vector_2(model, model_input_shape, image_set_dir, annotations):

    average_box_height = annotation_utils.get_average_box_height(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
    average_box_width = annotation_utils.get_average_box_width(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
    std_box_width = annotation_utils.get_average_box_height(annotations, region_keys=["training_regions", "test_regions"], measure="std")
    std_box_height = annotation_utils.get_average_box_width(annotations, region_keys=["training_regions", "test_regions"], measure="std")
    

    # image_set_box_arrays = get_sample_box_patches(image_set_dir, annotations)
    # # image_set_box_arrays = np.array(image_set_box_arrays)
    # # print(image_set_box_arrays.shape)
    # r = []
    # g = []
    # b = []
    # for box_array in image_set_box_arrays:
    #     r.append(np.percentile(np.array(box_array[:,:,0]), [0, 25, 50, 75, 100]).tolist())
    #     g.append(np.percentile(np.array(box_array[:,:,1]), [0, 25, 50, 75, 100]).tolist())
    #     b.append(np.percentile(np.array(box_array[:,:,2]), [0, 25, 50, 75, 100]).tolist())

    # r = np.mean(np.array(r), axis=0).tolist()
    # g = np.mean(np.array(g), axis=0).tolist()
    # b = np.mean(np.array(b), axis=0).tolist()

    feature_vector = [average_box_height, average_box_width, std_box_width, std_box_height]
    # feature_vector.extend(r)
    # feature_vector.extend(g)
    # feature_vector.extend(b)

    return feature_vector


def create_image_set_vectors(model, model_input_shape, image_set_dir, annotations, 
                             sample_rate=0.01, max_num_samples=None): #, model_dir):
    # username = image_set["username"]
    # farm_name = image_set["farm_name"]
    # field_name = image_set["field_name"]
    # mission_date = image_set["mission_date"]

    # average_box_height = image_set["average_box_height"]
    # average_box_width = image_set["average_box_width"]

    # username = image_set["username"]
    # farm_name = image_set["farm_name"]
    # field_name = image_set["field_name"]
    # mission_date = image_set["mission_date"]





    # image_set_annotations_dir = os.path.join(annotations_dir, 
    #                                 username, 
    #                                 farm_name,
    #                                 field_name,
    #                                 mission_date)
    # image_set_annotations_path = os.path.join(image_set_annotations_dir, "annotations.json")
    # annotations = annotation_utils.load_annotations(image_set_annotations_path)



    image_set_box_arrays = get_sample_box_patches(image_set_dir, annotations, 
                            sample_rate=sample_rate, max_num_samples=max_num_samples)
    # sample_dir = os.path.join(model_dir, "samples")
    # os.makedirs(sample_dir)
    # for i, box_array in enumerate(image_set_box_arrays):
    #     out_path = os.path.join(sample_dir, str(i) + ".png")
    #     cv2.imwrite(out_path, cv2.cvtColor(box_array, cv2.COLOR_RGB2BGR))


    num_box_arrays = len(image_set_box_arrays)
    features_lst = []
    batch_size = 256 #1024
    for i in range(0, num_box_arrays, batch_size):
        batch_patches = []
        # batch_patches = []
        for j in range(i, min(i+batch_size, num_box_arrays)):
            patch = tf.convert_to_tensor(image_set_box_arrays[j], dtype=tf.float32)
            patch = tf.image.resize(images=patch, size=model_input_shape[:2])
            batch_patches.append(patch)
        batch_patches = tf.stack(values=batch_patches, axis=0)


        batch_patches = tf.keras.applications.vgg16.preprocess_input(batch_patches)
        features = model.predict(batch_patches)
        # max_pool_2d = 
        for j, f in enumerate(features):
            # print(f.shape)
            # # f = np.apply_over_axes(np.sum, f, [0, 1]).flatten()
            
            # print(f.shape)
            # exit()
            # f = f.flatten()

            patch = image_set_box_arrays[i+j]
            height = patch.shape[0]
            width = patch.shape[1]
            ratio = max(height, width) / min(height, width)

            f = normalize(f)

            feature = [] #[height, width, ratio]
            feature.extend(f.tolist())

            features_lst.append(feature)

    return features_lst

    # mean_feature_vector = np.mean(np.array(features_lst), axis=0)

    # return mean_feature_vector.tolist()



def recreate_all_model_descriptors():
    for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
        for model_dir in glob.glob(os.path.join(usr_dir, "models", "available", "public", "*")):
            print("recreating descriptor for", model_dir)
            create_model_descriptors(model_dir)

        for model_dir in glob.glob(os.path.join(usr_dir, "models", "available", "private", "*")):
            print("recreating descriptor for", model_dir)
            create_model_descriptors(model_dir)