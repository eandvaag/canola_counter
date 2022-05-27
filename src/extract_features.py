import os
import glob
import logging
import tqdm

import numpy as np
import tensorflow as tf
from fast_histogram import histogram1d


import extract_patches as ep
from image_set import Image

from graph import graph_model
from io_utils import w3c_io, json_io



def tmp_del():
    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                features_dir = os.path.join(mission_path, "features")
                features_path = os.path.join(features_dir, "features.npy")
                if os.path.exists(features_path):
                    os.remove(features_path)
                coords_path = os.path.join(features_dir, "patch_coords.json")
                if os.path.exists(coords_path):
                    os.remove(coords_path)




def load_features(farm_name, field_name, mission_date, include_coords=True, completed_only=False):
    image_set_dir = os.path.join("usr", "data", "image_sets", 
                                             farm_name, field_name, mission_date)
    features_dir = os.path.join(image_set_dir, "features") #, "features.npy")
    #if not os.path.exists(features_path):
    #    raise RuntimeError("Features path does not exist")

    annotations_path = os.path.join("usr", "data", "image_sets", 
                            farm_name, field_name, mission_date, 
                            "annotations", "annotations_w3c.json")

    annotations = json_io.load_json(annotations_path)

    coords = {
        "patch_coords": [],
        "image_names": []
    }
    image_set_features = []
    for features_path in glob.glob(os.path.join(features_dir, "*.npy")):
        image_name = os.path.basename(features_path)[:-4]
        if not completed_only or annotations[image_name]["status"] == "completed":
            coords_path = features_path[:-4] + "_patch_coords.json"
            image_features = np.load(features_path)
            if include_coords:
                image_patch_coords = json_io.load_json(coords_path)
                coords["patch_coords"].extend(image_patch_coords)
                coords["image_names"].extend([image_name for _ in range(image_features.shape[0])])
        
            image_set_features.extend(image_features.tolist())

    image_set_features = np.array(image_set_features)
    #features = np.load(features_path)
    if include_coords:
        #coords_path = os.path.join(image_set_dir, "features", "patch_coords.json")
        #coords = json_io.load_json(coords_path)
        return image_set_features, coords
    else:
        return image_set_features


def load_all_features(nonzero_only=True, omit=[], completed_only=False):
    all_features = {}
    all_coords = {}

    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                image_set_tup = (farm_name, field_name, mission_date)
                if image_set_tup not in omit:
                    f, c = load_features(farm_name, field_name, mission_date, 
                                  include_coords=True, completed_only=completed_only)
                    if f.size > 0 or not nonzero_only:
                        all_features[image_set_tup] = f
                        all_coords[image_set_tup] = c
                else:
                    print("omitting", image_set_tup)
    return all_features, all_coords
                    
                    






def extract_patch_features(patches, config):
    logger = logging.getLogger(__name__)

    logger.info("Extracting patch features...")
    all_features = []

    # manual = False
    # if manual:
    #     raise RuntimeError("unimplemented")
    # else:
    model = graph_model.get_model("YOLOv4TinyBackbone", config)

    batch_size = 256

    patches = np.array(patches)
    num_patches = patches.size

    # if extraction_type == "box_patches":
    #     input_image_shape = np.array([150, 150, 3])
    # else:
    input_image_shape = config.arch["input_image_shape"]


    for i in tqdm.trange(0, num_patches, batch_size):
        batch_patches = []
        for j in range(i, min(num_patches, i+batch_size)):
            patch = tf.convert_to_tensor(patches[j]["patch"], dtype=tf.float32)
            patch = tf.image.resize(images=patch, size=input_image_shape[:2])
            batch_patches.append(patch)
        batch_patches = tf.stack(values=batch_patches, axis=0)
        
        features = model.predict(batch_patches)
        for f in features:
            f = f.flatten()
            #if i == 0:
            #    print("f.shape: {}".format(f.shape))
            all_features.append(f)

    logger.info("Finished extracting patch features.")

    return np.array(all_features)





def build_feature_vectors(config):

    logger = logging.getLogger(__name__)
    image_set_root = os.path.join("usr", "data", "image_sets")

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                image_set_dir = os.path.join("usr", "data", "image_sets", 
                                             farm_name, field_name, mission_date)



                features_dir = os.path.join(image_set_dir, "features")

                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                    logger.info("Now processing {} {} {}".format(farm_name, field_name, mission_date))


                    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")

                    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

                    
                    # is_target = farm_name == target_farm_name and \
                    #             field_name == target_field_name and \
                    #             mission_date == target_mission_date

                    # if is_target:           
                    #     image_names = list(annotations.keys())
                    #     #max_size = TARGET_MAX_DATASET_SIZE
                    # else:
                    #     image_names = w3c_io.get_completed_images(annotations)
                    #     #max_size = MAX_DATASET_SIZE


                    image_names = list(annotations.keys())


                    # image_set_features = []
                    # image_set_coords = {
                    #     "patch_coords": [],
                    #     "image_names": []
                    # }

                    images_dir = os.path.join(image_set_dir, "images")

                    try:
                        #if patch_size == "image_set_dependent":
                        image_set_patch_size = w3c_io.get_patch_size(annotations)
                        #else:
                        #    image_set_patch_size = patch_size


                        for image_name in image_names:

                            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
                            image = Image(image_path)

                            patches = ep.extract_patch_records_from_image_tiled(
                                image, 
                                image_set_patch_size, 
                                annotations[image_name],
                                patch_overlap_percent=50)

                            #image_features = extract_patch_features(patches, config)
                            image_features = extract_patch_features_hist(patches, config)
                            #image_set_features.extend(image_features.tolist())
                            #image_set_coords["patch_coords"].extend([patch["patch_coords"] for patch in patches])
                            #image_set_coords["image_names"].extend([image_name for _ in patches])
                            patch_coords = [patch["patch_coords"] for patch in patches]

                            features_path = os.path.join(features_dir, image_name + ".npy")
                            patch_coords_path = os.path.join(features_dir, image_name + "_patch_coords.json")
                            np.save(features_path, image_features)
                            json_io.save_json(patch_coords_path, patch_coords)
                        

                    except RuntimeError:
                        pass
                        #raise RuntimeError("Need annotations in image set to determine image set dependent patch size.")


                    # image_set_features = np.array(image_set_features)



                    # if image_set_features.shape[0] > max_size:
                    #     inds = np.random.choice(image_set_features.shape[0], MAX_DATASET_SIZE, replace=False)
                    #     image_set_features = image_set_features[inds]
                    #     image_set_coords["patch_coords"] = (np.array(image_set_coords["patch_coords"])[inds]).tolist()
                    #     image_set_coords["image_names"] = (np.array(image_set_coords["image_names"])[inds]).tolist()


                    # features_path = os.path.join(features_dir, "features.npy")
                    # patch_coords_path = os.path.join(features_dir, "patch_coords.json")
                    # np.save(features_path, image_set_features)
                    # json_io.save_json(patch_coords_path, image_set_coords)




def build_image_feature_vectors(config):
    pass


def build_target_feature_vectors(config, completed_only=False):

    logger = logging.getLogger(__name__)
    image_set_root = os.path.join("usr", "data", "image_sets")

    farm_name = config.arch["target_farm_name"]
    field_name = config.arch["target_field_name"]
    mission_date = config.arch["target_mission_date"]


    logger.info("Now processing {} {} {}".format(farm_name, field_name, mission_date))

    image_set_dir = os.path.join("usr", "data", "image_sets", 
                                farm_name, field_name, mission_date)
    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")

    annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    if completed_only:
        image_names = w3c_io.get_completed_images(annotations)
    else:
        image_names = list(annotations.keys())

    image_set_features = []
    image_set_coords = {
        "patch_coords": [],
        "image_names": []
    }
    features_dir = os.path.join(image_set_dir, "features")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    images_dir = os.path.join(image_set_dir, "images")
    try:
        #if patch_size == "image_set_dependent":
        image_set_patch_size = w3c_io.get_patch_size(annotations)
        #else:
        #    image_set_patch_size = patch_size


        for image_name in image_names:

            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = Image(image_path)

            patches = ep.extract_patch_records_from_image_tiled(
                image, 
                image_set_patch_size, 
                annotations[image_name],
                patch_overlap_percent=50)

            #image_features = extract_patch_features(patches, config)
            image_features = extract_patch_features_hist(patches, config)
            image_set_features.extend(image_features.tolist())
            image_set_coords["patch_coords"].extend([patch["patch_coords"] for patch in patches])
            image_set_coords["image_names"].extend([image_name for _ in patches])
        

    except RuntimeError:
        raise RuntimeError("Need annotations in image set to determine image set dependent patch size.")


    image_set_features = np.array(image_set_features)



    # if image_set_features.shape[0] > max_size:
    #     inds = np.random.choice(image_set_features.shape[0], MAX_DATASET_SIZE, replace=False)
    #     image_set_features = image_set_features[inds]
    #     image_set_coords["patch_coords"] = (np.array(image_set_coords["patch_coords"])[inds]).tolist()
    #     image_set_coords["image_names"] = (np.array(image_set_coords["image_names"])[inds]).tolist()


    features_path = os.path.join(features_dir, "features.npy")
    patch_coords_path = os.path.join(features_dir, "patch_coords.json")
    np.save(features_path, image_set_features)
    json_io.save_json(patch_coords_path, image_set_coords)






def extract_patch_features_hist(patches, config):
    logger = logging.getLogger(__name__)

    logger.info("Extracting patch features...")
    all_features = []

    # manual = False
    # if manual:
    #     raise RuntimeError("unimplemented")
    # else:
    #model = graph_model.get_model("YOLOv4TinyBackbone", config)
    model = graph_model.get_model("YOLOv4TinyBackboneNoPool", config)

    batch_size = 256

    patches = np.array(patches)
    num_patches = patches.size

    # if extraction_type == "box_patches":
    #     input_image_shape = np.array([150, 150, 3])
    # else:
    input_image_shape = config.arch["input_image_shape"]


    max_vals = np.full((512, 1), -np.inf)
    min_vals = np.full((512, 1), np.inf)

    for i in tqdm.trange(0, num_patches, batch_size):
        batch_patches = []
        for j in range(i, min(num_patches, i+batch_size)):
            patch = tf.convert_to_tensor(patches[j]["patch"], dtype=tf.float32)
            patch = tf.image.resize(images=patch, size=input_image_shape[:2])
            batch_patches.append(patch)
        batch_patches = tf.stack(values=batch_patches, axis=0)
        
        features = model.predict(batch_patches)
        #print(features[0].shape)

        # res = Parallel(os.cpu_count())(delayed(hist_calc)(f) for f in features)
        # all_features.extend(res)
        for f in features:
            h = np.array([])
            #print(f.shape)
            #exit()

            for z in range(0, f.shape[2]):
            #     fmax = np.max(f[:,:,i])
            #     fmin = np.min(f[:,:,i])
            #     if fmax > max_vals[i]:
            #         max_vals[i] = fmax
            #     if fmin < min_vals[i]:
            #         min_vals[i] = fmin

                #h = np.concatenate([h, np.histogram(f[:,:,z].flatten(), range=(-1, 10))[0]])
                h = np.concatenate([h, histogram1d(f[:,:,z].flatten(), range=(-1, 10), bins=5)])

            #f = f.flatten()
            #if i == 0:
            #    print("f.shape: {}".format(f.shape))
            
            
            # res = Parallel(10)(delayed(hist_calc)(ind, f[:,:,ind]) for ind in range(f.shape[2]))

            # res.sort(key=lambda x: x[0])
            # for r in res:
            #     h = np.concatenate([h, r[1]])


            #np.concatenate([h, res[1][ind]])


            all_features.append(h)
        


    logger.info("Finished extracting patch features.")
    # print("min_vals", min_vals)
    # print("max_vals", max_vals)
    # exit()

    return np.array(all_features)


# def hist_calc(f):
#     h = np.array([])
#     for z in range(0, f.shape[2]):
#         h = np.concatenate([h, histogram1d(f[:,:,z].flatten(), range=(-1, 10), bins=10)])
#     return f