import logging
import os
import glob
import time
import numpy as np
from sklearn.neighbors import KDTree

import model_descriptor
from io_utils import json_io
from models.common import annotation_utils

def get_image_set_match_score(target_vectors, source_vectors):

    all_vectors = np.concatenate([target_vectors, source_vectors])
    vectors_mean = np.mean(all_vectors, axis=0)
    vectors_std = np.std(all_vectors, axis=0)

    valid_cols = vectors_std != 0
    if np.all(np.logical_not(valid_cols)):
        raise RuntimeError("Unable to standardize feature vectors.")
    all_vectors = all_vectors[:, valid_cols]

    all_vectors = np.divide((all_vectors - vectors_mean), vectors_std)

    target_vectors = all_vectors[:target_vectors.shape[0], :]
    source_vectors = all_vectors[target_vectors.shape[0]:, :]

    score = 0
    tree = KDTree(source_vectors, leaf_size=2, metric='l1')
    for i in range(target_vectors.shape[0]):
        query_vector = target_vectors[i]
        dists, inds = tree.query([query_vector], k=1) #candidate_vectors.shape[0]) #3)
        # print("dists", dists)
        dist = dists[0][0]
        score += (dist ** 2)
    return score


def auto_select_model(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]
    object_name = item["object_name"]

    logger.info("Creating target image set vector...")
    start_time = time.time()

    model, model_input_shape = model_descriptor.get_feature_vector_model()
    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)
    target_vectors = model_descriptor.create_image_set_vectors(model, model_input_shape, image_set_dir, annotations, sample_rate=1.0)
    target_vectors = np.array(target_vectors)
    # print("target_vectors.shape", target_vectors.shape)
    # print("target_vectors", target_vectors)

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("Finished creating target image set vector. Took {} seconds.".format(elapsed))
    
    vector_ind_to_model = {}
    candidate_vectors = []
    best_score = 1e10
    best_match = None
    vector_ind = 0
    
    logger.info("Fetching available models...")
    start_time = time.time()
    for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
        public_models_dir = os.path.join(usr_dir, "models", "available", "public")
        for model_dir in glob.glob(os.path.join(public_models_dir, "*")):
            log_path = os.path.join(model_dir, "log.json")
            log = json_io.load_json(log_path)
            if log["model_object"] in ["canola_seedling", "wheat_head"]: #== object_name:
                valid = True
                # for image_set in log["image_sets"]:

                #     if image_set["username"] == username and \
                #         image_set["farm_name"] == farm_name and \
                #         image_set["field_name"] == field_name and \
                #         image_set["mission_date"] == mission_date:

                #         valid = False
                #         break

                if valid:
                    feature_vectors_path = os.path.join(model_dir, "feature_vectors.json")
                    feature_vectors = json_io.load_json(feature_vectors_path)

                    score = get_image_set_match_score(target_vectors, np.array(feature_vectors["feature_vectors"]))
                    
                    # for image_set in log["image_sets"]:
                    #     print("valid: {}-{}-{}-{}".format(image_set["username"], 
                    #             image_set["farm_name"], image_set["field_name"], image_set["mission_date"]))


                    #     feature_vectors_path = os.path.join(model_dir, "feature_vectors.json")
                    #     feature_vectors = json_io.load_json(feature_vectors_path)
                    #     # for feature_vector in feature_vectors:
                    #     for i in range(len(feature_vectors["feature_vectors"])):
                    #         vector_ind_to_model[vector_ind] = {
                    #             "username": image_set["username"],
                    #             "farm_name": image_set["farm_name"],
                    #             "field_name": image_set["field_name"],
                    #             "mission_date": image_set["mission_date"]
                    #         }
                    #         candidate_vectors.append(feature_vectors["feature_vectors"][i])
                    #         vector_ind += 1

                    print("Model: {}. Score: {}".format(os.path.basename(model_dir), score))

                    if score < best_score:
                        best_score = score
                        best_match = os.path.basename(model_dir)

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("Finished fetching available models. Took {} seconds.".format(elapsed))
    
    print("Best match: {}".format(best_match))
    
    # candidate_vectors = np.array(candidate_vectors)

    # # all_vectors = candidate_vectors
    # # all_vectors.append(image_set_vector.tolist())
    # # all_vectors = np.array(all_vectors)
    # print("candidate_vectors.shape: {}".format(candidate_vectors.shape))

    # # old_settings = np.seterr(all='ignore')
    # # np.seterr(over='raise')


    # candidates_mean = np.mean(candidate_vectors, axis=0)
    # candidates_std = np.std(candidate_vectors, axis=0)

    # candidate_vectors = np.divide((candidate_vectors - candidates_mean), 
    #                                     candidates_std)

    # invalid_cols = np.logical_or(np.any(np.isinf(candidate_vectors), axis=0), np.any(np.isnan(candidate_vectors), axis=0))
    # print("invalid_cols", invalid_cols)
    # candidate_vectors = candidate_vectors[:, np.logical_not(invalid_cols)]
    # if candidate_vectors.size == 0:
    #     raise RuntimeError("Unable to auto-select model.")

    # print("candidate_vectors.shape: {}".format(candidate_vectors.shape))
    # # candidate_vectors = candidate_vectors[:all_vectors.shape[0]-1, :]
    # query_vector = np.divide((image_set_vector - candidates_mean), 
    #                                     candidates_std)
    # tree = KDTree(candidate_vectors, leaf_size=2)
    # dists, inds = tree.query(np.expand_dims(query_vector, axis=0), k=candidate_vectors.shape[0]) #3)

    # print("dists", dists)
    # print("inds", inds)

    # for ind, dist in zip(inds[0], dists[0]):
    #     print("{}: {}".format(vector_ind_to_model[ind], dist))

    # # logger.info("Chose {}".format(vector_ind_to_model[inds[0][0]]))




def test():
    logging.basicConfig(level=logging.INFO)

    item = {}
    item["username"] = "erik"
    # item["farm_name"] = "Arvalis_3"
    # item["field_name"] = "France"
    # item["mission_date"] = "2022-11-22"
    # item["object_name"] = "wheat_head"

    # item["farm_name"] = "ortho_test_02"
    # item["field_name"] = "ortho_test_02"
    # item["mission_date"] = "2022-11-05"
    # item["object_name"] = "canola_seedling"

    # item["farm_name"] = "Arvalis_11"
    # item["field_name"] = "France"
    item["farm_name"] = "ETHZ_1"
    item["field_name"] = "Switzerland"
    item["mission_date"] = "2022-11-22"
    item["object_name"] = "wheat_head"

    auto_select_model(item)


if __name__ == "__main__":
    test()