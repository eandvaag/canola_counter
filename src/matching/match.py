import os
import glob
import logging

import math as m
import numpy as np
import cv2
import tqdm
import uuid
import random
import time

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances.euclidean import EuclideanDistance
from sklearn.neighbors import KDTree, BallTree #LSHForest
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse import csr_matrix


from image_set import Image
from io_utils import w3c_io, json_io
import extract_patches as ep
import extract_features as ef



SAMPLE_SIZE = 200
SOURCE_POOL_SIZE = 25000
TARGET_MAX_SIZE = 100 #25000




def probability_skew(config):
    
    logger = logging.getLogger(__name__)

    logger.info("Calculating shares...")
    image_set_root = os.path.join("usr", "data", "image_sets")

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    target_features = ef.load_features(target_farm_name, target_field_name, target_mission_date, include_coords=False)
    distances = np.zeros(shape=(target_features.shape[0], 0))

    epsilon = 1e-10
    source_image_sets = []
    shares = {}
    allocated_v = []
    available_v = []
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if not ((farm_name == target_farm_name and \
                        field_name == target_field_name) and \
                        mission_date == target_mission_date):
                        
                    if farm_name not in shares:
                        shares[farm_name] = {}
                    if field_name not in shares[farm_name]:
                        shares[farm_name][field_name] = {}


                    shares[farm_name][field_name][mission_date] = {
                        "allocated": 0,
                        "available": 0
                    }
                    try:
                        source_features = ef.load_features(farm_name, field_name, mission_date, include_coords=False)

                        logger.info("Now processing {} {} {}. Loaded {} features.".format(
                            farm_name, field_name, mission_date, source_features.shape[0]
                        ))
                        if source_features.shape[0] > 0:

                            num_available = source_features.shape[0]

                            inds = np.random.choice(source_features.shape[0], SAMPLE_SIZE, replace=False)
                            source_features = source_features[inds]
                            
                            cur_distances = pairwise_distances(target_features, source_features, metric="euclidean") + epsilon #target_features) + epsilon #source_features)
                            
                            source_image_sets.append((farm_name, field_name, mission_date))

                            distances = np.concatenate([distances, cur_distances], axis=1)


                            # if farm_name not in shares:
                            #     shares[farm_name] = {}
                            # if field_name not in shares[farm_name]:
                            #     shares[farm_name][field_name] = {}
                            
                            # shares[farm_name][field_name][mission_date] = {
                            #     "allocated": 0,
                            #     "available": num_available
                            # }

                            allocated_v.append(0)
                            available_v.append(num_available)


                    except RuntimeError:
                        pass


    #for i, source_image_set in enumerate(source_image_sets):
    

    k = SAMPLE_SIZE
    smallest_dist_ind = np.argsort(distances)[:, :k]
    smallest_dist = np.take_along_axis(distances, smallest_dist_ind, axis=1)
    similarities = np.zeros(shape=np.shape(distances))
    
    largest_sim = (1 / smallest_dist)
    np.put_along_axis(similarities, smallest_dist_ind, largest_sim, axis=1)

    
    total_sim_sum = np.sum(similarities)



    allocated_v = np.array(allocated_v)
    available_v = np.array(available_v)

    for i, source_image_set in enumerate(source_image_sets):
        subset_sim = similarities[:, SAMPLE_SIZE*i:SAMPLE_SIZE*(i+1)]
        image_set_prob = np.sum(subset_sim) / total_sim_sum
        image_set_total = m.ceil(SOURCE_POOL_SIZE * image_set_prob)

        available = available_v[i]
        assigned = min(available, image_set_total)
        available_v[i] -= assigned
        allocated_v[i] = assigned
        rem = image_set_total - available
        while rem > 0:
            mask = available_v > 0
    
            if not mask.any():
                raise RuntimeError("Nowhere to allocate")

            per_image_set_needed = np.array([rem // available_v[mask].size] * available_v[mask].size)
            per_image_set_needed[:rem % available_v[mask].size] += 1
            per_image_set_needed_full = np.zeros(available_v.size, dtype=np.int64)
            per_image_set_needed_inds = np.where(available_v > 0)[0]
            per_image_set_needed_full[per_image_set_needed_inds] = per_image_set_needed

            reduction = available_v - per_image_set_needed_full
            taken = np.clip(per_image_set_needed_full, None, available_v)
            allocated_v += taken
            redistribute = (-1) * np.sum(reduction[reduction < 0])
            rem = redistribute
            available_v -= taken


    # for i, source_image_set in enumerate(source_image_sets):
        
    #     farm_name = source_image_set[0]
    #     field_name = source_image_set[1]
    #     mission_date = source_image_set[2]


    #     subset_sim = similarities[:, SAMPLE_SIZE*i:SAMPLE_SIZE*(i+1)]
    #     image_set_prob = np.sum(subset_sim) / total_sim_sum
    #     image_set_total = m.ceil(SOURCE_POOL_SIZE * image_set_prob)

    #     shares[farm_name][field_name][mission_date]["allocated"] = image_set_total

    # for i, source_image_set in enumerate(source_image_sets):
    #     farm_name = source_image_set[0]
    #     field_name = source_image_set[1]
    #     mission_date = source_image_set[2]

    #     allocated = shares[farm_name][field_name][mission_date]["allocated"]
    #     available = shares[farm_name][field_name][mission_date]["available"]

    #     if available < allocated:
    #         print("{} {} {} | available: {}. allocated: {}".format(
    #             farm_name, field_name, mission_date, available, allocated
    #         ))
    #         raise RuntimeError("available < allocated. Need to fix this.")

    #         to_distribute = allocated - available

    #         while to_distribute > 0:


            # for j, source_image_set_2 in enumerate(source_image_sets):
            #     if i != j:
            #         farm_name_2 = source_image_set[0]
            #         field_name_2 = source_image_set[1]
            #         mission_date_2 = source_image_set[2]

            #         allocated = shares[farm_name_2][field_name_2][mission_date_2]["allocated"]
            #         available = shares[farm_name_2][field_name_2][mission_date_2]["available"]    

            #         if available > allocated:            

    #shares = {}
    for i, source_image_set in enumerate(source_image_sets):

        farm_name = source_image_set[0]
        field_name = source_image_set[1]
        mission_date = source_image_set[2]

            
        shares[farm_name][field_name][mission_date] = {
            "allocated": allocated_v[i],
            "available": available_v[i]
        }

    logger.info("Finished calculating shares.")

    return shares



def output_debug_matches(config, distances, intervals, source_image_sets, target_coords_rec, source_coords_rec):
    
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    extraction_rec = {}
    debug_out_dir = os.path.join(config.model_dir, "matches")
    os.makedirs(debug_out_dir)

    k = 10
    smallest_dist_ind = np.argsort(distances)[:, :k]
    #smallest_dist = np.take_along_axis(distances, smallest_dist_ind, axis=1)
    #distances = np.zeros(shape=np.shape(distances))

    #np.put_along_axis(distances, smallest_dist_ind, smallest_dist, axis=1)



    target_extraction_rec = {}
    patch_coords_recs = {}
    for i in tqdm.trange(smallest_dist_ind.shape[0], desc="Creating extraction records"):
        target_image_name = target_coords_rec["image_names"][i]
        target_patch_coords = target_coords_rec["patch_coords"][i]

        if target_image_name not in target_extraction_rec:
            target_extraction_rec[target_image_name] =  {
                    "target_patch_id": [],
                    "patch_coords": [] 
                }
        target_extraction_rec[target_image_name]["target_patch_id"].append(i)
        target_extraction_rec[target_image_name]["patch_coords"].append(target_patch_coords)

        for col_ind in smallest_dist_ind[i]:
            
            source_image_set_ind = np.searchsorted(intervals, col_ind, side="right") - 1
            source_image_set = source_image_sets[source_image_set_ind]
            starting_ind = intervals[source_image_set_ind]
            patch_ind = col_ind - starting_ind
            #print("source_image_set_ind: {}, col_ind: {}, patch_ind: {}".format(source_image_set_ind, col_ind, patch_ind))

            if source_image_set not in patch_coords_recs:
                patch_coords_recs[source_image_set] = json_io.load_json(os.path.join("usr", "data", "image_sets",
                                        source_image_set[0], source_image_set[1], source_image_set[2],
                                        "features", "patch_coords.json"))
            coords_rec = patch_coords_recs[source_image_set]
            source_image_name = coords_rec["image_names"][patch_ind]
            patch_coords = coords_rec["patch_coords"][patch_ind]


            if source_image_set not in extraction_rec:
                extraction_rec[source_image_set] = {}

            if source_image_name not in extraction_rec[source_image_set]:
                extraction_rec[source_image_set][source_image_name] = {
                    "target_patch_id": [],
                    "patch_coords": [] 
                }
            extraction_rec[source_image_set][source_image_name]["target_patch_id"].append(i)
            extraction_rec[source_image_set][source_image_name]["patch_coords"].append(patch_coords)



    #logger.info("Started extraction...")
    target_images_dir = os.path.join("usr", "data", "image_sets", target_farm_name, target_field_name, target_mission_date, "images")
    for target_image_name in target_extraction_rec.keys():
        print(target_image_name)
        patch_coords_lst = target_extraction_rec[target_image_name]["patch_coords"]
        patch_ids = target_extraction_rec[target_image_name]["target_patch_id"]
        image_path = glob.glob(os.path.join(target_images_dir, target_image_name + ".*"))[0]
        image_array = Image(image_path).load_image_array()
        patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
        
        for i, patch in enumerate(patches):
            out_dir = os.path.join(debug_out_dir, str(patch_ids[i]))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, str(patch_ids[i]) + ".png")
            cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))


    for source_image_set in extraction_rec.keys():
        source_images_dir = os.path.join("usr", "data", "image_sets",
                                     source_image_set[0], source_image_set[1], source_image_set[2], "images")
        for source_image_name in extraction_rec[source_image_set].keys():
            patch_coords_lst = extraction_rec[source_image_set][source_image_name]["patch_coords"]
            patch_ids = extraction_rec[source_image_set][source_image_name]["target_patch_id"]
            image_path = glob.glob(os.path.join(source_images_dir, source_image_name + ".*"))[0]
            image_array = Image(image_path).load_image_array()
            patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
            
            for i, patch in enumerate(patches):
                out_dir = os.path.join(debug_out_dir, str(patch_ids[i]))
                #if not os.path.exists(out_dir):
                #    os.makedirs(out_dir)
                out_path = os.path.join(out_dir, "source_" + str(uuid.uuid4()) + ".png")
                cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

    # patch_records = []        


    #     annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    #     annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    #     for source_image_name in extraction_rec[source_image_set].keys():
    #         image_path = glob.glob(os.path.join(image_set_dir, "images", source_image_name + ".*"))[0]
    #         image = Image(image_path)
    #         #image_array = image.load_image_array()
    #         patch_coords_lst = extraction_rec[source_image_set][source_image_name]



    #         patch_records.extend(ep.extract_patch_records_from_image(image, 
    #                                 patch_coords_lst, 
    #                                 annotations[source_image_name], 
    #                                 starting_patch_num=0))



def target_retrieval(config):

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]


    target_features, target_coords_rec = ef.load_features(
        target_farm_name, target_field_name, target_mission_date, include_coords=True)

    target_images_dir = os.path.join("usr", "data", "image_sets",
    target_farm_name, target_field_name, target_mission_date, "images")

    print("building tree")
    start_time = time.time()
    #tree = KDTree(target_features)
    tree = BallTree(target_features)
    end_time = time.time()
    print("tree built in {} seconds".format(end_time-start_time))

    match_out = os.path.join(config.model_dir, "match")


    inds = np.random.choice(target_features.shape[0], 20, replace=False)
    for ind in inds:
        #m_distances, m_indices = lshf.kneighbors(target_features[ind], n_neighbors=3)
        m_distances, m_indices = tree.query((target_features[ind]).reshape(1, -1), k=10)
        print("m_indices", m_indices)

        out_dir = os.path.join(match_out, str(ind))
        os.makedirs(out_dir)

        target_image_name = target_coords_rec["image_names"][ind]
        patch_coords_lst = [target_coords_rec["patch_coords"][ind]]
        image_path = glob.glob(os.path.join(target_images_dir, target_image_name + ".*"))[0]
        image_array = Image(image_path).load_image_array()
        patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
        out_path = os.path.join(out_dir, "query_" + str(ind) + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(patches[0], cv2.COLOR_RGB2BGR))

        for i, m_ind in enumerate(m_indices[0]):

            # source_image_set_ind = np.searchsorted(intervals, m_ind, side="right") - 1
            # source_image_set = source_image_sets[source_image_set_ind]
            # starting_ind = intervals[source_image_set_ind]
            # patch_ind = m_ind - starting_ind
            # source_images_dir = os.path.join("usr", "data", "image_sets",
            # source_image_set[0], source_image_set[1], source_image_set[2], "images")

            target_image_name = target_coords_rec["image_names"][m_ind]
            patch_coords_lst = [target_coords_rec["patch_coords"][m_ind]]
            image_path = glob.glob(os.path.join(target_images_dir, target_image_name + ".*"))[0]
            image_array = Image(image_path).load_image_array()
            patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
            out_path = os.path.join(out_dir, "match_" + str(i) + ".png")
            cv2.imwrite(out_path, cv2.cvtColor(patches[0], cv2.COLOR_RGB2BGR))

def get_nn(config):
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    target_tup = (target_farm_name, target_field_name, target_mission_date)
    print("loading all features")
    source_features, source_patch_coords = ef.load_all_features(omit=[target_tup])
    #exit()
    

    c_source_features = np.empty((0,source_features[list(source_features.keys())[0]].shape[1])) #None
    source_image_sets = []
    intervals = []
    for image_set in source_features.keys():
        source_image_sets.append(image_set)
        intervals.append(c_source_features.shape[0])
        c_source_features = np.concatenate([c_source_features, source_features[image_set]])
            
    del source_features
    print("c_source_features.shape", c_source_features.shape)

    target_features, target_coords_rec = ef.load_features(
        target_farm_name, target_field_name, target_mission_date, include_coords=True)
    print("target_features.shape", target_features.shape)

    target_images_dir = os.path.join("usr", "data", "image_sets",
    target_farm_name, target_field_name, target_mission_date, "images")


    # rbp = RandomBinaryProjections('rbp', 10)
    # engine = Engine(c_source_features.shape[1], distance=EuclideanDistance(), lshashes=[rbp])

    # print("pairwise_distances")
    # start_time = time.time()
    # for i in tqdm.trange(target_features.shape[0]):
    #     pairwise_distances(target_features[i].reshape(1, -1), c_source_features, metric="euclidean")
    # end_time = time.time()
    # print("Pairwise distances calculated in {} seconds".format(end_time-start_time))

    print("Building tree")
    start_time = time.time()
    tree = KDTree(c_source_features) #, leaf_size=2)
    #tree = BallTree(c_source_features, leaf_size=2)
    # for i in tqdm.trange(c_source_features.shape[0]):
    #     engine.store_vector(c_source_features[i], data=i)
        
    end_time = time.time()
    print("Tree built in {} seconds".format(end_time-start_time))

    extraction_rec = {}
    patch_records = []
    # print("Querying tree")
    # start_time = time.time()
    #m_distances, m_indices = tree.query(target_features, k=1)
    # for i in tqdm.trange(target_features.shape[0]):
    #     q = engine.neighbours(target_features[i])
    #     if i == 0:
    #         print(q[0][1])
    #         #print(q)
    # end_time = time.time()
    # print("Finished querying in {} seconds".format(end_time-start_time))
    # exit()
    #prev_count = 0
    for i in tqdm.trange(target_features.shape[0], desc="Querying for nearest neighbours"):
    #for i in tqdm.trange(m_indices.shape[0], desc="Building extraction record"):
        #m_indices[0][0]
        m_distances, m_indices = tree.query((target_features[i]).reshape(1, -1), k=1)
        #print("m_indices", m_indices)

        source_image_set_ind = np.searchsorted(intervals, m_indices[0][0], side="right") - 1
        source_image_set = source_image_sets[source_image_set_ind]
        starting_ind = intervals[source_image_set_ind]
        patch_ind = m_indices[0][0] - starting_ind

        source_image_name = source_patch_coords[source_image_set]["image_names"][patch_ind]
        patch_coords = source_patch_coords[source_image_set]["patch_coords"][patch_ind]

        # print("source_image_set", source_image_set)
        # print("starting_ind", starting_ind)
        # print("patch_ind", patch_ind)
        # print("source_image_name", source_image_name)
        # print("patch_coords", patch_coords)
        #exit()
        
        if source_image_set not in extraction_rec:
            extraction_rec[source_image_set] = {}
        if source_image_name not in extraction_rec[source_image_set]:
            extraction_rec[source_image_set][source_image_name] = []
        extraction_rec[source_image_set][source_image_name].append(patch_coords)

        # count = 0
        # for i_image_set in extraction_rec.keys():
        #     for i_image_name in extraction_rec[i_image_set].keys():
        #         patch_coords_lst = extraction_rec[i_image_set][i_image_name]
        #         count += len(patch_coords_lst)

        # if count != prev_count + 1:
        #     print("Count != prev count || count: {}, prev_count: {}".format(count, prev_count))
        #     print(extraction_rec)
        #     exit()
        # prev_count = count
        # print("Found {} coord lists in extraction rec".format(count))

    # count = 0
    # for i_image_set in extraction_rec.keys():
    #     for i_image_name in extraction_rec[i_image_set].keys():
    #         patch_coords_lst = extraction_rec[i_image_set][i_image_name]
    #         count += len(patch_coords_lst)
    # print("Found {} coord lists in extraction rec".format(count))


    image_num = 0
    num_unique = 0
    for source_image_set in tqdm.tqdm(extraction_rec.keys(), desc="Extracting patches"):

        image_set_dir = os.path.join("usr", "data", "image_sets",
                                     source_image_set[0], source_image_set[1], source_image_set[2])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

        for source_image_name in extraction_rec[source_image_set].keys():
            image_path = glob.glob(os.path.join(image_set_dir, "images", source_image_name + ".*"))[0]
            image = Image(image_path)
            
            patch_coords_lst = extraction_rec[source_image_set][source_image_name]
            #patch_coords_lst = #(np.unique(np.array(extraction_rec[source_image_set][source_image_name]), axis=0)).tolist()
            num_unique += len((np.unique(np.array(extraction_rec[source_image_set][source_image_name]), axis=0)).tolist())

            patch_records.extend(ep.extract_patch_records_from_image(image, 
                                    patch_coords_lst, 
                                    annotations[source_image_name], 
                                    starting_patch_num=image_num))

            image_num += len(patch_coords_lst)

    patch_records = np.array(patch_records)
    print("Generated {} patch records.".format(patch_records.size))
    print("Number of unique patches: {}".format(num_unique))
    return patch_records
            


def retrieval(config):

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    target_tup = (target_farm_name, target_field_name, target_mission_date)
    print("loading all features")
    source_features, source_patch_coords = ef.load_all_features(omit=[target_tup])
    #exit()
    

    c_source_features = np.empty((0,source_features[list(source_features.keys())[0]].shape[1])) #None
    source_image_sets = []
    intervals = []
    for image_set in source_features.keys():
        source_image_sets.append(image_set)
        intervals.append(c_source_features.shape[0])
        #if c_source_features is None:
        #    c_source_features = source_features[image_set]
        #else:
        c_source_features = np.concatenate([c_source_features, source_features[image_set]])
            

    del source_features
    print("c_source_features.shape", c_source_features.shape)

    target_features, target_coords_rec = ef.load_features(
        target_farm_name, target_field_name, target_mission_date, include_coords=True)

    target_images_dir = os.path.join("usr", "data", "image_sets",
    target_farm_name, target_field_name, target_mission_date, "images")

    # print("calculating pairwise distances")
    # pairwise_distances(target_features, target_features, metric="euclidean")
    # print("done")
    # exit()
    #lshf = LSHForest()
    #lshf.fit(target_features)
    print("building tree")
    start_time = time.time()
    #tree = KDTree(c_source_features)
    tree = BallTree(c_source_features)
    end_time = time.time()
    print("tree built in {} seconds".format(end_time-start_time))

    match_out = os.path.join(config.model_dir, "match")


    inds = np.random.choice(target_features.shape[0], 20, replace=False)
    for ind in inds:
        #m_distances, m_indices = lshf.kneighbors(target_features[ind], n_neighbors=3)
        m_distances, m_indices = tree.query((target_features[ind]).reshape(1, -1), k=10)
        print("m_indices", m_indices)

        out_dir = os.path.join(match_out, str(ind))
        os.makedirs(out_dir)

        target_image_name = target_coords_rec["image_names"][ind]
        patch_coords_lst = [target_coords_rec["patch_coords"][ind]]
        image_path = glob.glob(os.path.join(target_images_dir, target_image_name + ".*"))[0]
        image_array = Image(image_path).load_image_array()
        patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
        out_path = os.path.join(out_dir, "query_" + str(ind) + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(patches[0], cv2.COLOR_RGB2BGR))

        for i, m_ind in enumerate(m_indices[0]):

            source_image_set_ind = np.searchsorted(intervals, m_ind, side="right") - 1
            source_image_set = source_image_sets[source_image_set_ind]
            starting_ind = intervals[source_image_set_ind]
            patch_ind = m_ind - starting_ind
            source_images_dir = os.path.join("usr", "data", "image_sets",
            source_image_set[0], source_image_set[1], source_image_set[2], "images")

            source_image_name = source_patch_coords[source_image_set]["image_names"][patch_ind]
            patch_coords_lst = [source_patch_coords[source_image_set]["patch_coords"][patch_ind]]
            image_path = glob.glob(os.path.join(source_images_dir, source_image_name + ".*"))[0]
            image_array = Image(image_path).load_image_array()
            patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
            out_path = os.path.join(out_dir, "match_" + str(i) + ".png")
            cv2.imwrite(out_path, cv2.cvtColor(patches[0], cv2.COLOR_RGB2BGR))




    # for target_image_name in target_extraction_rec.keys():
    #     print(target_image_name)
    #     patch_coords_lst = target_extraction_rec[target_image_name]["patch_coords"]
    #     patch_ids = target_extraction_rec[target_image_name]["target_patch_id"]
    #     image_path = glob.glob(os.path.join(target_images_dir, target_image_name + ".*"))[0]
    #     image_array = Image(image_path).load_image_array()
    #     patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
        
    #     for i, patch in enumerate(patches):
    #         out_dir = os.path.join(debug_out_dir, str(patch_ids[i]))
    #         if not os.path.exists(out_dir):
    #             os.makedirs(out_dir)
    #         out_path = os.path.join(out_dir, str(patch_ids[i]) + ".png")
    #         cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))


def match(config):

    logger = logging.getLogger(__name__)

    shares = probability_skew(config)
    print("got shares", shares)

    image_set_root = os.path.join("usr", "data", "image_sets")

    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    #num_source_features = {}
    intervals = []
    source_image_sets = []

    target_features, target_coords_rec = ef.load_features(target_farm_name, target_field_name, target_mission_date, include_coords=True)

    source_coords_rec = {}

    # source_construction_params = config.training["source_construction_params"]
    # desired_training_set_size = source_construction_params["size"]
    if target_features.shape[0] > TARGET_MAX_SIZE:
        inds = np.random.choice(target_features.shape[0], TARGET_MAX_SIZE, replace=False)
        print("inds", inds)
        target_features = target_features[inds]
        target_coords_rec["image_names"] = (np.array(target_coords_rec["image_names"])[inds]).tolist()
        target_coords_rec["patch_coords"] = (np.array(target_coords_rec["patch_coords"])[inds]).tolist()


    # if desired_training_set_size != "max" and desired_training_set_size < target_features.shape[0]:
    #     inds = np.random.choice(target_features.shape[0], desired_training_set_size, replace=False)
    #     target_features = target_features[inds]

    distances = np.zeros(shape=(target_features.shape[0], 0))
    print("distances.shape", distances.shape)

    epsilon = 1e-10
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                print("distances.shape", distances.shape)

                if not ((farm_name == target_farm_name and \
                        field_name == target_field_name) and \
                        mission_date == target_mission_date):
                    try:
                        source_features = ef.load_features(farm_name, field_name, mission_date, include_coords=False)

                        logger.info("Now processing {} {} {}. Loaded {} features.".format(
                            farm_name, field_name, mission_date, source_features.shape[0]
                        ))
                        num_allocated = shares[farm_name][field_name][mission_date]["allocated"]

                        if source_features.shape[0] > 0 and num_allocated > 0:

                            inds = np.random.choice(source_features.shape[0], num_allocated, replace=False)
                            source_features = source_features[inds]
                            #source_coords_rec[(farm_name, field_name, mission_date)] = {}
                            #source_coords_rec[(farm_name, field_name, mission_date)]["image_names"] = (np.array(source_coords_rec["image_names"])[inds]).tolist()
                            #source_coords_rec[(farm_name, field_name, mission_date)]["patch_coords"] = (np.array(source_coords_rec["patch_coords"])[inds]).tolist()

                            intervals.append(distances.shape[1])
                            cur_distances = pairwise_distances(target_features, source_features, metric="euclidean") + epsilon #target_features) + epsilon #source_features)
                            
                            source_image_sets.append((farm_name, field_name, mission_date))

                            distances = np.concatenate([distances, cur_distances], axis=1)



                            #del source_features
                            #del cur_distances

                    except RuntimeError:
                        pass

    logger.info("All distances calculated. Distance matrix shape is {}".format(distances.shape))
    
    intervals = np.array(intervals)
    print("intervals", intervals)

    output_debug_matches(config, distances, intervals, source_image_sets, target_coords_rec, source_coords_rec)

    exit()




    logger.info("Running bipartite matching...")
    row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(distances))
    logger.info("Bipartite matching complete.")
    
    selected_distances = distances[row_ind, col_ind]
    order = selected_distances.argsort()
    sorted_col_ind = col_ind[order[::-1]]
    extraction_rec = {}
    for col_ind in sorted_col_ind:
        
        source_image_set_ind = np.searchsorted(intervals, col_ind, side="right") - 1
        source_image_set = source_image_sets[source_image_set_ind]
        starting_ind = intervals[source_image_set_ind]
        patch_ind = col_ind - starting_ind
        #print("source_image_set_ind: {}, col_ind: {}, patch_ind: {}".format(source_image_set_ind, col_ind, patch_ind))
        coords_rec = json_io.load_json(os.path.join("usr", "data", "image_sets",
                                       source_image_set[0], source_image_set[1], source_image_set[2],
                                       "features", "patch_coords.json"))
        source_image_name = coords_rec["image_names"][patch_ind]
        patch_coords = coords_rec["patch_coords"][patch_ind]


        if source_image_set not in extraction_rec:
            extraction_rec[source_image_set] = {}

        if source_image_name not in extraction_rec[source_image_set]:
            extraction_rec[source_image_set][source_image_name] = []

        extraction_rec[source_image_set][source_image_name].append(patch_coords)

    logger.info("Started extraction...")

    patch_records = []        
    for source_image_set in extraction_rec.keys():
        image_set_dir = os.path.join("usr", "data", "image_sets",
                                     source_image_set[0], source_image_set[1], source_image_set[2])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

        for source_image_name in extraction_rec[source_image_set].keys():
            image_path = glob.glob(os.path.join(image_set_dir, "images", source_image_name + ".*"))[0]
            image = Image(image_path)
            #image_array = image.load_image_array()
            patch_coords_lst = extraction_rec[source_image_set][source_image_name]



            patch_records.extend(ep.extract_patch_records_from_image(image, 
                                    patch_coords_lst, 
                                    annotations[source_image_name], 
                                    starting_patch_num=0))
            


            #exg_patches = ep.extract_patches_from_image_array(image_array, patch_coords_lst)
    logger.info("Finished extraction.")


    return np.array(patch_records)

                
                
                


    # selected_source_patches = source_patches[sorted_col_ind]
    # selected_source_patches = np.array(selected_source_patches[:desired_source_size])

