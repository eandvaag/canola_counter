
from html.entities import name2codepoint
import os
import glob
import math as m
import random
import numpy as np
import tensorflow as tf
import logging

#import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from pulp import *
import networkx as nx
from networkx.algorithms import bipartite
#import networkx as nx
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse import csr_matrix

from graph import graph_model
from io_utils import w3c_io
import extract_patches as ep
from image_set import DataSet

#just a convenience function to generate a dict of dicts
def create_wt_doubledict(from_nodes, to_nodes, wts):

    wt = {}
    for u in from_nodes:
        wt[u] = {}
        for v in to_nodes:
            wt[u][v] = 0

    for k,val in wts.items():
        u,v = k[0], k[1]
        wt[u][v] = val
    return(wt)

def solve_wbm(from_nodes, to_nodes, wt, ucap, vcap):
    ''' A wrapper function that uses pulp to formulate and solve a WBM'''

    prob = LpProblem("wbm_problem", LpMinimize) #Maximize) #inimize) # LpMaximize)

    # Create The Decision variables
    choices = LpVariable.dicts("e", (from_nodes, to_nodes), 0, 1, LpInteger)

    # Add the objective function 
    prob += lpSum([wt[u][v] * choices[u][v]
                   for u in from_nodes
                   for v in to_nodes]), "Total weights of selected edges"


    # Constraint set ensuring that the total from/to each node 
    # is less than its capacity
    for u in from_nodes:
        for v in to_nodes:
            prob += lpSum([choices[u][v] for v in to_nodes]) <= ucap[u], ""
            prob += lpSum([choices[u][v] for u in from_nodes]) <= vcap[v], ""


    # The problem data is written to an .lp file
    prob.writeLP("WBM.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    #print( "Status:", LpStatus[prob.status])
    return(prob)



def diverse_bipartite_b_match(config, desired_source_size, source_features, target_features, source_patches, target_patches):
    pass

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def bipartite_b_match(config, desired_source_size, source_features, target_features, source_patches, target_patches):

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    b = m.ceil(desired_source_size / target_features.shape[0])
    print("b", b)
    org_target_size = target_features.shape[0]
    print("target_features.shape", target_features.shape)
    target_features = np.tile(target_features, (b, 1))
    print("target_features.shape", target_features.shape)
    distances = pairwise_distances(target_features, target_features) #source_features)
    print("distances symmetric ?? {}".format(check_symmetric(distances)))
    row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(distances))
    print("distances.shape", distances.shape)
    print("row_ind", row_ind)
    print("col_ind", col_ind)
    #selected_distances = distances[np.arange(distances.shape[0]), col_ind]
    selected_distances = distances[row_ind, col_ind]
    print("selected_distances", selected_distances)
    order = selected_distances.argsort()
    print("order", order)
    sorted_col_ind = col_ind[order[::-1]]
    selected_source_patches = source_patches[sorted_col_ind]
    selected_source_patches = np.array(selected_source_patches[:desired_source_size])

    output = True
    if output:
        b_match_dir = os.path.join(config.model_dir, "b_matching")
        os.makedirs(b_match_dir)
        for i in range(target_features.shape[0]):
            target_ind = i % org_target_size
            target_dir = os.path.join(b_match_dir, str(target_ind))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                target_patch_data = {"patch": target_patches[target_ind]["patch"], 
                "patch_name": "target_" + str(target_ind) + ".png"
                }
                ep.write_patches(target_dir, [target_patch_data])

            patch_data = source_patches[col_ind[i]]

            #{"patch": source_patches[col_ind[i]],
            #              "patch_name": str(i) + ".png"}
            ep.write_patches(target_dir, [patch_data])
        exit()

    return selected_source_patches



def iterative_bipartite_match(desired_source_size, source_features, target_features, source_patches):

    num_taken = 0
    selected_source_patches = []
    while num_taken < desired_source_size:
        # TODO need to put in loop, catch exception, and add more neighbours on exception
        # found_match = False
        
        #while not found_match:
        #try:
        #print("k is", k)
        #nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        #nbrs.fit(source_features)
        #distances, indices = nbrs.kneighbors(target_features)

        distances = pairwise_distances(target_features, source_features)
        print("shape of distances", distances.shape)
        # n = 2000
        # print("taking first {} cols".format(n))
        # distances = distances[:, :n]
        # print("shape of distances", distances.shape)
        print("distances", distances)
        #print("indices", indices)
        print("distances.shape", distances.shape)
        #print("indices.shape", indices.shape)
        print("source_features.shape", source_features.shape)
        print("target_features.shape", target_features.shape)
        #mat = np.zeros(shape=(target_features.shape[0], source_features.shape[0])) #num_target_annotations, num_source_annotations))
        #np.put_along_axis(mat, indices, distances, 1)
        #print("mat.shape", mat.shape)
        #print("mat", mat)
        print("running bipartite matching")

        
        row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(distances)) #mat))
        print("row_ind", row_ind)
        print("col_ind", col_ind)
        # found_match = True
        #except Exception as e:
        #    print("No full matching exists...increasing k")
        #    raise e #k += 1

    

        selected_source_patches.extend(source_patches[col_ind])
        source_features = np.delete(source_features, col_ind, axis=0)
        source_patches = np.delete(source_patches, col_ind, axis=0)
        num_taken += col_ind.size


    selected_source_patches = np.array(selected_source_patches)
    print("now have {} patches".format(selected_source_patches.size))
    print("taking {}".format(desired_source_size))
    selected_source_patches = selected_source_patches[:desired_source_size]
    print("now have {} patches".format(selected_source_patches.size))

    return selected_source_patches



# def create_graph_subset(config): #, model_dir):
#     logger = logging.getLogger(__name__)

#     image_set_root = os.path.join("usr", "data", "image_sets")
    
#     desired_source_size = config.training["source_construction_params"]["size"]
#     target_farm_name = config.arch["target_farm_name"]
#     target_field_name = config.arch["target_field_name"]
#     target_mission_date = config.arch["target_mission_date"]

#     annotation_records, num_source_annotations = ep.get_source_annotations(target_farm_name, target_field_name, target_mission_date)

#     if num_source_annotations < desired_source_size:
#         raise RuntimeError("Insufficient number of source annotations available. Requested: {}. Found: {}.".format(
#             desired_source_size, num_source_annotations))



#     #target_annotations_path = os.path.join(image_set_root, 
#     #                                       target_farm_name, target_field_name, target_mission_date,
#     #                                       "annotations", "annotations_w3c.json")
#     #target_annotations = w3c_io.load_annotations(target_annotations_path, {"plant": 0})
#     #num_target_annotations = ep.get_num_annotations(target_annotations)

#     data = get_feature_vectors(config) #target_farm_name, target_field_name, target_mission_date, use_full_patches)
#     source_features = data["source_features"]
#     target_features = data["target_features"]
#     source_patches = data["source_patches"]
#     target_patches = data["target_patches"]

#     num_target_annotations = len(target_patches)
#     k = desired_source_size / num_target_annotations
#     if k < 1:
#         sample_inds = random.sample(np.arange(len(target_patches)).tolist(), desired_source_size)
#         target_patches = target_patches[sample_inds]
#         target_features = target_features[sample_inds]
#         k = 1
#         num_target_annotations = desired_source_size

#         # take a random subset of target images, equal in size to source_size
#         # num_target_annotations = source_size
#         # set k = 1
#     else:
#         k = m.ceil(k)
#         # use entire target_set
#         # set k = ceil(k)
#     k = 5
#     print("k is: {}".format(k))


#     # match target images with source images
#     # capacity of each target node is equal to k
#     # every (source node, target node) contains an edge 
#     # guaranteed to have exactly source_size source images
#     print("source_feature.shape", source_features.shape)
#     print("target_features.shape", target_features.shape)


#     selected_source_patches = bipartite_b_match(config, desired_source_size, source_features, target_features, source_patches, target_patches)
#     #selected_source_patches = iterative_bipartite_match(desired_source_size, source_features, target_features, source_patches)


#     usr_data_root = os.path.join("usr", "data")
#     patches_dir = os.path.join(usr_data_root, "models", config.arch["model_uuid"], "source_patches", "0")
#     training_patches_dir = os.path.join(patches_dir, "training")
#     validation_patches_dir = os.path.join(patches_dir, "validation")
#     os.makedirs(training_patches_dir)
#     os.makedirs(validation_patches_dir)

#     logger.info("Extracting source patches")
#     #training_size = round(desired_source_size * 0.8)
#     training_size = round(selected_source_patches.size * 0.8)
#     #training_subset = random.sample(np.arange(desired_source_size).tolist(), training_size)
#     training_subset = random.sample(np.arange(selected_source_patches.size).tolist(), training_size)
#     training_patches = []
#     validation_patches = []
#     #for i in range(desired_source_size):
#     for i in range(selected_source_patches.size):
#         if i in training_subset:
#             training_patches.append(selected_source_patches[i])
#         else:
#             validation_patches.append(selected_source_patches[i])

#     if use_full_patches:
#         ep.write_annotated_patch_records(training_patches, training_patches_dir)
#         ep.write_annotated_patch_records(validation_patches, validation_patches_dir)
#     else:
#         ep.extract_patches_from_gt_box_records(training_patches, training_patches_dir)
#         ep.extract_patches_from_gt_box_records(validation_patches, validation_patches_dir)
#     logger.info("Finished extracting source patches")


#     return
#     matched_source_size = np.unique(indices).size
#     #vcap = {i: 1 for i in range(matched_source_size)}

#     wts = {}
#     to_nodes = []
#     for i in range(distances.shape[0]):
#         for j in range(distances.shape[1]): #range(i, distances.shape[1]):
#             wts[(i,indices[i][j])] = distances[i][j]
#             to_nodes.append(indices[i][j])
#     print("number of weights", len(wts.keys()))
#     print("wts", wts)


#     from_nodes = np.arange(num_target_annotations)
#     #to_nodes = np.arange(num_source_annotations)
#     to_nodes = np.array(to_nodes)
#     print("from_nodes", from_nodes)
#     print("to_nodes", to_nodes)
#     ucap = {i: k for i in from_nodes}
#     vcap = {i: 1 for i in to_nodes}


#     print("from_nodes.shape", from_nodes.shape)
#     print("to_nodes.shape", to_nodes.shape)

#     # G = nx.Graph()
#     # G.add_nodes_from(from_nodes, bipartite=0)
#     # G.add_nodes_from(to_nodes, bipartite=1)

#     # for i in range(distances.shape[0]):
#     #     for j in range(distances.shape[1]): #range(i, distances.shape[1]):
#     #         G.add_edge(i, indices[i][j], weight=distances[i][j])
#     #         #        wts[(i,j)] = distances[i][j]
#     # print("running bipartite_matching")
#     # matching = nx.algorithms.bipartite.minimum_weight_full_matching(G, from_nodes, "weight")
#     # print("finished bipartite matching")


#     wt = create_wt_doubledict(from_nodes, to_nodes, wts)
#     print("running optimizer")
#     p = solve_wbm(from_nodes, to_nodes, wt, ucap, vcap)
#     print("optimization complete")

#     selected_from = [v.name.split("_")[1] for v in p.variables() if v.value() > 1e-3]
#     selected_to   = [v.name.split("_")[2] for v in p.variables() if v.value() > 1e-3]
#     print("num selected_to", len(selected_to))
#     print("num selected_from", len(selected_from))





def get_feature_vectors(config):

    usr_data_root = os.path.join("usr", "data")
    image_set_root = os.path.join(usr_data_root, "image_sets")

    
    target_farm_name = config.arch["target_farm_name"]
    target_field_name = config.arch["target_field_name"]
    target_mission_date = config.arch["target_mission_date"]

    source_construction_params = config.training["source_construction_params"]

    #test_root = "/home/eaa299/Documents/work/graph_prj/testing"
    #test_root_patches = "/home/eaa299/Documents/work/graph_prj/testing/patches"
 

    model = graph_model.get_model()

    source_patches = []
    target_patches = []

    source_features = []
    target_features = []

    source_count = 0
    source_labels = {}

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        print(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            print("  ", field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                print("     ", mission_path, end="")

                farm_name = os.path.basename(farm_path)
                field_name = os.path.basename(field_path)
                mission_date = os.path.basename(mission_path)

                dataset = DataSet({
                    "farm_name": farm_name,
                    "field_name": field_name,
                    "mission_date": mission_date
                })

                if (farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date:
                    features_lst = target_features
                    patches_lst = target_patches
                else:
                    features_lst = source_features
                    patches_lst = source_patches

                patches = ep.extract_patches_for_graph_match(dataset, use_full_patches)
                num_patches = len(patches)

                patches_lst.extend(patches)

                batch_size = 1024
                if use_full_patches:
                    input_image_shape = config.arch["input_image_shape"]
                    #raise RuntimeError("TEST ERROR") 
                else:
                    input_image_shape = np.array([150, 150, 3])

                for i in range(0, num_patches, batch_size):
                    batch_patches = []
                    for j in range(i, min(num_patches, i+batch_size)):
                        patch = tf.convert_to_tensor(patches[j]["patch"], dtype=tf.float32)
                        patch = tf.image.resize(images=patch, size=input_image_shape[:2])
                        batch_patches.append(patch)
                    batch_patches = tf.stack(values=batch_patches, axis=0)
                    
                    features = model.predict(batch_patches)
                    for f in features:
                        f = f.flatten()

                        features_lst.append(f)
                        #if not target:
                        #    source_labels[source_count] = image_set_label
                        #    source_count += 1
    data = {
        "source_patches": np.array(source_patches),
        "target_patches": np.array(target_patches),
        "source_features": np.array(source_features),
        "target_features": np.array(target_features)
    }
    return data



# def match(target_farm_name, target_field_name, target_mission_date):

#     usr_data_root = os.path.join("usr", "data")
#     image_set_root = os.path.join(usr_data_root, "image_sets")

#     test_root = "/home/eaa299/Documents/work/graph_prj/testing"
#     test_root_patches = "/home/eaa299/Documents/work/graph_prj/testing/patches"
 

#     model = graph_model.get_model()

#     source_features = []
#     target_features = []

#     source_count = 0
#     source_labels = {}

#     for farm_path in glob.glob(os.path.join(image_set_root, "*")):
#         print(farm_path)
#         for field_path in glob.glob(os.path.join(farm_path, "*")):
#             print("  ", field_path)
#             for mission_path in glob.glob(os.path.join(field_path, "*")):
#                 print("     ", mission_path, end="")
#                 annotation_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
#                 annotations = w3c_io.load_annotations(annotation_path, {"plant": 0})
#                 completed_images = []
#                 for img_name in annotations.keys():
#                     if annotations[img_name]["status"] == "completed":
#                         completed_images.append(img_name)

#                 print(" | num_completed:", len(completed_images))

#                 farm_name = os.path.basename(farm_path)
#                 field_name = os.path.basename(field_path)
#                 mission_date = os.path.basename(mission_path)

#                 dataset = DataSet({
#                     "farm_name": farm_name,
#                     "field_name": field_name,
#                     "mission_date": mission_date,
#                     "image_names": completed_images,
#                     "patch_extraction_params": None
#                 })


#                 image_set_label = farm_name + "/" + field_name + "/" + mission_date
#                 patch_dir = os.path.join(test_root_patches, image_set_label)
#                 ep.extract_patches_for_graph(dataset, patch_dir)



#                 patches_record_path = os.path.join(patch_dir, "patches-record.tfrec")

#                 input_image_shape = np.array([150, 150, 3])
#                 dl = graph_model.DataLoader([patches_record_path], input_image_shape, 1024)
#                 dataset, num_patches = dl.create_batched_dataset()
    
#                 if (farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date:
#                     features_lst = target_features
#                     target = True
#                 else:
#                     features_lst = source_features
#                     target = False
#                     #labels[image_set_label] = 
#                 print("Processing {} patches.".format(num_patches))
#                 steps = np.sum([1 for i in dataset])
#                 bar = tqdm.tqdm(dataset, total=steps)
#                 for batch_data in bar:
#                     batch_images = dl.read_batch_data(batch_data)
                    
#                     features = model.predict(batch_images)
#                     for f in features:
#                         f = f.flatten()
#                         #print("np.shape(features)", np.shape(f))

#                         features_lst.append(f)
#                         if not target:
#                             source_labels[source_count] = image_set_label
#                             source_count += 1




#     source_features = np.array(source_features)
#     target_features = np.array(target_features)
    
#     nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
#     nbrs.fit(source_features)
#     distances, indices = nbrs.kneighbors(target_features)

#     G = nx.DiGraph()

#     for i in range(len(target_features)):
#         index = len(source_features) + i
#         G.add_nodes_from([(index, {"dataset_name": target_farm_name + "/" + target_field_name + "/" + target_mission_date, "bipartite": 0})])


#     for i in range(len(source_features)):
#         G.add_nodes_from([(i, {"dataset_name": source_labels[i], "bipartite": 1})])



#     for target_index, (indices_row, distances_row) in enumerate(zip(indices, distances)):
#         for (source_index, distance) in zip(indices_row, distances_row):
#             G.add_edges_from([(len(source_features) + target_index, source_index, {"weight": distance})])


#     #my_matching = bipartite.matching.minimum_weight_full_matching(G, target_nodes, "weight")


#     nx.write_gexf(G, "/home/eaa299/Documents/work/graph_prj/testing/weighted_directional_graph.gexf")



