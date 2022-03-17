import os
import glob

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import networkx as nx


import matplotlib.pyplot as plt


import extract_patches as ep
from image_set import DataSet


def get_model():
    weights = 'imagenet'
    model = tf.keras.applications.ResNet50(
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

from sklearn.neighbors import NearestNeighbors




def construct_graph(target_farm_name, target_field_name, target_mission_date):
    pass






def test():


    feats_1 = [[0, 1], [0, 0] , [0, 2], [0, 3]]
    feats_2 = [[0, 0], [0, 1], [0, 2], [0, 3]]

    feats_1 = np.float32(feats_1)
    feats_2 = np.float32(feats_2)


    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
    nbrs.fit(feats_1)
    distances, indices = nbrs.kneighbors(feats_2)

    print("distances", distances)
    print("indices", indices)
    exit()


    #print("feats_1.shape", feats_1.shape)
    #print("feats_2.shape", feats_2.shape)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(f_1, f_2, k=2)#new_desc, prev_desc, k=2)
    matches_2_1 = flann.knnMatch(feats_2, feats_1, k=2)
    matches_1_2 = flann.knnMatch(feats_1, feats_2, k=2)


    print("matches_2_1", matches_2_1)
    print("matches_1_2", matches_1_2)

    for m, n in matches_2_1:
        print("m, n", m.queryIdx, n.queryIdx)

    exit()
    #good_query = []
    #good_train = []
    #good = []
    #good_pt_matches_1 = []
    #good_pt_matches_2 = []
    good_match_inds_1 = []
    good_match_inds_2 = []
    #thresh = 0.25
    #ratios = []
    #distances = []

    # for m, n in matches_1_2:

    #     distances.append(m.distance)

    #     # Discard the less useful matches.
    #     if m.distance < good_match_thresh * n.distance:

    #     #if m.distance < 25:
    #         #good_query.append(m.queryIdx)
    #         #good_train.append(m.trainIdx)
    #         #good.append([m])
    #         good_match_inds_1.append(m.trainIdx)
    #         good_match_inds_2.append(m.queryIdx)
    #         #good_pt_matches_1.append(pts_1[m.trainIdx])
    #         #goot_pt_matches_2.append(pts_2[m.queryIdx])
    good_match_tups = []

    for m, n in matches_2_1:
        if m.distance < good_match_thresh * n.distance:
            good_match_tups.append([m.trainIdx, m.queryIdx])


    for m, n in matches_1_2:
        if m.distance < good_match_thresh * n.distance:
            good_match_tups.append([m.queryIdx, m.trainIdx])

    #print("good_match_tups", good_match_tups)

    u, counts = np.unique(np.array(good_match_tups), return_counts=True, axis=0)

    inds = np.where(counts == 2)


def patch_similarity():


    ds = DataSet({
                            "farm_name": "UNI",
                            "field_name": "LowN1",
                            "mission_date": "2021-06-07",
                            "image_names": ["1", "6"], # "6", "8", "16", "26", "31", "36", "40"],
                            "patch_extraction_params": {
                                "method": "tile",
                                "patch_size": 200,
                                "patch_overlap_percent": 0
                            }
                        })

    ds2 = DataSet({
                            "farm_name": "row_spacing",
                            "field_name": "nasser",
                            "mission_date": "2020-06-08",
                            "image_names": ["CF198960", "CF199260"],
                            "patch_extraction_params": {
                                "method": "tile",
                                "patch_size": 200,
                                "patch_overlap_percent": 0
                            }
                        })

    ds3 = DataSet({
                        "farm_name": "row_spacing",
                        "field_name": "brown",
                        "mission_date": "2021-06-01",
                        "image_names": ["102", "203"],
                        "patch_extraction_params": {
                            "method": "tile",
                            "patch_size": 200,
                            "patch_overlap_percent": 0
                        }
                    })

    ds4 = DataSet({
                        "farm_name": "BlaineLake",
                        "field_name": "River",
                        "mission_date": "2021-06-09",
                        "image_names": ["3", "28"],
                        "patch_extraction_params": {
                            "method": "tile",
                            "patch_size": 200,
                            "patch_overlap_percent": 0
                        }
                    })

    patch_dir_1 = "/home/eaa299/Documents/work/graph_prj/g_1"
    patch_dir_2 = "/home/eaa299/Documents/work/graph_prj/g_2"
    patch_dir_3 = "/home/eaa299/Documents/work/graph_prj/g_3"
    patch_dir_4 = "/home/eaa299/Documents/work/graph_prj/g_4"
    ep.extract_patches_for_graph(ds, patch_dir_1)
    ep.extract_patches_for_graph(ds2, patch_dir_2)
    ep.extract_patches_for_graph(ds3, patch_dir_3)
    ep.extract_patches_for_graph(ds4, patch_dir_4)
    #training_tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

    patch_dirs = [patch_dir_1, patch_dir_2, patch_dir_3, patch_dir_4]
    #yolov4 = YOLOv4Tiny(config)
    model = get_model()

    patches_per_dataset = 50
    X = []
    dataset_features = {"d" + str(i) : [] for i in range(len(patch_dirs))}
    for i, patch_dir in enumerate(patch_dirs):
        patch_paths = glob.glob(os.path.join(patch_dir, "*.png"))

        #print("---")
        for patch_path in patch_paths[:patches_per_dataset]:
            #print("patch_name", os.path.basename(patch_path))
            #print("----")
            patch = cv2.cvtColor(cv2.imread(patch_path), cv2.COLOR_BGR2RGB)
            #print("patch.shape", patch.shape)
            patch = tf.convert_to_tensor(patch, dtype=tf.float32)
            patch = tf.expand_dims(patch, axis=0)
            #print("patch.shape", tf.shape(patch))
            features = model.predict(patch)
            #print(features.shape)
            features = features.flatten()

            print(features.shape)
            exit()
            #print(features.shape)
            X.append(features)

            dataset_features["d" + str(i)].append(features)


    


    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    source_datasets = []
    source_datasets.extend(dataset_features["d0"])
    source_datasets.extend(dataset_features["d1"])
    source_datasets.extend(dataset_features["d2"])
    target_dataset = dataset_features["d3"]
    nbrs.fit(source_datasets)
    distances, indices = nbrs.kneighbors(target_dataset)

    
    print("distances", distances)
    print("indices", indices)
    
    G = nx.DiGraph()
    colors = ["UNI/LowN1/2021-06-07", "row_spacing/nasser/2020-06-08", "row_spacing/brown/2021-06-01", "BlaineLake/River/2021-06-09"]
    i = 0
    for dataset in ["d0", "d1", "d2", "d3"]:
        for img_feat in dataset_features[dataset]:
            dataset_name = colors[i // patches_per_dataset]
            G.add_nodes_from([(i, {"dataset_name": dataset_name})])
            i += 1 

    for j, (ind_row, distances_row) in enumerate(zip(indices, distances)):
        for index, distance in zip(ind_row, distances_row):
            G.add_edges_from([(index, j + patches_per_dataset * 3, {"weight": distance})])


    nx.write_gexf(G, "/home/eaa299/Documents/work/graph_prj/weighted_directional_graph.gexf")

    exit()

    similarity_matrix = cosine_similarity(X)
    #similarity_matrix = euclidean_distances(X)
    #print(similarity_matrix)
    # colors = ["UNI/LowN1/2021-06-07", "row_spacing/nasser/2020-06-08", "row_spacing/brown/2021-06-01"]
    # G = nx.Graph()
    # for i in range(len(similarity_matrix[:,0])):
    #     dataset_name = colors[i // patches_per_dataset]
    #     G.add_nodes_from([(i, {"dataset_name": dataset_name})])
    #     for j in range(len(similarity_matrix[0,:])):
    #         G.add_edges_from([(i, j, {"weight": similarity_matrix[i][j]})])


    # nx.write_gexf(G, "/home/eaa299/Documents/work/graph_prj/similarity_graph.gexf")


    # for i in range(len(similarity_matrix[:,0])):
    #     for j in range(len(similarity_matrix[0,:])):

    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.savefig("/home/eaa299/Documents/work/graph_prj/heatmap_similarity_bboxes.png")

if __name__ == "__main__":
    patch_similarity()