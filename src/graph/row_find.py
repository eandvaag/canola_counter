import os

import numpy as np
import scipy
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

from io_utils import w3c_io
from models.common import box_utils

import matplotlib.pyplot as plt

def degree_matrix(graph):
    return np.diag(np.sum(graph, axis=1))

def laplacian_eigenvectors(graph):
    degree = degree_matrix(graph)
    laplacian = degree - graph
    _, eigenvectors = scipy.linalg.eigh(laplacian, subset_by_index=[0, 1])
    return eigenvectors


def find_clusters(graph, points, n_clusters):
    eigs = laplacian_eigenvectors(graph)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(eigs)
    return kmeans.labels_
    #for i in range(len(kmeans.labels_)):
    #    if kmeans.labels_[i] == 0:
    #        points[i].color = "green"
    #    else:
    #        points[i].color = "red"


def find_row():
    w3c_path = os.path.join("usr", "data", "image_sets", "UNI", "LowN1", "2021-06-07", "annotations", "annotations_w3c.json")

    boxes, classes = w3c_io.load_boxes_and_classes(w3c_path, {"plant": 0})

    #points = [[point.x, point.y] for point in dataset]

    chosen_k = '40' #boxes.keys()[0]

    chosen_boxes = box_utils.swap_xy_np(boxes[chosen_k])
    
    # x, y centres
    centres = (chosen_boxes[..., :2] + chosen_boxes[..., 2:]) / 2.0

    #centres = np.array([[2, 0], [4, 0], [6, 0], [0, 2], [0, 4], [0, 6], [0, 8], [2, 10], [4, 10], [6, 10], [4, 5], [6, 5], [8, 5], [10, 5]])

    graph = kneighbors_graph(centres, n_neighbors=50, mode="distance").toarray()

    labels = find_clusters(graph, centres, 7)


    data = {}
    data_colors = ["green", "blue", "red", "brown", "yellow", "purple", "pink"]
    color_index = 0
    for i in range(len(labels)):
        if labels[i] not in data:
            data[labels[i]] = {"xs": [], "ys": [], "color": data_colors[color_index]}
            color_index += 1
        data[labels[i]]["xs"].append(centres[i][0])
        data[labels[i]]["ys"].append(centres[i][1])


    for k in data.keys():
        plt.scatter(data[k]["xs"], data[k]["ys"], color=data[k]["color"])


    plt.savefig("/home/eaa299/Documents/graph_test.png")










