import tqdm

import numpy as np
import cv2
import skimage.feature as feat
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import tensorflow as tf


def calculate_glcm_features_inner(I, distances, angles, props):
    glcm_features = []

    glcms = feat.greycomatrix(I, distances, angles, symmetric=True, normed=True)
    for prop in props:
        stats = feat.greycoprops(glcms, prop)
        glcm_features.extend(stats.flatten())

    return np.array(glcm_features)


def calculate_lbp_features(I):
    lbp_uniform = feat.local_binary_pattern(I, P, R, method='uniform')
    lbp_uniform_hist, _ = np.histogram(lbp_uniform, bins=10, range=(0,9))

    lbp_var = feat.local_binary_pattern(I, P, R, method='var')
    lbp_var_hist, _ = np.histogram(lbp_var, bins=16, range=(0,7000))
    return np.concatenate((lbp_uniform_hist, lbp_var_hist))




def calculate_glcm_features(I):
    #glcm_training_features = []
    distances = [1, 8, 80]
    angles = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]
    props = ['dissimilarity']
    #glcm_training_features.append(calculate_glcm_features(I, distances, angles, props))
    return calculate_glcm_features_inner(I, distances, angles, props)
    

def calculate_HOG_features(I):
    return feat.hog(I, channel_axis=2)

    hog = cv2.HOGDescriptor()
    return hog.compute(I)

def calculate_ORB_features(I):
    orb = cv2.ORB_create()
    kp = orb.detect(I, None)
    kp, des = orb.compute(I, kp)
    return des.flatten()

    extr = feat.ORB()
    extr.detect_and_extract(I)
    return extr.descriptors.flatten()

def manual_desc(data, extraction_type, config):
    for dataset_loc in ["source", "target"]:

        patches = data[dataset_loc + "_patches"]
        features_lst = data[dataset_loc + "_features"]
        
        num_patches = patches.shape[0]

        if extraction_type == "box_patches":
            input_image_shape = np.array([150, 150, 3])
        else:
            input_image_shape = config.arch["input_image_shape"]


        for i in tqdm.trange(0, num_patches):

            patch = tf.convert_to_tensor(patches[i]["patch"], dtype=tf.float32)
            patch = tf.image.resize(images=patch, size=input_image_shape[:2]).numpy().astype(np.uint8)
            #f = calculate_glcm_features(img_as_ubyte(rgb2gray(patch)))
            #f = calculate_ORB_features(patch) #img_as_ubyte(rgb2gray(patch)))
            f = calculate_HOG_features(patch)
            #print(f.shape)
            features_lst.append(f)