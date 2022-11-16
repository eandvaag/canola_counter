import argparse
import os
import glob
import numpy as np
import cv2
from osgeo import gdal

from joblib import Parallel, delayed

# from skimage.filters import threshold_otsu

from image_set import Image
import image_utils
from io_utils import json_io
import lock
from models.common import box_utils

# def range_map(old_val, old_min, old_max, new_min, new_max):
#     new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
#     return new_val


# # def set_luminance_lab(self, image_array, luminance_adjust):
# #     """Sets the luminance (L channel in LAB color space) value for an image to a set fixed value"""
# #     im_Lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
# #     im_L = im_Lab[:, :, 0]
# #     im_L[:, :] = luminance_adjust
# #     im_Lab = np.dstack((im_L, im_Lab[:, :, 1], im_Lab[:, :, 2]))
# #     img_modified = cv2.cvtColor(im_Lab, cv2.COLOR_LAB2RGB)
# #     return img_modified




# def largest_indices(arr, n):
#     """
#     Returns the indices of the n largest elements.
#     :param arr: A numpy array
#     :param n: An positive integer
#     :return: A numpy array of indices and shape
#     """
#     # does nothing on a 1-d array
#     flattened_array = arr.flatten()
#     # Partition the array into the indices of largest values
#     indices = np.argpartition(flattened_array, -n)[-n:]
#     # Sorts in ascending order
#     indices = indices[np.argsort(-flattened_array[indices])]
#     return np.unravel_index(indices, arr.shape)


# def get_gradient_magnitude(image_array):
#     """"Compute the gradient magnitude for the passed in image
#     :return image (numpy array)
#     """
#     d_depth = cv2.CV_8U #32F
#     d_x = cv2.Sobel(image_array, d_depth, 1, 0)
#     d_y = cv2.Sobel(image_array, d_depth, 0, 1)
#     return np.sqrt(d_x ** 2 + d_y ** 2)



# def create_edge_mask(exg_array, percent_strongest_edge=1):
#     print(exg_array.shape)
#     gradient_array = get_gradient_magnitude(exg_array)
    
#     print(gradient_array.shape)
#     #scaled_gradient_array = image_utils.scale_image(gradient_array, -1, 1, 0, 255, rint=True, np_type='uint8')

#     num_strongest_edges = int(percent_strongest_edge/100 * len(gradient_array))

#     print("num_strongest_edges", num_strongest_edges)

#     indices = largest_indices(gradient_array, num_strongest_edges)

#     mask = np.zeros(gradient_array.shape)
#     mask[indices] = 255

#     #kernel = np.ones((121, 121), np.uint8)
#     #kernel = np.ones((121, 121), np.uint8)
#     kernel = np.ones((51, 51), np.uint8)
#     return cv2.dilate(mask, kernel, iterations=1)




# def determine_default_segmentation_value(exg_array, image_array, fallback_thresh):

#     edge_mask = create_edge_mask(exg_array)

#     #test = np.copy(exg_array)
#     image_array[edge_mask == 0] = (0, 0, 0)
#     cv2.imwrite("my_test_exg_array.png", image_array)
#     cv2.imwrite("my_sample_edge_mask.png", edge_mask)


#     masked_exg_array = exg_array[edge_mask == 255]
#     print(exg_array.shape)
#     print(masked_exg_array.shape)
#     print(masked_exg_array)
#     #cv2.imwrite("my_masked_exg_array.png", masked_exg_array)

#     thresh_val, _ = cv2.threshold(masked_exg_array, 0, 255, cv2.THRESH_OTSU)

#     if thresh_val < fallback_thresh:
#         thresh_val = fallback_thresh

#     sel_val = range_map(thresh_val, 0, 255, -2, 2)
#     return sel_val
    

# def my_test():

#     exg = Image("usr/data/kaylie/image_sets/test_farm/test_field/2022-08-22/excess_green/1.png")
#     image = Image("usr/data/kaylie/image_sets/test_farm/test_field/2022-08-22/images/1.JPG")
#     exg_array = exg.load_image_array()
#     image_array = image.load_image_array()
#     sel_val = determine_default_segmentation_value(exg_array, image_array,  0)
#     print(sel_val)


# def determine_exg_threshold_with_detections():
#     detections = 


def update_vegetation_record_for_orthomosaic(image_set_dir, vegetation_record, excess_green_record, annotations):

    chunk_size = 5000
    image_name = list(annotations.keys())[0]


    if image_name not in vegetation_record or excess_green_record[image_name]["sel_val"] == vegetation_record[image_name]["sel_val"]:
        return



    image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
    image = Image(image_path)
    w, h = image.get_wh()
    
    ds = gdal.Open(image.image_path)

    chunk_coords_lst = []
    for i in range(0, h, chunk_size):
        for j in range(0, w, chunk_size):
            chunk_coords_lst.append([i, j, min(i+chunk_size, h), min(j+chunk_size, w)])

    # image_chunk = ds.ReadAsArray(i, j, chunk_size, chunk_size)

    results = Parallel(10)(
        delayed(get_updated_vegetation_percentages_for_chunk)(
            excess_green_record, annotations, ds, chunk_coords) for chunk_coords in chunk_coords_lst)


    vegetation_record[image_name] = {}
    vegetation_record[image_name]["sel_val"] = excess_green_record[image_name]["sel_val"]
    vegetation_record[image_name]["image"] = 0
    for region_key in ["training_regions", "test_regions"]:
        vegetation_record[image_name][region_key] = []
        for i in range(len(annotations[image_name][region_key])):
            vegetation_record[image_name][region_key].append(0)


    for result in results:
        vegetation_record[image_name]["image"] += result["chunk"]
        for region_key in ["training_regions", "test_regions"]:
            for i in range(len(annotations[image_name][region_key])):
                vegetation_record[image_name][region_key][i] += result[region_key][i]
        
    vegetation_record[image_name]["image"] = round(float((vegetation_record[image_name]["image"] / (w * h)) * 100), 2)
    for region_key in ["training_regions", "test_regions"]:
        for i, region in enumerate(annotations[image_name][region_key]):
            region_area = (region[2] - region[0]) * (region[3] - region[1])
            vegetation_record[image_name][region_key][i] = round(float((vegetation_record[image_name][region_key][i] / region_area) * 100), 2)

    return vegetation_record

def get_updated_vegetation_percentages_for_chunk(excess_green_record, annotations, ds, chunk_coords):

    chunk_array = ds.ReadAsArray(chunk_coords[1], 
                                 chunk_coords[0], 
                                 chunk_coords[3]-chunk_coords[1], 
                                 chunk_coords[2]-chunk_coords[0])

    exg_array = image_utils.excess_green(chunk_array)
    image_name = list(annotations.keys())[0]
    sel_val = excess_green_record[image_name]["sel_val"]
    chunk_vegetation_pixel_count = int(np.sum(exg_array > sel_val))

    result = {
        "chunk": chunk_vegetation_pixel_count,
        "training_regions": [],
        "test_regions": []
    }

    for region_key in ["training_regions", "test_regions"]:
        for region in annotations[image_name][region_key]:
            intersects, intersect_region = box_utils.get_intersection_rect(region, chunk_coords)

            if not intersects:
                result[region_key].append(0)
            else:
                intersect_vals = exg_array[intersect_region[0]:intersect_region[2], intersect_region[1]:intersect_region[3]]
                # intersect_area = (intersect_region[2] - intersect_region[0]) * (intersect_region[3] - intersect_region[1])
                vegetation_pixel_count = int(np.sum(intersect_vals > sel_val)) # / intersect_vals.size)
                result[region_key].append(vegetation_pixel_count)

    return result


def update_vegetation_record_for_image_set(image_set_dir, vegetation_record, excess_green_record, annotations):
    needs_update = []
    for image_name in annotations.keys():
        if image_name not in vegetation_record or excess_green_record[image_name]["sel_val"] != vegetation_record[image_name]["sel_val"]:
            needs_update.append(image_name)

    results = Parallel(10)(
        delayed(get_updated_vegetation_percentages_for_image)(image_set_dir, excess_green_record, annotations, image_name) for image_name in needs_update)

    print("new vegetation record results", results)

    for result in results:
        image_name = result[0]
        vegetation_results = result[1]
        vegetation_record[image_name] = {}
        vegetation_record[image_name]["sel_val"] = excess_green_record[image_name]["sel_val"]
        vegetation_record[image_name]["image"] = vegetation_results["image"]
        vegetation_record[image_name]["training_regions"] = vegetation_results["training_regions"]
        vegetation_record[image_name]["test_regions"] = vegetation_results["test_regions"]

    return vegetation_record



def get_updated_vegetation_percentages_for_image(image_set_dir, excess_green_record, annotations, image_name):

    image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
    image = Image(image_path)
    image_array = image.load_image_array()
    exg_array = image_utils.excess_green(image_array)
    sel_val = excess_green_record[image_name]["sel_val"]
    image_vegetation_percentage = round(float((np.sum(exg_array > sel_val) / exg_array.size) * 100), 2)
    
    result = {
        "image": image_vegetation_percentage,
        "training_regions": [],
        "test_regions": []
    }
    for region_key in ["training_regions", "test_regions"]:
        for region in annotations[image_name][region_key]:
            region_vals = exg_array[region[0]:region[2], region[1]:region[3]]
            vegetation_percentage = round(float((np.sum(region_vals > sel_val) / region_vals.size) * 100), 2)
            result[region_key].append(vegetation_percentage)

    return (image_name, result)


def create_excess_green_for_image(image_set_dir, image_path):
    image_name = os.path.basename(image_path)[:-4]

    image = Image(image_path)
    #image_array = image.load_image_array().astype(np.int64)
    #exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    #exg_array = exg_array.astype(np.int64)
    exg_array = image_utils.excess_green(image)
    
    min_val = round(float(np.min(exg_array)), 2)
    max_val = round(float(np.max(exg_array)), 2)
    #sel_val = round(float(threshold_otsu(exg_array)), 2)
    sel_val = round((min_val + max_val) / 2, 2)
    percent_vegetation = round(float((np.sum(exg_array > sel_val) / exg_array.size) * 100), 2)

    exg_array = image_utils.scale_image(exg_array, -2, 2, 0, 255, np_type=np.uint8)

    # sel_val = determine_default_segmentation_value(exg_array, sel_val)
    


    

    


    cv2.imwrite(os.path.join(image_set_dir, "excess_green", image_name + ".png"), exg_array)


    #lockfile = lock.lock_acquire("excess_green_lock")

    # if os.path.exists(record_path):
    #     record = json_io.load_json(record_path)
    # else:
        
    # record[image_name] = {
    #     "min_val": min_val,
    #     "max_val": max_val,
    #     "sel_val": sel_val,
    #     "ground_cover_percentage": percent_vegetation
    # }

    return (image_name, {
        "min_val": min_val,
        "max_val": max_val,
        "sel_val": sel_val,
        "ground_cover_percentage": percent_vegetation
    })



def create_excess_green_for_image_set(image_set_dir): #, image_name):


    #image_set_dir = "usr/data/image_sets/test_farm/test_field/2022-06-07"
    #image_set_dir = "usr/data/image_sets/test_farm_2/test_field_2/2022-06-08"

    # record = {}
    # for image_path in glob.glob(os.path.join(image_set_dir, "images", "*")):
    # #print(image_path)
    #     image_name = os.path.basename(image_path)[:-4]
    
    image_paths = glob.glob(os.path.join(image_set_dir, "images", "*")) #image_name + ".*"))
    #assert len(image_paths) == 1
    #image_path = image_paths[0]
    record_path = os.path.join(image_set_dir, "excess_green", "record.json")
    record = {}

    rec_tups = Parallel(10)(
        delayed(create_excess_green_for_image)(image_set_dir, image_path) for image_path in image_paths)

    for rec_tup in rec_tups:
        record[rec_tup[0]] = rec_tup[1]


    # for image_path in image_paths:

    #     image_name = os.path.basename(image_path)[:-4]

    #     image = Image(image_path)
    #     #image_array = image.load_image_array().astype(np.int64)
    #     #exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    #     #exg_array = exg_array.astype(np.int64)
    #     exg_array = image_utils.excess_green(image)
        
    #     min_val = round(float(np.min(exg_array)), 2)
    #     max_val = round(float(np.max(exg_array)), 2)
    #     #sel_val = round(float(threshold_otsu(exg_array)), 2)
    #     sel_val = round((min_val + max_val) / 2, 2)
    #     percent_vegetation = round(float((np.sum(exg_array > sel_val) / exg_array.size) * 100), 2)

    #     exg_array = image_utils.scale_image(exg_array, -2, 2, 0, 255, np_type=np.uint8)

        
    #     # print(np.min(exg_array))
    #     # print(np.max(exg_array))



    #     cv2.imwrite(os.path.join(image_set_dir, "excess_green", image_name + ".png"), exg_array)


    #     #lockfile = lock.lock_acquire("excess_green_lock")

    #     # if os.path.exists(record_path):
    #     #     record = json_io.load_json(record_path)
    #     # else:
            
    #     record[image_name] = {
    #         "min_val": min_val,
    #         "max_val": max_val,
    #         "sel_val": sel_val,
    #         "ground_cover_percentage": percent_vegetation
    #     }

    json_io.save_json(record_path, record)

        #lock.lock_release(lockfile)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    #parser.add_argument("image_name", type=str)
    args = parser.parse_args()

    create_excess_green_for_image_set(args.image_set_dir) #, args.image_name)