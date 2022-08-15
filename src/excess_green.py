import argparse
import os
import glob
import cv2
import numpy as np
from joblib import Parallel, delayed

from skimage.filters import threshold_otsu

from image_set import Image
import image_utils
from io_utils import json_io
import lock


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