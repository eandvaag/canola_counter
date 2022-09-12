import os
import shutil
import glob
import math as m
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import time


RESULT_LIFETIME = 2 * 3600

from sklearn.metrics import pairwise_distances


# from image_set import DataSet
from io_utils import json_io, exif_io

def range_map(old_val, old_min, old_max, new_min, new_max):
    new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_val


def similarity_map(username, farm_name, field_name, mission_date):

    # farm_name = dataset.farm_name
    # field_name = dataset.field_name
    # mission_date = dataset.mission_date

    
    image_set_dir =  os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date)

    annotations_path = os.path.join(image_set_dir,
                        "annotations", "annotations_w3c.json")
    
    annotations = json_io.load_json(annotations_path)

    annotated_features = []
    unannotated_features = []
    feature_counts = []
    unannotated_image_names = []
    for image_name in annotations.keys():
        features_path = os.path.join(image_set_dir, "features", image_name + ".npy")
        features = np.load(features_path)

        if annotations[image_name]["status"] == "completed":
            annotated_features.extend(features.tolist())
        else:
            unannotated_features.extend(features.tolist())
            feature_counts.append(features.shape[0])
            unannotated_image_names.append(image_name)

    print("calculating pairwise distances")
    distances = pairwise_distances(unannotated_features, annotated_features)
    print("finished")
    #print("feature_counts", feature_counts)
    scores = []
    prev_feature_sum = 0
    for i, feature_count in enumerate(feature_counts):
        sel = distances[prev_feature_sum:prev_feature_sum + feature_count, :]
        sel_sum = np.sum(sel)
        #print("adding {}, score: {}".format(unannotated_image_names[i], sel_sum))
        #res[unannotated_image_names[i]] = sel_sum
        scores.append(sel_sum)
        prev_feature_sum = prev_feature_sum + feature_count

    #json_io.print_json(res)
    scores = np.array(scores)
    inds = np.argsort(scores)
    sorted_scores = scores[inds]
    sorted_names = np.array(unannotated_image_names)[inds]
    for i, (score, name) in enumerate(zip(sorted_scores, sorted_names)):
        print("{}: Image: {} | Score: {}".format(i, name, score))


def create_plot(grid_z0, extent, vmin, vmax, cmap, out_path):
    plt.figure()
    plt.imshow(grid_z0.T, extent=extent, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    #out_path = os.path.join(out_dir, "annotated_map.svg")
    plt.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0)


def create_interpolation_map(username, farm_name, field_name, mission_date, annotations_path, out_dir, map_download_uuid,
                             interpolation="linear", completed_only=False, pred_path=None, comparison_type="side_by_side"):

    # farm_name = dataset.farm_name
    # field_name = dataset.field_name
    # mission_date = dataset.mission_date

    #annotations_path = os.path.join("usr", "data", "image_sets",
    #                    farm_name, field_name, mission_date,
    #                    "annotations", "annotations_w3c.json")
    annotations = json_io.load_json(annotations_path)

    #images_root = os.path.join("usr", "data", "image_sets",
    #                    farm_name, field_name, mission_date, "images")

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)


    if pred_path is not None:
        predictions = json_io.load_json(pred_path)


    if (metadata["missing"]["latitude"] or metadata["missing"]["longitude"]) or metadata["camera_height"] == "":# or metadata["missing"]["area_m2"]:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)


    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]

    if make not in camera_specs:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    if model not in camera_specs[make]:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    camera_entry = camera_specs[make][model]
    sensor_height = camera_entry["sensor_height"]
    sensor_width = camera_entry["sensor_width"]
    focal_length = camera_entry["focal_length"]

    camera_height = metadata["camera_height"]

    all_points = []
    predicted_values = []
    completed_points = []
    annotated_values = []
    for image_name in annotations.keys():
        lon = metadata["images"][image_name]["longitude"]
        lat = metadata["images"][image_name]["latitude"]

        all_points.append([lon, lat])

        status = annotations[image_name]["status"]

        gsd_h = (camera_height * sensor_height) / (focal_length * metadata["images"][image_name]["height_px"])
        gsd_w = (camera_height * sensor_width) / (focal_length * metadata["images"][image_name]["width_px"])

        gsd = min(gsd_h, gsd_w)

        image_height_m = metadata["images"][image_name]["height_px"] * gsd
        image_width_m = metadata["images"][image_name]["width_px"] * gsd

        area_m2 = image_width_m * image_height_m

        if status == "completed_for_training" or status == "completed_for_testing":

            #print("image_name", image_name)
            annotated_value = len(annotations[image_name]["annotations"]) / area_m2
            completed_points.append([lon, lat])
            annotated_values.append(annotated_value)

        if pred_path is not None:
            if not completed_only or (status == "completed_for_training" or status == "completed_for_testing"):  
                #predicted_value = len(predictions["image_predictions"][image_name]["pred_image_abs_boxes"]) / metadata["images"][image_name]["area_m2"]
                v = 0
                for annotation in predictions[image_name]["annotations"]:
                    for b in annotation["body"]:
                        if b["purpose"] == "score" and float(b["value"]) >= 0.5:
                            v += 1
                
                #predicted_value = len(predictions[image_name]["annotations"]) 
                predicted_value = v / area_m2
                predicted_values.append(predicted_value)

    # print("predicted_values", predicted_values)
    # print("annotated_values", annotated_values)


    all_points = np.array(all_points)
    completed_points = np.array(completed_points)
    annotated_values = np.array(annotated_values)

    num_completed = annotated_values.size 
    if num_completed < 3 and pred_path is None:
        return

    if pred_path is not None:
        predicted_values = np.array(predicted_values)

    min_x = np.min(all_points[:,0])
    max_x = np.max(all_points[:,0])

    min_y = np.min(all_points[:,1])
    max_y = np.max(all_points[:,1])

    #points[:,0] = points[:,0] / max_x
    #points[:,1] = points[:,1] / max_y

    all_points[:,0] = range_map(all_points[:,0], min_x, max_x, 0, 1)
    all_points[:,1] = range_map(all_points[:,1], min_y, max_y, 0, 1)

    if num_completed >= 3:
        completed_points[:,0] = range_map(completed_points[:,0], min_x, max_x, 0, 1)
        completed_points[:,1] = range_map(completed_points[:,1], min_y, max_y, 0, 1)


        grid_x, grid_y = np.mgrid[np.min(completed_points[:,0]):np.max(completed_points[:,0]):1000j, 
                                np.min(completed_points[:,1]):np.max(completed_points[:,1]):1000j]

        grid_z0 = griddata(completed_points, annotated_values, (grid_x, grid_y), method=interpolation)


        extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
                np.min(completed_points[:,1]), np.max(completed_points[:,1]))


    all_grid_x, all_grid_y = np.mgrid[np.min(all_points[:,0]):np.max(all_points[:,0]):1000j, 
                              np.min(all_points[:,1]):np.max(all_points[:,1]):1000j]                        


    if pred_path is not None:

        if completed_only:
            pred_grid_z0 = griddata(completed_points, predicted_values, (grid_x, grid_y), method=interpolation)

            pred_extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
                        np.min(completed_points[:,1]), np.max(completed_points[:,1]))
        else:
            pred_grid_z0 = griddata(all_points, predicted_values, (all_grid_x, all_grid_y), method=interpolation)

            pred_extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
                           np.min(all_points[:,1]), np.max(all_points[:,1]))


    if comparison_type == "side_by_side":

        if pred_path is None:
            vmax = m.ceil(np.max(annotated_values))
        elif num_completed < 3:
            vmax = m.ceil(np.max(predicted_values))
        else:
            vmax = m.ceil(max(np.max(predicted_values), np.max(annotated_values)))

        vmin = 0

        colors = ["wheat", "forestgreen"]
        cmap = LinearSegmentedColormap.from_list("mycmap", colors)


        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if num_completed >= 3:

            out_path = os.path.join(out_dir, map_download_uuid + "_annotated_map.svg")
            create_plot(grid_z0, extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)


        if pred_path is not None:

            out_path = os.path.join(out_dir, map_download_uuid + "_predicted_map.svg")
            create_plot(pred_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)
    
    # elif comparison_type == "diff":

    #     diff_grid_z0 = pred_grid_z0 - grid_z0

    #     vmin = m.floor(np.nanmin(diff_grid_z0))
    #     vmax = m.ceil(np.nanmax(diff_grid_z0))
    #     vlim = max(abs(vmin), abs(vmax))
    #     vmin = -vlim
    #     vmax = vlim


    #     colors = ["royalblue", "oldlace", "tomato"] #"ghostwhite", "tomato"]
    #     #colors = ["royalblue", "whitesmoke", "tomato"]  #"tomato"]
    #     cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    #     out_path = os.path.join(out_dir, "difference_map.svg")
    #     create_plot(diff_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)


    min_max_rec = {
        "vmin": vmin,
        "vmax": vmax
    }
    min_max_rec_path = os.path.join(out_dir, map_download_uuid + "_min_max_rec.json")
    json_io.save_json(min_max_rec_path, min_max_rec)



def remove_all_maps():
    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)
                maps_path = os.path.join(mission_path, "maps")
                if os.path.exists(maps_path):
                    shutil.rmtree(maps_path)


# def test():
#     ds = DataSet({"farm_name": "UNI", "field_name": "LowN1", "mission_date": "2021-06-07"})
#     predictions_path = os.path.join("usr", "data", "results", "UNI", "LowN1", "2021-06-07", 
#     "7a404942-dcd6-414c-a4aa-fcc39db4425b", "b4a13b7f-5e4d-4818-8c14-ea4095900fb3", "predictions.json")
#     predictions = json_io.load_json(predictions_path)
#     create_interpolation_map(ds, predictions)

def remove_old_maps(out_dir):
    if os.path.exists(out_dir):
        for f_path in glob.glob(os.path.join(out_dir, "*")):
            alive_time = time.time() - os.path.getmtime(f_path)
            if alive_time > RESULT_LIFETIME:
                os.remove(f_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str)
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    parser.add_argument("annotations_path", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("map_download_uuid", type=str)
    #parser.add_argument('-density', action='store_true')

    parser.add_argument('-nearest', action='store_true')
    parser.add_argument('-completed_only', action='store_true')
    parser.add_argument('-pred_path', type=str)
    # parser.add_argument('-diff', action='store_true')
    args = parser.parse_args()
    

    print("running map builder")
    print(args.username)
    print(args.farm_name)
    print(args.field_name)
    print(args.mission_date) 
    print(args.annotations_path) 
    print(args.out_dir)
    print(args.map_download_uuid)
    print(args.pred_path)  
    # dataset = DataSet({
    #     "farm_name": args.farm_name,
    #     "field_name": args.field_name,
    #     "mission_date": args.mission_date
    # })

    # if args.density:
    #     metric = "density"
    # else:
    #     metric = "count"

    if args.nearest:
        interpolation = "nearest"
    else:
        interpolation = "linear"

    # if args.diff:
    #     comparison_type = "diff"
    # else:
    comparison_type = "side_by_side"

    remove_old_maps(args.out_dir)


    create_interpolation_map(args.username,
                            args.farm_name,
                            args.field_name,
                            args.mission_date,
                            args.annotations_path,
                            args.out_dir, 
                            args.map_download_uuid,
                            interpolation=interpolation, 
                            completed_only=args.completed_only,
                            pred_path=args.pred_path,
                            comparison_type=comparison_type)