import os
import shutil
import glob
import math as m
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

from image_set import DataSet
from io_utils import json_io, exif_io

def range_map(old_val, old_min, old_max, new_min, new_max):
    new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_val


def create_interpolation_map(dataset, out_dir, interpolation="linear", completed_only=False, pred_path=None):

    farm_name = dataset.farm_name
    field_name = dataset.field_name
    mission_date = dataset.mission_date

    annotations_path = os.path.join("usr", "data", "image_sets",
                        farm_name, field_name, mission_date,
                        "annotations", "annotations_w3c.json")
    annotations = json_io.load_json(annotations_path)

    images_root = os.path.join("usr", "data", "image_sets",
                        farm_name, field_name, mission_date, "images")

    metadata_path = os.path.join("usr", "data", "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)


    if pred_path is not None:
        predictions = json_io.load_json(pred_path)


    if (metadata["missing"]["latitude"] or metadata["missing"]["longitude"]) or metadata["missing"]["area_m2"]:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    # data = {}
    # for image_name in annotations.keys():
    #     image_path = glob.glob(os.path.join(images_root, image_name + ".*"))[0]
    #     #metadata = exif_io.get_exif_metadata(image_path)


    #     if pred_path is None:
    #         predicted_density = None
    #     else:
    #         predicted_density = len(predictions["image_predictions"][image_name]["pred_image_abs_boxes"])

    #     status = annotations[image_name]["status"]
    #     data[image_name] = {
    #         #"predicted_count": predicted_count,
    #         "latitude": metadata["images"][image_name]["latitude"], # ["EXIF:GPSLatitude"],
    #         "longitude": metadata["images"][image_name]["longitude"], #["EXIF:GPSLongitude"],
    #         "status": status,
    #         "annotated_density": len(annotations[image_name]["annotations"]) / metadata["images"][image_name]["area_m2"]
    #     }

    #     # predicted_values = .append()

    #     #gps_latitude = 
    #     #gps_longitude = 
    #     #points.append([gps_longitude, gps_latitude])

        

    #     # if status == "completed":
    #     #     annotated_count = len(annotations[image_name]["annotations"])

    #     #     completed_points.append([gps_longitude, gps_latitude])
    #     #     values.append(annotated_count)

    all_points = []
    predicted_values = []
    completed_points = []
    annotated_values = []
    for image_name in annotations.keys():
        lon = metadata["images"][image_name]["longitude"]
        lat = metadata["images"][image_name]["latitude"]

        all_points.append([lon, lat])

        status = annotations[image_name]["status"]
        if status == "completed":
            annotated_value = len(annotations[image_name]["annotations"]) / metadata["images"][image_name]["area_m2"]
            completed_points.append([lon, lat])
            annotated_values.append(annotated_value)

        if pred_path is not None:
            if not completed_only or status == "completed":  
                predicted_value = len(predictions["image_predictions"][image_name]["pred_image_abs_boxes"]) / metadata["images"][image_name]["area_m2"]
                predicted_values.append(predicted_value)




    #print(completed_points)
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
    completed_points[:,0] = range_map(completed_points[:,0], min_x, max_x, 0, 1)
    completed_points[:,1] = range_map(completed_points[:,1], min_y, max_y, 0, 1)


    grid_x, grid_y = np.mgrid[np.min(completed_points[:,0]):np.max(completed_points[:,0]):1000j, 
                              np.min(completed_points[:,1]):np.max(completed_points[:,1]):1000j]
    #grid_x, grid_y = np.mgrid[0:1:100j, 
    #                          0:1:100j]
    grid_z0 = griddata(completed_points, annotated_values, (grid_x, grid_y), method=interpolation)
    # print("grid_x", grid_x)
    # print("grd_y", grid_y)
    # print("grid_z0", grid_z0)

    extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
              np.min(completed_points[:,1]), np.max(completed_points[:,1]))

    print("grid_z0.shape", grid_z0.shape)
    
    if pred_path is None:
        max_val = m.ceil(np.max(annotated_values))
    elif num_completed < 3:
        max_val = m.ceil(np.max(predicted_values))
    else:
        max_val = m.ceil(max(np.max(predicted_values), np.max(annotated_values)))



    colors = ["wheat", "forestgreen"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    if num_completed >= 3:
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(grid_z0.T, extent=extent, origin="lower", vmin=0, vmax=max_val, cmap=cmap)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "annotated_map.svg")

        plt.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0)


    if pred_path is not None:

        if completed_only:
            pred_grid_z0 = griddata(completed_points, predicted_values, (grid_x, grid_y), method=interpolation)

            pred_extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
                           np.min(completed_points[:,1]), np.max(completed_points[:,1]))
        else:
            pred_grid_z0 = griddata(all_points, predicted_values, (grid_x, grid_y), method=interpolation)

            pred_extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
                           np.min(all_points[:,1]), np.max(all_points[:,1]))


        fig = plt.figure(figsize=(5, 5))
        plt.imshow(pred_grid_z0.T, extent=pred_extent, origin="lower", vmin=0, vmax=max_val, cmap=cmap)

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "predicted_map.svg")

        plt.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0)



    # if predictions is not None:

    #     all_predicted_values = np.array(all_predicted_values)
    #     grid_x, grid_y = np.mgrid[np.min(all_points[:,0]):np.max(all_points[:,0]):1000j, 
    #                           np.min(all_points[:,1]):np.max(all_points[:,1]):1000j]

    #     grid_z0 = griddata(all_points, all_predicted_values, (grid_x, grid_y), method="linear")

    #     extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
    #               np.min(all_points[:,1]), np.max(all_points[:,1]))

    #     plt.figure()
    #     plt.imshow(grid_z0.T, extent=extent, origin="lower", vmin=0, vmax=max_val, cmap=cmap)
    #     plt.plot(all_points[:,0], all_points[:,1], 'k.')
    #     #plt.plot(completed_points[:,0], completed_points[:,1], 'r.') #, extent=extent, origin="lower")

    #     plt.xlim([min_x-0.0001, max_x+0.0001])
    #     plt.ylim([min_y-0.0001, max_y+0.0001])
    #     plt.colorbar()

    #     plt.savefig("predicted_map.svg")


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


def test():
    ds = DataSet({"farm_name": "UNI", "field_name": "LowN1", "mission_date": "2021-06-07"})
    predictions_path = os.path.join("usr", "data", "results", "UNI", "LowN1", "2021-06-07", 
    "7a404942-dcd6-414c-a4aa-fcc39db4425b", "b4a13b7f-5e4d-4818-8c14-ea4095900fb3", "predictions.json")
    predictions = json_io.load_json(predictions_path)
    create_interpolation_map(ds, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    parser.add_argument("out_dir", type=str)
    #parser.add_argument('-density', action='store_true')
    parser.add_argument('-nearest', action='store_true')
    parser.add_argument('-completed_only', action='store_true')
    parser.add_argument('-pred_path', type=str)
    args = parser.parse_args()
    
    dataset = DataSet({
        "farm_name": args.farm_name,
        "field_name": args.field_name,
        "mission_date": args.mission_date
    })

    # if args.density:
    #     metric = "density"
    # else:
    #     metric = "count"

    if args.nearest:
        interpolation = "nearest"
    else:
        interpolation = "linear"


    create_interpolation_map(dataset, 
                            args.out_dir, 
                            interpolation=interpolation, 
                            completed_only=args.completed_only,
                            pred_path=args.pred_path)