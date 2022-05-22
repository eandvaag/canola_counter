import os
import glob
import tqdm
import argparse
# import time

from image_set import Image
from io_utils import json_io


# def add_camera_specs(image_set_dir):

#     images_dir = os.path.join(image_set_dir, "images")
#     image_paths = glob.glob(os.path.join(images_dir, "*"))
#     # attempts = 0
#     # while len(image_paths) == 0 and attempts < 10:
#     #     time.sleep(5)
#     #     attempts += 1
#     #     print("sleeping")
    

#     print("{}: Make: {} | Model: {}".format(image_set_dir, make, model))
#     metadata_dir = os.path.join(image_set_dir, "metadata")
#     if not os.path.exists(metadata_dir):
#         os.makedirs(metadata_dir)
#     camera_path = os.path.join(image_set_dir, "metadata", "camera.json")
#     json_io.save_json(camera_path, camera_info)


def extract_metadata(image_set_dir, flight_height=None):

    image_set_metadata = {}

    images_dir = os.path.join(image_set_dir, "images")
    metadata_dir = os.path.join(image_set_dir, "metadata")

    if flight_height is None:
        flight_height = "unknown"
    image_set_metadata = {
        "flight_height": flight_height,
        "images": {},
        "missing": {
            "area_m2": False,
            # "height_m": False,
            "latitude": False,
            "longitude": False
        }
    }

    for image_path in tqdm.tqdm(glob.glob(os.path.join(images_dir, "*")), desc="Extracting metadata"):

        image_name = os.path.basename(image_path)[:-4]

        image = Image(image_path)

        md = image.get_metadata()
        if flight_height is None:
            area_m2 = "unknown"
            image_set_metadata["missing"]["area_m2"] = True
        else:
            try:
                area_m2 = image.get_area_m2(md, flight_height)
            except:
                area_m2 = "unknown"
                image_set_metadata["missing"]["area_m2"] = True

        # try:
        #     height_m = image.get_height_m(md)
        # except:
        #     height_m = "unknown"
        #     image_set_metadata["missing"]["height_m"]  = True


        if "EXIF:GPSLatitude" in md:
            gps_latitude = md["EXIF:GPSLatitude"]
        else:
            gps_latitude = "unknown"
            image_set_metadata["missing"]["latitude"] = True

        if "EXIF:GPSLongitude" in md:
            gps_longitude = md["EXIF:GPSLongitude"]
        else:
            gps_longitude = "unknown"
            image_set_metadata["missing"]["longitude"] = True


        image_set_metadata["images"][image_name] = {
            "latitude": gps_latitude,
            "longitude": gps_longitude,
            #"height_m": height_m,
            "area_m2": area_m2
        }


    image_path = glob.glob(os.path.join(images_dir, "*"))[0]

    image = Image(image_path)
    md = image.get_metadata()
    if "EXIF:Make" in md:
        make = md["EXIF:Make"]
    else:
        make = "unknown"
    if "EXIF:Model" in md:
        model = md["EXIF:Model"]
    else:
        model = "unknown"
    camera_info = {
        "make": make,
        "model": model
    }

    image_set_metadata["camera_info"] = camera_info


    
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_path = os.path.join(metadata_dir, "metadata.json")
    #print("writing metadata for {}. metadata_missing? {}".format(image_set_dir, image_set_metadata["missing"]))
    #json_io.print_json(image_set_metadata)
    json_io.save_json(metadata_path, image_set_metadata)


def extract_metadata_for_all_image_sets():
    image_set_root = os.path.join("usr", "data", "image_sets")

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                extract_metadata(mission_path, flight_height=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    parser.add_argument("--flight_height", type=float)

    args = parser.parse_args()
    image_set_dir = args.image_set_dir
    flight_height = args.flight_height

    # print("flight_height: {}".format(flight_height))

    # add_camera_specs(image_set_dir)
    # add_camera_specs_for_all_image_sets()

    extract_metadata(image_set_dir, flight_height=flight_height)
