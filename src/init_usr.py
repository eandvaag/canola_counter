import os
import glob

from io_utils import json_io


init_cameras = {
    "GoPro": {
        "HERO9 Black": {
            "sensor_width": 6.17,
            "sensor_height": 4.55,
            "focal_length": 3
        },
        "HERO6 Black": {
            "sensor_width": 6.17,
            "sensor_height": 4.55,
            "focal_length": 3
        }
    },
    "Hasselblad": {
        "L1D-20c": {
            "sensor_width": 13.2,
            "sensor_height": 8.8,
            "focal_length": 10.3
        }
    },
    "Phase One": {
        "iXU1000": {
            "sensor_width": 53.4,
            "sensor_height": 40.0,
            "focal_length": 55.0
        }
    }
}


def init_usr(username):

    usr_dir = os.path.join("usr", "data", username)
    cameras_dir = os.path.join(usr_dir, "cameras")
    image_sets_dir = os.path.join(usr_dir, "image_sets")


    os.makedirs(usr_dir)
    os.makedirs(cameras_dir)
    os.makedirs(image_sets_dir)

    cameras_path = os.path.join(cameras_dir, "cameras.json")
    json_io.save_json(cameras_path, init_cameras)


def update_init_cameras(replace=False):

    for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
        cameras_path = os.path.join(usr_dir, "cameras", "cameras.json")

        if os.path.exists(cameras_path):

            if replace:
                json_io.save_json(cameras_path, init_cameras)

            else:
                usr_cameras = json_io.load_json(cameras_path)
                for make in init_cameras.keys():
                    if make not in usr_cameras:
                        usr_cameras[make] = {}
                    for model in init_cameras[make].keys():
                        usr_cameras[make][model] = init_cameras[make][model]

                json_io.save_json(cameras_path, usr_cameras)