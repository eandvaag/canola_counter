import os


from io_utils import json_io
from image_set import ImgSet


def create_and_save_class_map_data(training_config):

    class_map = {}
    class_num = 0
    for training_img_set_conf in training_config["training_image_sets"]:
        img_set = ImgSet(training_img_set_conf["trial_name"], training_img_set_conf["mission_date"])
        for class_name in img_set.class_map.keys():
            if class_name not in class_map:
                class_map[class_name] = class_num
                class_num += 1


    reverse_class_map = {v: k for k, v in class_map.items()}
    num_classes = len(class_map.keys())


    instance_uuid = training_config["instance_uuid"]
    class_map_path = os.path.join("usr", "data", "models", instance_uuid, "class_map.json")
    class_map_data = {
        "class_map": class_map,
        "reverse_class_map": reverse_class_map,
        "num_classes": num_classes
    }
    json_io.save_json(class_map_path, class_map_data)



def load_class_map_data(instance_uuid):
    class_map_path = os.path.join("usr", "data", "models", instance_uuid, "class_map.json")
    class_map_data = json_io.load_json(class_map_path)

    # json keys must be strings
    reverse_class_map = {int(k): v for k, v in class_map_data["reverse_class_map"].items()}
    class_map_data["reverse_class_map"] = reverse_class_map

    return class_map_data