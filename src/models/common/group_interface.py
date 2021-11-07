import os
import shutil
import logging
import copy
import randomname
import uuid

from io_utils import json_io
from models.common import model_interface



def create_model_configs(group_config, index):

    #excluded_inference_keys = ["trial_name", "mission_date", "dataset_names"]

    arch_config = copy.deepcopy(group_config["arch_config"])
    training_config =  copy.deepcopy(group_config["training_config"])
    inference_config =  copy.deepcopy(group_config["inference_config"])


    for i in range(len(group_config["variation_config"]["param_values"][index])):
        param_config = group_config["variation_config"]["param_configs"][i]
        param_name = group_config["variation_config"]["param_names"][i]
        param_val = group_config["variation_config"]["param_values"][index][i]

        if param_config == "arch":
            arch_config[param_name] = param_val
        elif param_config == "training":
            training_config[param_name] = param_val
        elif param_config == "inference":
            #if param_name in excluded_inference_keys:
            #    raise RuntimeError("Error: cannot vary an excluded key. Your key: {}. Excluded keys: {}.".format(
            #                        param_name, excluded_inference_keys))
            inference_config[param_name] = param_val
        else:
            raise RuntimeError("Invalid value for 'param_config': {} " + \
                               "(Expected 'arch', 'training', or 'inference').".format(param_config))

    return arch_config, training_config, inference_config



def generate_model_names(num):
    return [randomname.get_name() for _ in range(num)]


def generate_model_uuids(num):
    return [str(uuid.uuid4()) for _ in range(num)]



def run_group(req_args):

    logger = logging.getLogger(__name__)


    group_uuid = str(uuid.uuid4())
    group_config = copy.deepcopy(req_args)
    group_config["group_uuid"] = group_uuid
    group_config["model_instance_uuids"] = []
    group_config["model_instance_names"] = []
    group_config["model_info"] = {}

    group_name = group_config["group_name"]
    group_description = group_config["group_description"]


    logger.info("Started processing group '{}' (uuid: {}).".format(group_name, group_uuid))

    num_models = len(group_config["variation_config"]["param_values"]) * group_config["variation_config"]["replications"]

    group_config["model_instance_uuids"] = generate_model_uuids(num_models)
    group_config["model_instance_names"] = generate_model_names(num_models)


    # Save the group before training the models. If aborted, the group can now be destroyed.
    group_config_path = os.path.join("usr", "data", "groups", group_uuid + ".json")
    json_io.save_json(group_config_path, group_config)


    model_index = 0
    for param_index in range(len(group_config["variation_config"]["param_values"])):

        arch_config, training_config, inference_config = create_model_configs(group_config, param_index)

        for _ in range(group_config["variation_config"]["replications"]):


            instance_uuid = group_config["model_instance_uuids"][model_index]
            instance_name = group_config["model_instance_names"][model_index]

            arch_config["instance_uuid"] = instance_uuid
            arch_config["instance_name"] = instance_name
            
            training_config["instance_uuid"] = instance_uuid
            training_config["instance_name"] = instance_name

            inference_config["instance_uuid"] = instance_uuid
            inference_config["instance_name"] = instance_name

            model_interface.create_model(arch_config)
            model_interface.train_model(training_config)
            pred_info = model_interface.run_inference(inference_config)


            for p in pred_info:

                trial_name = p["trial_name"]
                mission_date = p["mission_date"]
                dataset_name = p["dataset_name"]
                pred_dirname = p["prediction_dirname"]

                if trial_name not in group_config["model_info"]:
                    group_config["model_info"][trial_name] = {}
                if mission_date not in group_config["model_info"][trial_name]:
                    group_config["model_info"][trial_name][mission_date] = {}
                if dataset_name not in group_config["model_info"][trial_name][mission_date]:
                    group_config["model_info"][trial_name][mission_date][dataset_name] = {
                        "instance_uuids": [],
                        "instance_names": [],
                        "prediction_dirnames": []
                    }

                group_config["model_info"][trial_name][mission_date][dataset_name]["instance_uuids"].append(instance_uuid)
                group_config["model_info"][trial_name][mission_date][dataset_name]["instance_names"].append(instance_name)
                group_config["model_info"][trial_name][mission_date][dataset_name]["prediction_dirnames"].append(pred_dirname)

            model_index += 1

    json_io.save_json(group_config_path, group_config)

    key = group_uuid
    value = {"group_uuid": group_uuid,
             "group_name": group_name}

    for trial_name in group_config["model_info"].keys():
        for mission_date in group_config["model_info"][trial_name].keys():
            for dataset_name in group_config["model_info"][trial_name][mission_date].keys():
                model_interface.add_entry_to_inference_record(trial_name, mission_date, dataset_name, "groups", key, value)


    logger.info("Finished processing group '{}' (uuid: {}).".format(group_config["group_name"], group_uuid))




def destroy_group(req_args):

    logger = logging.getLogger(__name__)

    group_uuid = req_args["group_uuid"]

    group_config_path = os.path.join("usr", "data", "groups", group_uuid + ".json")
    group_config = json_io.load_json(group_config_path)
    group_name = group_config["group_name"]


    logger.info("Starting to destroy group '{}' (uuid: {}).".format(group_name, group_uuid))

    inference_lookup_path =  os.path.join("usr", "data", "records", "inference_lookup.json")
    inference_lookup = json_io.load_json(inference_lookup_path)

    for trial_name in group_config["model_info"].keys():
        for mission_date in group_config["model_info"][trial_name].keys():
            for dataset_name in group_config["model_info"][trial_name][mission_date].keys():


                d = group_config["model_info"][trial_name][mission_date][dataset_name]

                inference_entry = inference_lookup["inference_runs"][trial_name][mission_date][dataset_name]

                for model_uuid in d["instance_uuids"]:
                    
                    if inference_entry is not None:
                        if model_uuid in inference_entry["models"]:
                            del inference_entry["models"][model_uuid]
                        if group_uuid in inference_entry["groups"]:
                            del inference_entry["groups"][group_uuid]


                    model_dir = os.path.join("usr", "data", "models", model_uuid)
                    shutil.rmtree(model_dir)
    

    json_io.save_json(inference_lookup_path, inference_lookup)
    os.remove(group_config_path)


    logger.info("Finished destroying group '{}' (uuid: {}).".format(group_name, group_uuid))

