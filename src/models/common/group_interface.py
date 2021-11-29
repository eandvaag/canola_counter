import os
import shutil
import logging
import copy
import randomname
import uuid

from io_utils import json_io
from models.common import model_interface, inference_record_io


ADDED_TO_GROUP = "added_to_group"
FINISHED_ARCH = "finished_arch"
FINISHED_TRAINING = "finished_training"
FINISHED_INFERENCE = "finished_inference"



def create_model_configs(group_config, model_index):

    group_uuid = group_config["group_uuid"]
    group_name = group_config["group_name"]

    m = group_config["model_info"][model_index]

    model_uuid = m["model_uuid"]
    model_name = m["model_name"]

    arch_config = copy.deepcopy(group_config["arch_config"])
    training_config =  copy.deepcopy(group_config["training_config"])
    inference_config =  copy.deepcopy(group_config["inference_config"])

    arch_config["model_uuid"] = model_uuid
    arch_config["model_name"] = model_name
    arch_config["group_uuid"] = group_uuid
    arch_config["group_name"] = group_name
    
    training_config["model_uuid"] = model_uuid
    training_config["model_name"] = model_name
    training_config["group_uuid"] = group_uuid
    training_config["group_name"] = group_name

    inference_config["model_uuid"] = model_uuid
    inference_config["model_name"] = model_name
    inference_config["group_uuid"] = group_uuid
    inference_config["group_name"] = group_name


    if "variation_config" in group_config:
        param_index = model_index // group_config["replications"]

        for i in range(len(group_config["variation_config"]["param_values"][param_index])):
            param_config = group_config["variation_config"]["param_configs"][i]
            param_name = group_config["variation_config"]["param_names"][i].split("/")
            param_val = group_config["variation_config"]["param_values"][param_index][i]

            if param_config == "arch":
                rec = arch_config
            elif param_config == "training":
                rec = training_config
            elif param_config == "inference":
                rec = inference_config
            else:
                raise RuntimeError("Invalid value for 'param_config': {} " + \
                                   "(Expected 'arch', 'training', or 'inference').".format(param_config))

            for i in range(len(param_name)-1):
                rec = rec[param_name[i]]
            rec[param_name[-1]] = param_val

    return arch_config, training_config, inference_config



def generate_model_names(num):
    return [randomname.get_name() for _ in range(num)]


def generate_model_uuids(num):
    return [str(uuid.uuid4()) for _ in range(num)]


def add_models_to_group_config(group_config):

    if "variation_config" in group_config:
        num_models = len(group_config["variation_config"]["param_values"]) * group_config["replications"]
    else:
        num_models = group_config["replications"]

    model_uuids = generate_model_uuids(num_models)
    model_names = generate_model_names(num_models)

    group_config["model_info"] = []


    for (model_uuid, model_name) in zip(model_uuids, model_names):
        group_config["model_info"].append({
            "model_uuid": model_uuid,
            "model_name": model_name,
            "stage": ADDED_TO_GROUP,
        })



def run_group(req_args):

    logger = logging.getLogger(__name__)


    group_uuid = str(uuid.uuid4())
    group_config = copy.deepcopy(req_args)
    group_config["group_uuid"] = group_uuid
    group_config["model_info"] = {}

    group_name = group_config["group_name"]
    group_description = group_config["group_description"]


    logger.info("Started processing group '{}' (uuid: {}).".format(group_name, group_uuid))

    add_models_to_group_config(group_config)

    # Save the group before training the models. If aborted, the group can now be destroyed.
    group_config_path = os.path.join("usr", "data", "groups", group_uuid + ".json")
    json_io.save_json(group_config_path, group_config)


    if "variation_config" in group_config:
        outer_loop_range = len(group_config["variation_config"]["param_values"])
    else:
        outer_loop_range = 1


    model_index = 0
    
    for _ in range(outer_loop_range):
        for _ in range(group_config["replications"]):

            arch_config, training_config, inference_config = create_model_configs(group_config, 
                                                                                  model_index)

            m = group_config["model_info"][model_index]

            model_interface.create_model(arch_config)
            m["stage"] = FINISHED_ARCH
            json_io.save_json(group_config_path, group_config)


            model_interface.train_model(training_config)
            m["stage"] = FINISHED_TRAINING
            json_io.save_json(group_config_path, group_config)


            model_interface.run_inference(inference_config)
            m["stage"] = FINISHED_INFERENCE
            json_io.save_json(group_config_path, group_config)

            logger.info("Processing of model '{}' is complete.".format(m["model_name"]))
            
            model_index += 1

    logger.info("Finished processing group '{}' (uuid: {}).".format(group_config["group_name"], group_uuid))


def resume_group(req_args):

    logger = logging.getLogger(__name__)

    group_uuid = req_args["group_uuid"]
    group_config_path = os.path.join("usr", "data", "groups", group_uuid + ".json")
    group_config = json_io.load_json(group_config_path)


    if "variation_config" in group_config:
        outer_loop_range = len(group_config["variation_config"]["param_values"])
    else:
        outer_loop_range = 1


    model_index = 0
    for _ in range(outer_loop_range):
        for _ in range(group_config["replications"]):


            arch_config, training_config, inference_config = create_model_configs(group_config, 
                                                                                  model_index)


            m = group_config["model_info"][model_index]
            
            if m["stage"] == ADDED_TO_GROUP:
                model_interface.create_model(arch_config, on_found="replace")
                m["stage"] = FINISHED_ARCH
                json_io.save_json(group_config_path, group_config)

            if m["stage"] == FINISHED_ARCH:
                model_interface.train_model(training_config, on_found="replace")
                m["stage"] = FINISHED_TRAINING
                json_io.save_json(group_config_path, group_config)

            if m["stage"] == FINISHED_TRAINING:
                model_interface.run_inference(inference_config, on_found="replace")
                m["stage"] = FINISHED_INFERENCE
                json_io.save_json(group_config_path, group_config)

            logger.info("Processing of model '{}' is complete.".format(m["model_name"]))

            model_index += 1

    logger.info("Finished processing group '{}' (uuid: {}).".format(group_config["group_name"], group_uuid))






def retrieve_inference_entries_for_model(model_uuid):
    inference_entries = []

    model_dir = os.path.join("usr", "data", "models", model_uuid)
    inference_config_path = os.path.join(model_dir, "inference_config.json")
    if not os.path.exists(inference_config_path):
        return []

    inference_config = json_io.load_json(inference_config_path)
    image_set_confs = inference_config["image_sets"]
    for image_set_conf in image_set_confs:
        for dataset_name in image_set_conf["datasets"]:

            inference_entries.append({
                "farm_name": image_set_conf["farm_name"],
                "field_name": image_set_conf["field_name"],
                "mission_date": image_set_conf["mission_date"],
                "dataset_name": dataset_name,
                "model_uuid": model_uuid
            })

    return inference_entries




def destroy_group(req_args):

    logger = logging.getLogger(__name__)

    group_uuid = req_args["group_uuid"]

    group_config_path = os.path.join("usr", "data", "groups", group_uuid + ".json")
    group_config = json_io.load_json(group_config_path)
    group_name = group_config["group_name"]

    all_inference_entries = []
    logger.info("Starting to destroy group '{}' (uuid: {}).".format(group_name, group_uuid))

    for model_entry in group_config["model_info"]:

        model_uuid = model_entry["model_uuid"]
        model_name = model_entry["model_name"]

        model_dir = os.path.join("usr", "data", "models", model_uuid)

        if os.path.exists(model_dir):
            model_uuid = model_entry["model_uuid"]
            inference_entries = retrieve_inference_entries_for_model(model_uuid)

            all_inference_entries.extend(inference_entries)

            logger.info("Destroying model '{}' (uuid: {})".format(model_name, model_uuid))
            shutil.rmtree(model_dir)


    inference_record_io.remove_entries_from_inference_record(all_inference_entries)
    os.remove(group_config_path)


    logger.info("Finished destroying group '{}' (uuid: {}).".format(group_name, group_uuid))

