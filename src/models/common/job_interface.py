import os
import shutil
import logging
import copy
import randomname
import uuid
import datetime

from io_utils import json_io
from models.common import model_interface #, inference_record_io


ADDED_TO_JOB = "added_to_job"
FINISHED_ARCH = "finished_arch"
FINISHED_TRAINING = "finished_training"
FINISHED_INFERENCE = "finished_inference"



def create_model_configs(job_config, model_index):

    job_uuid = job_config["job_uuid"]
    job_name = job_config["job_name"]

    m = job_config["model_info"][model_index]

    model_uuid = m["model_uuid"]
    model_name = m["model_name"]

    arch_config = copy.deepcopy(job_config["arch_config"])
    training_config =  copy.deepcopy(job_config["training_config"])
    inference_config =  copy.deepcopy(job_config["inference_config"])

    arch_config["model_uuid"] = model_uuid
    arch_config["model_name"] = model_name
    arch_config["job_uuid"] = job_uuid
    arch_config["job_name"] = job_name
    
    training_config["model_uuid"] = model_uuid
    training_config["model_name"] = model_name
    training_config["job_uuid"] = job_uuid
    training_config["job_name"] = job_name

    inference_config["model_uuid"] = model_uuid
    inference_config["model_name"] = model_name
    inference_config["job_uuid"] = job_uuid
    inference_config["job_name"] = job_name


    if "variation_config" in job_config:
        param_index = model_index // job_config["replications"]

        for i in range(len(job_config["variation_config"]["param_values"][param_index])):
            param_config = job_config["variation_config"]["param_configs"][i]
            param_name = job_config["variation_config"]["param_names"][i].split("/")
            param_val = job_config["variation_config"]["param_values"][param_index][i]

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
    return ["model_" + str(i+1) for i in range(num)]
    #return [randomname.get_name() for _ in range(num)]


def generate_model_uuids(num):
    return [str(uuid.uuid4()) for _ in range(num)]


def add_models_to_job_config(job_config):

    if "variation_config" in job_config:
        num_models = len(job_config["variation_config"]["param_values"]) * job_config["replications"]
    else:
        num_models = job_config["replications"]

    model_uuids = generate_model_uuids(num_models)
    model_names = generate_model_names(num_models)

    job_config["model_info"] = []


    for (model_uuid, model_name) in zip(model_uuids, model_names):
        job_config["model_info"].append({
            "model_uuid": model_uuid,
            "model_name": model_name,
            "stage": ADDED_TO_JOB,
        })



def run_job(req_args):

    logger = logging.getLogger(__name__)


    job_uuid = str(uuid.uuid4())
    job_config = copy.deepcopy(req_args)

    job_config["start_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_config["job_uuid"] = job_uuid
    job_config["model_info"] = {}

    job_name = job_config["job_name"]

    logger.info("Started processing job '{}' (uuid: {}).".format(job_name, job_uuid))

    add_models_to_job_config(job_config)

    # Save the job before training the models. If aborted, the job can now be destroyed.
    job_config["status"] = "Running"
    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    json_io.save_json(job_config_path, job_config)


    if "variation_config" in job_config:
        outer_loop_range = len(job_config["variation_config"]["param_values"])
    else:
        outer_loop_range = 1

    exception_handle_method = job_config["on_exception"] if "on_exception" in job_config else "raise"
    model_index = 0
    try:
        for _ in range(outer_loop_range):
            for _ in range(job_config["replications"]):

                arch_config, training_config, inference_config = create_model_configs(job_config, 
                                                                                      model_index)

                m = job_config["model_info"][model_index]

                model_interface.create_model(arch_config)
                m["stage"] = FINISHED_ARCH
                json_io.save_json(job_config_path, job_config)


                model_interface.train_model(training_config)
                m["stage"] = FINISHED_TRAINING
                json_io.save_json(job_config_path, job_config)


                model_interface.run_inference(inference_config)
                m["stage"] = FINISHED_INFERENCE
                json_io.save_json(job_config_path, job_config)

                logger.info("Processing of model '{}' is complete.".format(m["model_name"]))
                
                model_index += 1

    except Exception as e:
        if exception_handle_method == "raise":
            #raise e
            raise_job_exception(job_uuid, e)
        #elif exception_handle_method == "destroy_and_raise":
        #    destroy_job({"job_uuid": job_uuid})
        #    raise e
        else:
            raise e


    job_config["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_config["status"] = "Finished"  
    json_io.save_json(job_config_path, job_config)
    logger.info("Finished processing job '{}' (uuid: {}).".format(job_config["job_name"], job_uuid))


# def resume_job(req_args):

#     logger = logging.getLogger(__name__)

#     job_uuid = req_args["job_uuid"]
#     job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
#     job_config = json_io.load_json(job_config_path)

#     job_name = job_config["job_name"]

#     logger.info("Resuming processing of job '{}' (uuid: {}).".format(job_name, job_uuid))

#     if "variation_config" in job_config:
#         outer_loop_range = len(job_config["variation_config"]["param_values"])
#     else:
#         outer_loop_range = 1


#     exception_handle_method = job_config["on_exception"] if "on_exception" in job_config else "raise"
#     model_index = 0
#     try:
#         for _ in range(outer_loop_range):
#             for _ in range(job_config["replications"]):


#                 arch_config, training_config, inference_config = create_model_configs(job_config, 
#                                                                                       model_index)


#                 m = job_config["model_info"][model_index]
                
#                 if m["stage"] == ADDED_TO_JOB:
#                     model_interface.create_model(arch_config, on_found="replace")
#                     m["stage"] = FINISHED_ARCH
#                     json_io.save_json(job_config_path, job_config)

#                 if m["stage"] == FINISHED_ARCH:
#                     model_interface.train_model(training_config, on_found="replace")
#                     m["stage"] = FINISHED_TRAINING
#                     json_io.save_json(job_config_path, job_config)

#                 if m["stage"] == FINISHED_TRAINING:
#                     model_interface.run_inference(inference_config, on_found="replace")
#                     m["stage"] = FINISHED_INFERENCE
#                     json_io.save_json(job_config_path, job_config)

#                 logger.info("Processing of model '{}' is complete.".format(m["model_name"]))

#                 model_index += 1

#     except Exception as e:
#         if exception_handle_method == "raise":
#             raise e
#         elif exception_handle_method == "destroy_and_raise":
#             destroy_job({"job_uuid": job_uuid}, e)
#             raise e
#         else:
#             raise e

  

#     logger.info("Finished processing job '{}' (uuid: {}).".format(job_config["job_name"], job_uuid))






def retrieve_inference_entries_for_model(model_uuid):
    inference_entries = []

    model_dir = os.path.join("usr", "data", "models", model_uuid)
    inference_config_path = os.path.join(model_dir, "inference_config.json")
    if not os.path.exists(inference_config_path):
        return []

    inference_config = json_io.load_json(inference_config_path)
    image_set_confs = inference_config["image_sets"]
    for image_set_conf in image_set_confs:
        #for dataset_name in image_set_conf["datasets"]:

        inference_entries.append({
            "farm_name": image_set_conf["farm_name"],
            "field_name": image_set_conf["field_name"],
            "mission_date": image_set_conf["mission_date"],
            #"dataset_name": dataset_name,
            "model_uuid": model_uuid
        })

    return inference_entries


def raise_job_exception(job_uuid, exception):

    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    job_config = json_io.load_json(job_config_path)

    job_config["status"] = "Stopped"
    job_config["exception"] = str(exception)
    json_io.save_json(job_config_path, job_config)

    raise exception


def destroy_job(req_args, exception):

    logger = logging.getLogger(__name__)

    job_uuid = req_args["job_uuid"]

    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    job_config = json_io.load_json(job_config_path)
    job_name = job_config["job_name"]

    all_inference_entries = []
    logger.info("Starting to destroy job '{}' (uuid: {}).".format(job_name, job_uuid))

    for model_entry in job_config["model_info"]:

        model_uuid = model_entry["model_uuid"]
        model_name = model_entry["model_name"]

        model_dir = os.path.join("usr", "data", "models", model_uuid)

        if os.path.exists(model_dir):
            model_uuid = model_entry["model_uuid"]
            #inference_entries = retrieve_inference_entries_for_model(model_uuid)

            #all_inference_entries.extend(inference_entries)

            logger.info("Destroying model '{}' (uuid: {})".format(model_name, model_uuid))
            shutil.rmtree(model_dir)


    for image_set_conf in job_config["inference_config"]["image_sets"]:
        farm_name = image_set_conf["farm_name"]
        field_name = image_set_conf["field_name"]
        mission_date = image_set_conf["mission_date"]

        results_dir = os.path.join("usr", "data", "results", farm_name, field_name, mission_date, job_uuid)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)

    #inference_record_io.remove_entries_from_inference_record(all_inference_entries)
    os.remove(job_config_path)




    logger.info("Finished destroying job '{}' (uuid: {}).".format(job_name, job_uuid))

