# import tensorflow as tf
# tf.keras.utils.set_random_seed(123)
# print("random seed set!")

import os
import shutil
import logging
import copy

import datetime
import signal

import math as m
import random
import numpy as np
import pandas as pd
import pandas.io.formats.excel

from io_utils import json_io, w3c_io
from models.common import model_interface, configure_job, model_config #, inference_record_io
import image_set





#ADDED_TO_JOB = "added_to_job"
#FINISHED_ARCH = "finished_arch"
#FINISHED_TRAINING = "finished_training"
#FINISHED_INFERENCE = "finished_inference"
MODEL_ENQUEUED = "Enqueued"
MODEL_TRAINING = "Training"
MODEL_PREDICTING = "Predicting"
MODEL_FINISHED = "Finished"

JOB_RUNNING = "Running"
JOB_STOPPED = "Stopped"
JOB_FINISHED = "Finished"


def create_model_configs(job_config, model_index):

    job_uuid = job_config["job_uuid"]
    job_name = job_config["job_name"]
    target_farm_name = job_config["target_farm_name"]
    target_field_name = job_config["target_field_name"]
    target_mission_date = job_config["target_mission_date"]
    test_reserved_images = job_config["test_reserved_images"]
    training_validation_images = job_config["training_validation_images"]
    #rm_edge_boxes = job_config["rm_edge_boxes"]

    m = job_config["model_info"][model_index]

    model_uuid = m["model_uuid"]
    model_name = m["model_name"]

    arch_config = copy.deepcopy(job_config["arch_config"])
    training_config =  copy.deepcopy(job_config["training_config"])
    inference_config =  copy.deepcopy(job_config["inference_config"])


    for config in [arch_config, training_config, inference_config]:
        config["model_uuid"] = model_uuid
        config["model_name"] = model_name
        config["job_uuid"] = job_uuid
        config["job_name"] = job_name
        config["target_farm_name"] = target_farm_name
        config["target_field_name"] = target_field_name
        config["target_mission_date"] = target_mission_date
        config["test_reserved_images"] = test_reserved_images
        config["training_validation_images"] = training_validation_images
        #config["rm_edge_boxes"] = rm_edge_boxes

    
    training_config["source_construction_params"] = job_config["source_construction_params"]
    inference_config["predict_on_completed_only"] = job_config["predict_on_completed_only"]
    inference_config["supplementary_targets"] = job_config["supplementary_targets"]

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



# def partition_datasets(farm_name, field_name, mission_date):

#     annotations_path = os.path.join("usr", "data", "image_sets", farm_name, field_name, mission_date,
#                                      "annotations", "annotations_w3c.json")
#     annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

#     completed_images = image_set.get_completed_images(annotations)
#     num_completed_images = len(completed_images)

#     training_partition_map = {
#         2: 1,
#         3: 2,
#         4: 3,
#         5: 4,
#         6: 5,
#         7: 5,
#         8: 6,
#         9: 6,
#         10: 7,
#         11: 7,
#         12: 8
#     }

#     if num_completed_images < 2:
#         raise RuntimeError("Insufficient number of fully annotated images. At least 2 fully annotated images are required.") 
#     elif num_completed_images in training_partition_map:
#         num_training = training_partition_map[num_completed_images]
#     else:
#         training_percent = 60
#         num_training = m.floor(num_completed_images * (training_percent / 100))

#     shuffled = random.sample(completed_images, num_completed_images)
#     training_images = shuffled[:num_training]
#     validation_images = shuffled[num_training:] #(num_training+num_validation)]

#     return training_images, validation_images






def run_test_job():
    import uuid
    # import numpy as np
    # import tensorflow as tf
    # import random as python_random
    # np.random.seed(123)
    # python_random.seed(123)
    # tf.random.set_seed(1234)


    
    logging.basicConfig(level=logging.INFO)

    subset_type = "graph_subset" #, "even_subset"]
    method_params_lst = [
        # {
        #     "match_method": "bipartite_b_matching",
        #     "extraction_type": "excess_green_box_combo", #"excess_green", #"surrounding_boxes", #"excess_green",
        #     "patch_size": "image_set_dependent", # alternatively, set to a fixed value
        #     "source_pool_size": 12000,
        #     "target_pool_size": 3000
        # },
        # {
        #     #"match_method": "bipartite_b_matching",
        #     "extraction_type": "excess_green_box_combo",
        #     "patch_size": "image_set_dependent",
        #     #"source_pool_size": 12000,
        #     #"target_pool_size": 3000
        # },
        {
            "match_method": "bipartite_b_matching", #"reciprocal_match", #"diverse_bipartite_b_matching",
            "extraction_type": "excess_green_box_combo", #"excess_green_box_combo", #"excess_green_box_combo", #"surrounding_boxes", #"excess_green", #"surrounding_boxes", #"excess_green", #"surounding_boxes", #"excess_green",
            "patch_size": "image_set_dependent",
            "source_pool_size": 25000, #18000, #12000, #12000,
            "target_pool_size": 500, #2000, #3000 #3000
            "exclude_target_from_source": True 
        },
        #{
        #    "extraction_type": "excess_green", #"surrounding_boxes", #"excess_green", #"surounding_boxes", #"excess_green",
        #    "patch_size": "image_set_dependent",
        #},
        # {
        #     "match_method": "bipartite_b_matching",
        #     "extraction_type": "surrounding_boxes",
        #     "patch_size": "image_set_dependent",
        # },

    ]
    #for subset_type in subset_types:
    for method_params in method_params_lst:

        job_uuid = str(uuid.uuid4())
        # job_config = {
        #     "job_uuid": job_uuid,
        #     "replications": 1,
        #     "job_name": "test_name_" + job_uuid,
        #     "source_construction_params": {
        #         "method_name": "graph_subset", #"direct", #"graph_subset", #"graph_subset", #"direct", #subset_type,
        #         "method_params": method_params,
        #         "size": 5000 #4000 #2000, #2000
        #     },
        #     "target_farm_name": "BlaineLake", #"row_spacing", #"UNI", #"row_spacing", #"UNI", #"BlaineLake", #"row_spacing", #"UNI", #"BlaineLake", #"row_spacing",  #"Biggar", , #"BlaineLake", #"row_spacing", #"BlaineLake", #"row_spacing",
        #     "target_field_name": "HornerWest", #"nasser", #"LowN1", #"nasser", #"LowN1", #"HornerWest", #"River", #"HornerWest", #"brown", #"LowN1", #"nasser", #"Dennis3",  #"River", #"nasser", #"HornerWest", #"nasser", 
        #     "target_mission_date": "2021-06-09" #"2020-06-08" #"2021-06-07" #"2020-06-08" #"2021-06-07" #"2021-06-09" #"2021-06-01" #"2021-06-07"  #"2020-06-08"  #"2021-06-04"  #"2021-06-09" #2020-06-08",  #"2021-06-09" #"2020-06-08"
        # }

        job_config = {
            "job_uuid": job_uuid,
            "replications": 1,
            "job_name": "test_name_" + job_uuid,
            "source_construction_params": {
                "method_name": "direct", #"even_subset", #"graph_subset", #"even_subset", #"graph_subset", #"even_subset", #"direct", #"graph_subset", #"graph_subset", #"direct", #subset_type,
                "method_params": method_params,
                "size": 500 #4000 #2000, #2000
            },
            "target_farm_name": "UNI", #"row_spacing", #"BlaineLake", #"row_spacing", #"UNI", #"row_spacing", #"UNI", #"BlaineLake", #"row_spacing", #"UNI", #"BlaineLake", #"row_spacing",  #"Biggar", , #"BlaineLake", #"row_spacing", #"BlaineLake", #"row_spacing",
            "target_field_name": "Sutherland", #"brown", #"HornerWest", #"nasser", #"LowN1", #"nasser", #"LowN1", #"HornerWest", #"River", #"HornerWest", #"brown", #"LowN1", #"nasser", #"Dennis3",  #"River", #"nasser", #"HornerWest", #"nasser", 
            "target_mission_date": "2021-06-05", #"2021-06-01", #"2021-06-09", #"2020-06-08" #"2021-06-07" #"2020-06-08" #"2021-06-07" #"2021-06-09" #"2021-06-01" #"2021-06-07"  #"2020-06-08"  #"2021-06-04"  #"2021-06-09" #2020-06-08",  #"2021-06-09" #"2020-06-08"
            "predict_on_completed_only": True,
            "supplementary_targets": [
                {
                    "target_farm_name": "row_spacing",
                    "target_field_name": "nasser",
                    "target_mission_date": "2020-06-08"
                },
                {
                    "target_farm_name": "UNI",
                    "target_field_name": "LowN1",
                    "target_mission_date": "2021-06-07"                    
                },
                {
                    "target_farm_name": "BlaineLake",
                    "target_field_name": "River",
                    "target_mission_date": "2021-06-09"                    
                }
            ]
        }

        json_io.save_json(os.path.join("usr", "data", "jobs", job_uuid + ".json"), job_config)
        run_job(job_uuid)



def run_job(job_uuid):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    #job_uuid = str(uuid.uuid4())

    def signal_handler(sig, frame):
        raise_job_exception(job_uuid, Exception("Interrupted by user"))
    signal.signal(signal.SIGINT, signal_handler)

    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    job_config = json_io.load_json(job_config_path)
    job_name = job_config["job_name"]


    job_config["start_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Started processing job '{}' (uuid: {}).".format(job_name, job_uuid))

    try:

        target_farm_name = job_config["target_farm_name"]
        target_field_name = job_config["target_field_name"]
        target_mission_date = job_config["target_mission_date"]
        test_reserved_images = job_config["test_reserved_images"]

        annotations_path = os.path.join("usr", "data", "image_sets",
                                        target_farm_name, target_field_name, target_mission_date,
                                        "annotations", "annotations_w3c.json")
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
        completed_image_names = w3c_io.get_completed_images(annotations)
        job_config["training_validation_images"] = [image_name for image_name in completed_image_names if image_name not in test_reserved_images]



        # Save the job before training the models. If aborted, the job can now be destroyed.
        job_config["status"] = JOB_RUNNING
        json_io.save_json(job_config_path, job_config)

         #, farm_name, field_name, mission_date, config_type="full_source")
        #fill_job_config(job_config, farm_name, field_name, mission_date)
        configure_job.fill_job_config(job_config)
        json_io.save_json(job_config_path, job_config)


        if "variation_config" in job_config:
            outer_loop_range = len(job_config["variation_config"]["param_values"])
        else:
            outer_loop_range = 1

        exception_handle_method = job_config["on_exception"] if "on_exception" in job_config else "raise"
        model_index = 0

        for _ in range(outer_loop_range):
            for _ in range(job_config["replications"]):

                arch_config, training_config, inference_config = create_model_configs(job_config, 
                                                                                      model_index)

                m = job_config["model_info"][model_index]

                model_interface.create_model(arch_config)

                m["stage"] = MODEL_TRAINING
                json_io.save_json(job_config_path, job_config)
                model_interface.train_model(training_config)

                m["stage"] = MODEL_PREDICTING
                json_io.save_json(job_config_path, job_config)
                model_interface.run_inference(inference_config)

                m["stage"] = MODEL_FINISHED
                json_io.save_json(job_config_path, job_config)

                logger.info("Processing of model '{}' is complete.".format(m["model_name"]))
                
                model_index += 1

    except Exception as e:
        #if exception_handle_method == "raise":
            #raise e
        raise_job_exception(job_uuid, e)
        #elif exception_handle_method == "destroy_and_raise":
        #    destroy_job({"job_uuid": job_uuid})
        #    raise e
        #else:
        #    raise e

    
    build_job_excel(job_config)


    job_config["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_config["status"] = JOB_FINISHED
    json_io.save_json(job_config_path, job_config)
    logger.info("Finished processing job '{}' (uuid: {}).".format(job_config["job_name"], job_uuid))


def build_job_excel(job_config):
    target_farm_name = job_config["target_farm_name"]
    target_field_name = job_config["target_field_name"]
    target_mission_date = job_config["target_mission_date"]
    job_uuid = job_config["job_uuid"]
    job_results_dir = os.path.join("usr", "data", "results", target_farm_name, target_field_name, target_mission_date, job_uuid)
    out_path = os.path.join(job_results_dir, "results.xlsx")
    model_info_lst = job_config["model_info"]



    model_uuid = model_info_lst[0]["model_uuid"]
    model_name = model_info_lst[0]["model_name"]
    model_dir = os.path.join(job_results_dir, model_uuid)
    excel_path = os.path.join(model_dir, "results.xlsx")
    job_df = pd.read_excel(excel_path)
    job_df.rename(columns={"model_plant_count": model_name + "_plant_count"}, inplace=True)
    #job_df.rename(columns={"model_plant_count_at_optimal_score": model_name + "_plant_count_at_optimal_score"}, inplace=True)

    for model_info in model_info_lst[1:]:
        model_uuid = model_info["model_uuid"]
        model_dir = os.path.join(job_results_dir, model_uuid)
        excel_path = os.path.join(model_dir, "results.xlsx")
        df = pd.read_excel(excel_path)

        job_df[model_info["model_name"] + "_plant_count"] = df["model_plant_count"]
        #job_df[model_info["model_name"] + "_plant_count_at_optimal_score"] = df["model_plant_count_at_optimal_score"]

    #print("job_df", job_df)
    pandas.io.formats.excel.ExcelFormatter.header_style = None
    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")    
    job_df.to_excel(writer, index=False, sheet_name="Sheet1", na_rep='NA')  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(job_df):  # loop through all columns
        series = job_df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save()




# def resume_job(job_uuid):


#     logger = logging.getLogger(__name__)

#     #job_uuid = str(uuid.uuid4())
    


#     def signal_handler(sig, frame):
#         raise_job_exception(job_uuid, Exception("Interrupted by user"))
#     signal.signal(signal.SIGINT, signal_handler)

#     job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
#     job_config = json_io.load_json(job_config_path)

#     job_name = job_config["job_name"]

#     job_config["start_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     logger.info("Resumed job '{}' (uuid: {}).".format(job_name, job_uuid))

#     try:
#         # Save the job before training the models. If aborted, the job can now be destroyed.
#         job_config["status"] = JOB_RUNNING
#         if "exception" in job_config:
#             del job_config["exception"]
#         json_io.save_json(job_config_path, job_config)

#          #, farm_name, field_name, mission_date, config_type="full_source")
#         #fill_job_config(job_config, farm_name, field_name, mission_date)
#         #configure_job.fill_job_config(job_config)
#         #json_io.save_json(job_config_path, job_config)


#         if "variation_config" in job_config:
#             outer_loop_range = len(job_config["variation_config"]["param_values"])
#         else:
#             outer_loop_range = 1


#         model_index = 0

#         for _ in range(outer_loop_range):
#             for _ in range(job_config["replications"]):

#                 arch_config, training_config, inference_config = create_model_configs(job_config, 
#                                                                                       model_index)

#                 m = job_config["model_info"][model_index]
#                 #print(m)
#                 #exit()

#                 if m["stage"] == MODEL_ENQUEUED:
#                     model_interface.create_model(arch_config)
#                     m["stage"] = MODEL_TRAINING

#                 if m["stage"] == MODEL_TRAINING:
#                     json_io.save_json(job_config_path, job_config)
#                     model_interface.train_model(training_config)
#                     m["stage"] = MODEL_PREDICTING

#                 if m["stage"] == MODEL_PREDICTING:
#                     json_io.save_json(job_config_path, job_config)
#                     model_interface.run_inference(inference_config)
#                     m["stage"] = MODEL_FINISHED

#                 json_io.save_json(job_config_path, job_config)

#                 logger.info("Processing of model '{}' is complete.".format(m["model_name"]))
                
#                 model_index += 1

#     except Exception as e:
#         #if exception_handle_method == "raise":
#             #raise e
#         raise_job_exception(job_uuid, e)
#         #elif exception_handle_method == "destroy_and_raise":
#         #    destroy_job({"job_uuid": job_uuid})
#         #    raise e
#         #else:
#         #    raise e


#     job_config["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     job_config["status"] = JOB_FINISHED  
#     json_io.save_json(job_config_path, job_config)
#     logger.info("Finished processing job '{}' (uuid: {}).".format(job_config["job_name"], job_uuid))


def resume_job(job_uuid):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    

    #job_uuid = str(uuid.uuid4())

    def signal_handler(sig, frame):
        raise_job_exception(job_uuid, Exception("Interrupted by user"))
    signal.signal(signal.SIGINT, signal_handler)

    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    job_config = json_io.load_json(job_config_path)
    job_name = job_config["job_name"]


    #job_config["start_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Resuming job '{}' (uuid: {}).".format(job_name, job_uuid))

    try:

        # target_farm_name = job_config["target_farm_name"]
        # target_field_name = job_config["target_field_name"]
        # target_mission_date = job_config["target_mission_date"]
        # test_reserved_images = job_config["test_reserved_images"]

        # annotations_path = os.path.join("usr", "data", "image_sets",
        #                                 target_farm_name, target_field_name, target_mission_date,
        #                                 "annotations", "annotations_w3c.json")
        # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
        # completed_image_names = w3c_io.get_completed_images(annotations)
        # job_config["training_validation_images"] = [image_name for image_name in completed_image_names if image_name not in test_reserved_images]



        # # Save the job before training the models. If aborted, the job can now be destroyed.
        # job_config["status"] = JOB_RUNNING
        # json_io.save_json(job_config_path, job_config)

        #  #, farm_name, field_name, mission_date, config_type="full_source")
        # #fill_job_config(job_config, farm_name, field_name, mission_date)
        # configure_job.fill_job_config(job_config)
        # json_io.save_json(job_config_path, job_config)


        if "variation_config" in job_config:
            outer_loop_range = len(job_config["variation_config"]["param_values"])
        else:
            outer_loop_range = 1

        exception_handle_method = job_config["on_exception"] if "on_exception" in job_config else "raise"
        model_index = 0

        for _ in range(outer_loop_range):
            for _ in range(job_config["replications"]):

                arch_config, training_config, inference_config = create_model_configs(job_config, 
                                                                                      model_index)

                m = job_config["model_info"][model_index]

                
                if m["stage"] == MODEL_ENQUEUED:
                    model_interface.create_model(arch_config)
                    m["stage"] = MODEL_TRAINING
                    json_io.save_json(job_config_path, job_config)

                if m["stage"] == MODEL_TRAINING:
                    model_interface.train_model(training_config)
                    m["stage"] = MODEL_PREDICTING
                    json_io.save_json(job_config_path, job_config)

                if m["stage"] == MODEL_PREDICTING:
                    model_interface.run_inference(inference_config)
                    m["stage"] = MODEL_FINISHED
                    json_io.save_json(job_config_path, job_config)


                logger.info("Processing of model '{}' is complete.".format(m["model_name"]))
                
                model_index += 1

    except Exception as e:
        #if exception_handle_method == "raise":
            #raise e
        raise_job_exception(job_uuid, e)
        #elif exception_handle_method == "destroy_and_raise":
        #    destroy_job({"job_uuid": job_uuid})
        #    raise e
        #else:
        #    raise e

    
    build_job_excel(job_config)


    job_config["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_config["status"] = JOB_FINISHED
    json_io.save_json(job_config_path, job_config)
    logger.info("Finished processing job '{}' (uuid: {}).".format(job_config["job_name"], job_uuid))






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

    job_config["status"] = JOB_STOPPED
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

