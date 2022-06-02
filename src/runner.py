import tensorflow as tf
tf.keras.utils.set_random_seed(123)
print("random seed set!")
import logging
logging.basicConfig(level=logging.INFO)



import os
import shutil
import glob
import uuid
import numpy as np
import matplotlib.pyplot as plt

from models.common import job_interface
from io_utils import json_io, w3c_io



def reduce_db_size(pct, target_farm_name, target_field_name, target_mission_date):

    image_set_root = os.path.join("usr", "data", "image_sets")

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if not ((farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date):
                    
                    annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                                "annotations", "annotations_w3c.json")
                    save_annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                                "annotations", "SAVE_annotations_w3c.json")                           
                    shutil.copy(annotations_path, save_annotations_path)

                    w3c_annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
                    annotations = json_io.load_json(annotations_path)

                    completed_images = w3c_io.get_completed_images(w3c_annotations)

                    keep_images = np.random.choice(completed_images, int(len(completed_images) * pct))

                    for image_name in completed_images:
                        if image_name not in keep_images:
                            annotations[image_name] = {
                                "status": "unannotated",
                                "annotations": []
                            }

                    json_io.save_json(annotations_path, annotations)

def restore_db(target_farm_name, target_field_name, target_mission_date):

    image_set_root = os.path.join("usr", "data", "image_sets")

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if not ((farm_name == target_farm_name and field_name == target_field_name) and mission_date == target_mission_date):
                    
                    annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                                "annotations", "annotations_w3c.json")
                    save_annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                                "annotations", "SAVE_annotations_w3c.json")

                    shutil.move(save_annotations_path, annotations_path)
                    os.remove(save_annotations_path)


def run_db_size_variation_test():


    dataset_sizes = [250, 500, 1000, 2000, 4000, 8000]
    methods = ["even_subset", "graph_subset"]

    method_params = {
            "match_method": "bipartite_b_matching",
            "extraction_type": "excess_green_box_combo",
            "patch_size": "image_set_dependent",
            "exclude_target_from_source": True 
    }

    target_datasets = [
        {
            "target_farm_name": "Biggar",
            "target_field_name": "Dennis3",
            "target_mission_date": "2021-06-04"
        }
    ]

    epoch_patience = {
        256: 30,
        1024: 30,
        4096: 30,
        8192: 30,
        16384: 30
    }


    db_pcts = [100, 50]


    run_record = {
        "job_uuids": [],
        "method_params": method_params,
        "methods": methods,
        "dataset_sizes": dataset_sizes,
        "target_datasets": target_datasets,
        "db_pcts": db_pcts,
        "explanation": "db size variation"
    }


    for dataset in target_datasets:
        for db_pct in db_pcts:

            reduce_db_size(db_pct, dataset["target_farm_name"], dataset["target_field_name"], dataset["target_mission_date"])

            for dataset_size in dataset_sizes:
                for i, method in enumerate(methods):
                    job_uuid = str(uuid.uuid4())

                    job_config = {
                        "job_uuid": job_uuid,
                        "replications": 1,
                        "job_name": "test_name_" + job_uuid,
                        "source_construction_params": {
                            "method_name": method,
                            "method_params": method_params,
                            "size": dataset_size
                        },
                        "target_farm_name": dataset["target_farm_name"],
                        "target_field_name": dataset["target_field_name"],
                        "target_mission_date": dataset["target_mission_date"],
                        "predict_on_completed_only": True,
                        "supplementary_targets": [],
                        "tol_test": 30, #epoch_patience[dataset_size]
                    }
                    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
                    json_io.save_json(job_config_path, job_config)
                    job_interface.run_job(job_uuid)

                    run_record["job_uuids"].append(job_uuid)

            restore_db(dataset["target_farm_name"], dataset["target_field_name"], dataset["target_mission_date"])




    run_uuid = str(uuid.uuid4())
    run_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    json_io.save_json(run_path, run_record)
    report(run_uuid)


def plot_tSNE():

    method_params = {
            "match_method": "bipartite_b_matching",
            "extraction_type": "excess_green_box_combo",
            "patch_size": "image_set_dependent",
            #"source_pool_size": 25000, #18000, #12000, #12000,
            #"target_pool_size": 500, #2000, #3000 #3000
            "exclude_target_from_source": True 
    }

    job_uuid = str(uuid.uuid4())
    job_config = {
        "job_uuid": job_uuid,
        "replications": 1,
        "job_name": "test_name_" + job_uuid,
        "source_construction_params": {
            "method_name": "create_tsne_plot",
            "method_params": method_params,
            #"size": dataset_size
        },
        "target_farm_name": "BlaineLake", #dataset["target_farm_name"],
        "target_field_name": "HornerWest", #dataset["target_field_name"],
        "target_mission_date": "2021-06-09", #dataset["target_mission_date"],
        "predict_on_completed_only": True,
        "supplementary_targets": [],
        "tol_test": 30, #epoch_patience[dataset_size]
    }

    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    json_io.save_json(job_config_path, job_config)
    job_interface.run_job(job_uuid)



def calc_sim_mAP_tuples():
    sim_map_tuples = []
    kept_runs_root = os.path.join("usr", "data", "runs", "kept_runs")
    for run_path in glob.glob(os.path.join(kept_runs_root, "*")):

        run_config = json_io.load_json(run_path)
        target_farm_name =  run_config["target_datasets"][0]["target_farm_name"]
        target_field_name =  run_config["target_datasets"][0]["target_field_name"]
        target_mission_date =  run_config["target_datasets"][0]["target_mission_date"]

        for job_uuid in run_config["job_uuids"]:
            job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
            job_config = json_io.load_json(job_config_path)

            model_uuid = job_config["model_info"][0]["model_uuid"]

            metrics_path = os.path.join("usr", "data", "results", target_farm_name,
                                    target_field_name, target_mission_date, job_uuid, model_uuid,
                                    "metrics.json")
            metrics = json_io.load_json(metrics_path)

            graph_stats_path = os.path.join("usr", "data", "models", model_uuid, "graph_stats.json")

            if os.path.exists(graph_stats_path):
                graph_stats = json_io.load_json(graph_stats_path)
                distances_size = graph_stats["records"][0]["selected_distances.size"]

                mean_dist = graph_stats["records"][0]["mean_distance"]
                ms_coco_mAP = metrics["point"]["Image MS COCO mAP"]["---"]

                sim_map_tuples.append((target_farm_name + "::" + target_field_name + "::" + target_mission_date, 
                                        distances_size, mean_dist, ms_coco_mAP))

                #print("({}, {})".format(mean_dist, ms_coco_mAP))

    s = sorted(sim_map_tuples)
    for t in s:
        print(t[0] + ", " + str(t[1]) + ", " + str(t[2]) + ", " + str(t[3]))




def assign_relative_error():
    image_set_stats = {}
    #metric = "MS COCO mAP"
    scores = []
    image_sets = []
    
    kept_runs_root = os.path.join("usr", "data", "runs", "kept_runs")
    for run_path in glob.glob(os.path.join(kept_runs_root, "*")):

        run_config = json_io.load_json(run_path)
        target_farm_name =  run_config["target_datasets"][0]["target_farm_name"]
        target_field_name =  run_config["target_datasets"][0]["target_field_name"]
        target_mission_date =  run_config["target_datasets"][0]["target_mission_date"]
        results_path = os.path.join("usr", "data", "runs", "display", 
                     target_farm_name, target_field_name, target_mission_date, "results.json")
        results = json_io.load_json(results_path)
        
        annotations_path = os.path.join("usr", "data", "image_sets",
                            target_farm_name, target_field_name, target_mission_date,
                            "annotations", "annotations_w3c.json")
        annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
        completed_images = w3c_io.get_completed_images(annotations)
        rel_error_results = []
        rel_opt_error_results = []
        for job_uuid in run_config["job_uuids"]:
            job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
            job_config = json_io.load_json(job_config_path)

            model_uuid = job_config["model_info"][0]["model_uuid"]
            predictions_path = os.path.join("usr", "data", "results", target_farm_name,
                                    target_field_name, target_mission_date, job_uuid, model_uuid,
                                    "predictions.json")
            predictions = json_io.load_json(predictions_path)


            metrics_path = os.path.join("usr", "data", "results", target_farm_name,
                                    target_field_name, target_mission_date, job_uuid, model_uuid,
                                    "metrics.json")
            metrics = json_io.load_json(metrics_path)

            relative_errors = []
            relative_opt_errors = []
            for image_name in completed_images:
                pred_plant_count = predictions["image_predictions"][image_name]["pred_class_counts"]["plant"]
                actual_plant_count = annotations[image_name]["boxes"].shape[0]
                relative_error = ((abs(pred_plant_count - actual_plant_count)) / actual_plant_count) * 100
                relative_errors.append(relative_error)

                thresh_val = metrics["point"]["optimal_score_threshold"]["threshold_value"]
                pred_opt_plant_count = (np.where(np.array(predictions["image_predictions"][image_name]["pred_scores"]) >= thresh_val)[0]).size
                relative_opt_error = ((abs(pred_opt_plant_count - actual_plant_count)) / actual_plant_count) * 100
                relative_opt_errors.append(relative_opt_error)

            rel_error_results.append(np.mean(relative_errors))
            rel_opt_error_results.append(np.mean(relative_opt_errors))

        i = 0
        res_d = {
            "graph_subset": [], 
            "even_subset": [], 
            "direct": []
        }
        res_d_opt = {
            "graph_subset": [], 
            "even_subset": [], 
            "direct": []
        }        
        for dataset_size in run_config["dataset_sizes"]:
            for method in run_config["methods"]:
                
                res_d[method].append(rel_error_results[i])
                res_d_opt[method].append(rel_opt_error_results[i])
                i += 1
        inds = np.argsort(run_config["dataset_sizes"])

        for method in res_d.keys():
            results["results"][method]["Image Mean Percent Error in Count"] = (np.array(res_d[method])[inds]).tolist()
            results["results"][method]["Image Mean Percent Error in Count at Optimal Score Thresh."] = (np.array(res_d_opt[method])[inds]).tolist()


        print("results for : {}-{}-{}: {}".format(target_farm_name, 
        target_field_name, target_mission_date, results))
        print()
        print("--------")
        print()

        json_io.save_json(results_path, results)

    #     farm_name = os.path.basename(farm_path)
    #     if farm_name not in image_set_stats:
    #         image_set_stats[farm_name] = {}
    #     for field_path in glob.glob(os.path.join(farm_path, "*")):
    #         field_name = os.path.basename(field_path)
    #         if field_name not in image_set_stats[farm_name]:
    #             image_set_stats[farm_name][field_name] = {}
    #         for mission_path in glob.glob(os.path.join(field_path, "*")):
    #             mission_date = os.path.basename(mission_path)

    #             if len(mission_date) == 10:
    #                 results_path = os.path.join(mission_path, "results.json")
    #                 results = json_io.load_json(results_path)
    #                 gs_res = np.array(results["results"]["graph_subset"][metric])
    #                 es_res = np.array(results["results"]["even_subset"][metric])
    #                 diff_sum = np.sum(gs_res - es_res)

    #                 image_sets.append(farm_name + "::" + field_name + "::" + mission_date)
    #                 scores.append(diff_sum)

    # scores = np.array(scores)
    # image_sets = np.array(image_sets)

    # inds = np.argsort(-scores)
    # sorted_scores = scores[inds]
    # sorted_image_sets = image_sets[inds]

    # print("sorted_order")
    # for i in range(len(sorted_image_sets)):
    #     print("ImageSet: {} | Score: {}".format(sorted_image_sets[i], sorted_scores[i]))



def display_results_order():
    image_set_stats = {}
    metric = "Image Mean Percent Error in Count at Optimal Score Thresh." #"MS COCO mAP"

    scores = []
    image_sets = []
    display_root = os.path.join("usr", "data", "runs", "display")
    for farm_path in glob.glob(os.path.join(display_root, "*")):
        farm_name = os.path.basename(farm_path)
        if farm_name not in image_set_stats:
            image_set_stats[farm_name] = {}
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            if field_name not in image_set_stats[farm_name]:
                image_set_stats[farm_name][field_name] = {}
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                if len(mission_date) == 10 or mission_date == "2021-06-01-high-res":
                    results_path = os.path.join(mission_path, "results.json")
                    results = json_io.load_json(results_path)
                    gs_res = np.array(results["results"]["graph_subset"][metric])
                    es_res = np.array(results["results"]["even_subset"][metric])
                    dir_res = np.array(results["results"]["direct"][metric])
                    #diff_sum = np.sum((gs_res - es_res) / (dir_res - es_res))
                    diff_sum = np.sum(((dir_res - gs_res))) # / (es_res - dir_res)**2)
                    #diff_sum = np.sum(gs_res / ((dir_res - es_res) / 2)) / 6
                    #diff_sum = np.sum((es_res - gs_res) / (es_res - dir_res))

                    image_sets.append(farm_name + "::" + field_name + "::" + mission_date)
                    scores.append(diff_sum)

    scores = np.array(scores)
    print("scores", scores)
    # max_score = np.max(scores)
    # print("max score", max_score)
    # scores /= max_score
    # print("scores", scores)
    # scores = 1 - scores
    # print("scores", scores)
    image_sets = np.array(image_sets)

    inds = np.argsort(-scores)
    sorted_scores = scores[inds]
    sorted_image_sets = image_sets[inds]

    print("sorted_order")
    for i in range(len(sorted_image_sets)):
        print("ImageSet: {} | Score: {}".format(sorted_image_sets[i], sorted_scores[i]))

                    

def gather_image_set_stats():
    image_set_stats = {}
    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        if farm_name not in image_set_stats:
            image_set_stats[farm_name] = {}
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            if field_name not in image_set_stats[farm_name]:
                image_set_stats[farm_name][field_name] = {}
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)
                # if mission_date not in record[farm_name][field_name]:
                #     record[farm_name][field_name][mission_date] = {}

                annotations_path = os.path.join(image_set_root, farm_name, field_name, mission_date,
                                             "annotations", "annotations_w3c.json")
                annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})
                num_completed_images = len(w3c_io.get_completed_images(annotations))
                num_annotations = 0
                for image_name in annotations.keys():
                    num_annotations += annotations[image_name]["boxes"].shape[0]
                image_set_stats[farm_name][field_name][mission_date] = {
                    "num_annotated_images": num_completed_images,
                    "num_annotations": num_annotations
                }

    display_dir = os.path.join("usr", "data", "runs", "display")
    image_set_stats_path = os.path.join(display_dir, "image_set_stats.json")
    json_io.save_json(image_set_stats_path, image_set_stats)
                


def run_kaylie_test():
    dataset_sizes = [20000] #3000] #[15000] #250] #10000]
    methods = ["direct"]

    # method_params = [
    # {
    #         "match_method": "bipartite_b_matching",
    #         "extraction_type": "excess_green_box_combo_two_phase",
    #         "patch_size": "image_set_dependent",
    #         #"source_pool_size": 25000, #18000, #12000, #12000,
    #         #"target_pool_size": 500, #2000, #3000 #3000
    #         "exclude_target_from_source": True 
    # },{
    method_params = {
            "match_method": "bipartite_b_matching",
            "extraction_type": "excess_green_box_combo",
            "patch_size": "image_set_dependent",
            #"source_pool_size": 25000, #18000, #12000, #12000,
            #"target_pool_size": 500, #2000, #3000 #3000
            "exclude_target_from_source": True 
    }

    target_datasets = [
        {
            "target_farm_name": "row_spacing",
            "target_field_name": "brown",
            "target_mission_date": "2021-06-01-high-res"
        }
    ]

    epoch_patience = {
        256: 30,
        1024: 30,
        4096: 30,
        8192: 30,
        16384: 30
    }



    run_record = {
        "job_uuids": [],
        "method_params": method_params,
        "methods": methods,
        "dataset_sizes": dataset_sizes,
        "target_datasets": target_datasets,
        "explanation": "compare methods"
    }

    for dataset in target_datasets:
        for dataset_size in dataset_sizes:
            for i, method in enumerate(methods):
                job_uuid = str(uuid.uuid4())

                job_config = {
                    "job_uuid": job_uuid,
                    "replications": 3,
                    "job_name": "test_name_" + job_uuid,
                    "source_construction_params": {
                        "method_name": method,
                        "method_params": method_params,
                        "size": dataset_size
                    },
                    "target_farm_name": dataset["target_farm_name"],
                    "target_field_name": dataset["target_field_name"],
                    "target_mission_date": dataset["target_mission_date"],
                    "predict_on_completed_only": False, #True,
                    "supplementary_targets": [],
                    "tol_test": 30, #30, #epoch_patience[dataset_size]
                    "test_reserved_images": ["204", "311", "805", "810", "817", "819", "821", "824"]
                }
                job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
                json_io.save_json(job_config_path, job_config)
                job_interface.run_job(job_uuid)

                run_record["job_uuids"].append(job_uuid)

    run_uuid = str(uuid.uuid4())
    run_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    json_io.save_json(run_path, run_record)


def run_tests():
    dataset_sizes = [8000, 4000, 2000, 1000, 500, 250] #[250, 500, 1000, 2000, 4000, 8000] #[256, 1024, 4096, 8192] #, 16384] #, 512, 1024, 2048, 4096]
    methods = ["graph_subset", "even_subset", "direct"] #, "graph_subset"] #, "even_subset", "graph_subset"]

    # method_params = [
    # {
    #         "match_method": "bipartite_b_matching",
    #         "extraction_type": "excess_green_box_combo_two_phase",
    #         "patch_size": "image_set_dependent",
    #         #"source_pool_size": 25000, #18000, #12000, #12000,
    #         #"target_pool_size": 500, #2000, #3000 #3000
    #         "exclude_target_from_source": True 
    # },{
    method_params = {
            "match_method": "bipartite_b_matching",
            "extraction_type": "excess_green_box_combo",
            "patch_size": "image_set_dependent",
            #"source_pool_size": 25000, #18000, #12000, #12000,
            #"target_pool_size": 500, #2000, #3000 #3000
            "exclude_target_from_source": True 
    }

    target_datasets = [
        {
            "target_farm_name": "UNI",
            "target_field_name": "Brown",
            "target_mission_date": "2021-06-05"
        }
    ]

    epoch_patience = {
        256: 30,
        1024: 30,
        4096: 30,
        8192: 30,
        16384: 30
    }



    run_record = {
        "job_uuids": [],
        "method_params": method_params,
        "methods": methods,
        "dataset_sizes": dataset_sizes,
        "target_datasets": target_datasets,
        "explanation": "compare methods"
    }

    for dataset in target_datasets:
        for dataset_size in dataset_sizes:
            for i, method in enumerate(methods):
                job_uuid = str(uuid.uuid4())

                if method == "graph_subset" or method == "even_subset":
                    annotations_path = os.path.join("usr", "data", "image_sets",
                        dataset["target_farm_name"], dataset["target_field_name"], dataset["target_mission_date"],
                        "annotations", "annotations_w3c.json")
                    annotations = w3c_io.load_json(annotations_path)
                    test_reserved_images = w3c_io.get_completed_images(annotations)
                else:
                    test_reserved_images = []

                job_config = {
                    "job_uuid": job_uuid,
                    "replications": 1,
                    "job_name": "test_name_" + job_uuid,
                    "source_construction_params": {
                        "method_name": method,
                        "method_params": method_params,
                        "size": dataset_size
                    },
                    "target_farm_name": dataset["target_farm_name"],
                    "target_field_name": dataset["target_field_name"],
                    "target_mission_date": dataset["target_mission_date"],
                    "predict_on_completed_only": True,
                    "supplementary_targets": [],
                    "tol_test": 30, #epoch_patience[dataset_size]
                    "test_reserved_images": test_reserved_images
                }
                job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
                json_io.save_json(job_config_path, job_config)
                job_interface.run_job(job_uuid)

                run_record["job_uuids"].append(job_uuid)

    run_uuid = str(uuid.uuid4())
    run_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    json_io.save_json(run_path, run_record)
    #report(run_uuid)
    prepare_report_for_display(run_uuid)


def prepare_report_for_display(run_uuid):
    run_record_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    run_record = json_io.load_json(run_record_path)
    
    dataset_sizes = sorted(run_record["dataset_sizes"])

    results = {
        "dataset_sizes": dataset_sizes,
        "results": {}
    }
    i = 0
    dataset = run_record["target_datasets"][0]
    target_farm_name = dataset["target_farm_name"]
    target_field_name = dataset["target_field_name"]
    target_mission_date = dataset["target_mission_date"]

    inds = np.argsort(run_record["dataset_sizes"])
    display_metrics = [
        "MS COCO mAP",
        "PASCAL VOC mAP",
        "Image Mean Abs. Diff. in Count",
        "Image R Squared",
        "Patch R Squared",
        "Image Mean Abs. Diff. in Count at Optimal Score Thresh.",
        "Optimal Score Thresh."
    ]

    for dataset_size in run_record["dataset_sizes"]:
        for method in run_record["methods"]:

            if method not in results["results"]:
                results["results"][method] = {
                    "MS COCO mAP": [],
                    "PASCAL VOC mAP": [],
                    "Image Mean Abs. Diff. in Count": [],
                    "Image R Squared": [],
                    "Patch R Squared": [],
                    "Image Mean Abs. Diff. in Count at Optimal Score Thresh.": [],
                    "Optimal Score Thresh.": []
                }

            job_uuid = run_record["job_uuids"][i]
            i += 1

            job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
            job_config = json_io.load_json(job_config_path)
            ms_coco_mAPs = []
            pascal_voc_mAPs = []
            mean_abs_diffs = []
            image_r_squareds = []
            patch_r_squareds = []
            opt_mean_abs_diffs = []
            opt_score_threshs = []
            #results[method][dataset_size]["per_image_ms_coco_mAP_vals"] = []
            for model_info in job_config["model_info"]:
                
                model_uuid = model_info["model_uuid"]

                metrics_path = os.path.join("usr", "data", "results", 
                                            target_farm_name, target_field_name, target_mission_date,
                                            job_uuid, model_uuid, "metrics.json")

                metrics = json_io.load_json(metrics_path)
                point_metrics = metrics["point"]
                image_metrics = metrics["image"]
                ms_coco_mAP = point_metrics["Image MS COCO mAP"]["---"]
                pascal_voc_mAP = point_metrics["Image PASCAL VOC mAP"]["---"]
                mean_abs_diff = point_metrics["Image Mean Abs. Diff. in Count"]["plant"]
                image_r_squared = point_metrics["Image R Squared"]["plant"]
                patch_r_squared = point_metrics["Patch R Squared"]["plant"]
                opt_mean_abs_diff = point_metrics["optimal_score_threshold"]["mean_absolute_difference"]
                opt_score_thresh = point_metrics["optimal_score_threshold"]["threshold_value"]
                #per_image_ms_coco_mAP_vals = []
                #for image_name in image_metrics.keys():
                #    if "Image MS COCO mAP" in image_metrics[image_name]:
                #        per_image_ms_coco_mAP_vals.append(image_metrics[image_name]["Image MS COCO mAP"])

                pascal_voc_mAPs.append(pascal_voc_mAP)
                ms_coco_mAPs.append(ms_coco_mAP)
                mean_abs_diffs.append(mean_abs_diff)
                image_r_squareds.append(image_r_squared)
                patch_r_squareds.append(patch_r_squared)
                opt_mean_abs_diffs.append(opt_mean_abs_diff)
                opt_score_threshs.append(opt_score_thresh)

            results["results"][method]["MS COCO mAP"].append(np.mean(ms_coco_mAPs))
            results["results"][method]["PASCAL VOC mAP"].append(np.mean(pascal_voc_mAPs))
            results["results"][method]["Image Mean Abs. Diff. in Count"].append(np.mean(mean_abs_diffs))
            results["results"][method]["Image R Squared"].append(np.mean(image_r_squareds))
            results["results"][method]["Patch R Squared"].append(np.mean(patch_r_squareds))
            results["results"][method]["Image Mean Abs. Diff. in Count at Optimal Score Thresh."].append(np.mean(opt_mean_abs_diffs))
            results["results"][method]["Optimal Score Thresh."].append(np.mean(opt_score_threshs))
            #results[method][dataset_size]["per_image_ms_coco_mAP_vals"].append(per_image_ms_coco_mAP_vals)


    for method in run_record["methods"]:
        for metric in display_metrics:
            results["results"][method][metric] = (np.array(results["results"][method][metric])[inds]).tolist()

    run_results_dir = os.path.join("usr", "data", "runs", "display", 
                                   target_farm_name, target_field_name, target_mission_date)

    if os.path.exists(run_results_dir):
        raise RuntimeError("run results already exist")

    else:
        os.makedirs(run_results_dir)
    
    results_path = os.path.join(run_results_dir, "results.json")
    json_io.save_json(results_path, results)



def report(run_uuid):
    run_record_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    run_record = json_io.load_json(run_record_path)
    # target_farm_name = run_record["target_datasets"][0]["target_farm_name"]
    # target_field_name = run_record["target_datasets"][0]["target_field_name"]
    # target_mission_date = run_record["target_datasets"][0]["target_mission_date"]

    results = {}
    i = 0
    for dataset in run_record["target_datasets"]:
        for dataset_size in run_record["dataset_sizes"]:

            for method in run_record["methods"]:
                if method not in results:
                    results[method] = {}
                if dataset_size not in results[method]:
                    results[method][dataset_size] = {}
                    
                #for job_uuid in run_record["job_uuids"]:
                job_uuid = run_record["job_uuids"][i]
                i += 1

                job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
                job_config = json_io.load_json(job_config_path)
                ms_coco_mAPs = []
                mean_abs_diffs = []
                results[method][dataset_size]["per_image_ms_coco_mAP_vals"] = []
                for model_info in job_config["model_info"]:
                    
                    model_uuid = model_info["model_uuid"]

                    metrics_path = os.path.join("usr", "data", "results", 
                                                dataset["target_farm_name"], dataset["target_field_name"], dataset["target_mission_date"],
                                                job_uuid, model_uuid, "metrics.json")

                    metrics = json_io.load_json(metrics_path)
                    point_metrics = metrics["point"]
                    image_metrics = metrics["image"]
                    ms_coco_mAP = point_metrics["Image MS COCO mAP"]["---"]
                    mean_abs_diff = point_metrics["Image Mean Abs. Diff. in Count"]["plant"]

                    per_image_ms_coco_mAP_vals = []
                    for image_name in image_metrics.keys():
                        if "Image MS COCO mAP" in image_metrics[image_name]:
                            per_image_ms_coco_mAP_vals.append(image_metrics[image_name]["Image MS COCO mAP"])

                    ms_coco_mAPs.append(ms_coco_mAP)
                    mean_abs_diffs.append(mean_abs_diff)

                results[method][dataset_size]["ave_ms_coco"] = np.mean(ms_coco_mAPs)
                results[method][dataset_size]["ave_mean_abs_diff"] = np.mean(mean_abs_diffs)
                results[method][dataset_size]["per_image_ms_coco_mAP_vals"].append(per_image_ms_coco_mAP_vals)


    run_results_dir = os.path.join("usr", "data", "runs", run_uuid)
    if os.path.exists(run_results_dir):
        shutil.rmtree(run_results_dir)
    os.makedirs(run_results_dir)

    method_colors = {
        "direct": "red",
        "even_subset":"green",
        "graph_subset": "blue",
        "graph_subset_basic": "orange"
    }
    

    plt.figure(0)
    for method in results.keys():
        dataset_sizes = []
        vals = []
        for dataset_size in results[method].keys():
            dataset_sizes.append(dataset_size)
            vals.append(results[method][dataset_size]["ave_ms_coco"])
        
        plt.plot(dataset_sizes, vals, color=method_colors[method], 
                 marker='o', linestyle='dashed', linewidth=2, markersize=8, label=method)

    plt.legend()
    plt.title("Average MS COCO mAP value by method")

    plt.savefig(os.path.join(run_results_dir, "ms_coco_plot.png"))


    plt.figure(1)
    for method in results.keys():
        dataset_sizes = []
        vals = []
        for dataset_size in results[method].keys():
            dataset_sizes.append(dataset_size)
            vals.append(results[method][dataset_size]["ave_mean_abs_diff"])
        
        plt.plot(dataset_sizes, vals, color=method_colors[method], 
                 marker='o', linestyle='dashed', linewidth=2, markersize=8, label=method)

    plt.legend()
    plt.title("Average Absolute Difference in Count")

    plt.savefig(os.path.join(run_results_dir, "abs_diff_plot.png"))


    #val_i = 0
    v_colors = []
    vals = []
    for method in results.keys():
        for dataset_size in results[method].keys():
            for i in range(len(results[method][dataset_size]["per_image_ms_coco_mAP_vals"])):
                per_image_ms_coco_mAP_vals = results[method][dataset_size]["per_image_ms_coco_mAP_vals"][i]
                vals.append(per_image_ms_coco_mAP_vals)
                v_colors.append(method_colors[method])
        #val_i += 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    parts = ax.violinplot(
                vals
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(v_colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    plt.savefig(os.path.join(run_results_dir, "ms_coco_boxplot.png"))

    # for dataset in target_datasets:
    #     for dataset_size in dataset_sizes:
    #         for method in methods:


    #             job_results_dir = os.path.join("usr", "data", "results",
    #                                 dataset["target_farm_name"], dataset["target_field_name"], dataset["target_mission_date"],
    #                                 job_uuid)


    #             job_config = json_io.load_json(job_config_path)
    #             model_info = job_config["model_info"]

    #             for i, model_results_dir in enumerate(glob.glob(os.path.join(job_results_dir, "*"))):

    #                 metrics_path = os.path.join(model_results_dir, "metrics.json")
    #                 metrics = json_io.load_json(metrics_path)

    #                 #boxplot_metrics = metrics["boxplot"]
    #                 #abs_diff_in_count = boxplot_metrics["Difference in Count (Image)"]["plant"]
    #                 #print("abs_diff_in_count", abs_diff_in_count)
    #                 #pct_diff_in_count = boxplot_metrics["Percent Difference in Count (Image)"]["plant"]

    #                 #predictions_path = os.path.join(model_results_dir, "predictions.json")
    #                 #predictions = json_io.load_json(predictions_path)

    #                 #for image_name in predictions["image_predictions"][image_name]
    #                 #    predictions["image_predictions"][image_name]["pred_scores"]
                    
    #                 image_metrics = metrics["image"]
                   

    #                 for image_name in image_metrics.keys():
    #                     if "Image MS COCO mAP" in image_metrics[image_name]:
    #                         data["Image MS COCO mAP"][method][i].append(image_metrics[image_name]["Image MS COCO mAP"])

                        



# if __name__ == "__main__":
#     run_tests()



def run_all_dep():

    targets = []
    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                targets.append({
                    "target_farm_name": farm_name,
                    "target_field_name": field_name,
                    "target_mission_date": mission_date
                })


    target = targets[0]
    supplementary_targets = targets[1:]



    job_uuid = str(uuid.uuid4())

    job_config = {
        "job_uuid": job_uuid,
        "replications": 1,
        "job_name": "train_on_everything_test_" + job_uuid,
        "source_construction_params": {
            "method_name": "everything",
            "method_params": [],
        },
        "target_farm_name": target["target_farm_name"],
        "target_field_name": target["target_field_name"],
        "target_mission_date": target["target_mission_date"],
        "predict_on_completed_only": False,
        "supplementary_targets": [],
        "tol_test": 30,
        "test_reserved_images": [],
        "supplementary_targets": supplementary_targets

    }
    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    json_io.save_json(job_config_path, job_config)
    job_interface.run_job(job_uuid)


def run_all():
    training_image_sets = []
    test_image_sets = []
    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                annotations_path = os.path.join("usr", "data", "image_sets", 
                                        farm_name, field_name, mission_date, 
                                        "annotations", "annotations_w3c.json")

                annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})


                completed_images = w3c_io.get_completed_images(annotations)
                num_annotations = w3c_io.get_num_annotations(annotations, require_completed=True)


                if len(completed_images) > 0 and num_annotations > 30:
                
                    training_image_sets.append({
                        "farm_name": farm_name,
                        "field_name": field_name,
                        "mission_date": mission_date,
                        "test_reserved_images": []
                    })
                    test_image_sets.append({
                        "farm_name": farm_name,
                        "field_name": field_name,
                        "mission_date": mission_date
                    })


    job_uuid = str(uuid.uuid4())

    job_config = {
        "job_uuid": job_uuid,
        "replications": 1,
        "job_name": "refactor_test_" + job_uuid,

        "training": {
            "image_sets": training_image_sets
        },
        "inference": {
            "image_sets": test_image_sets
        }

    }
    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    json_io.save_json(job_config_path, job_config)
    job_interface.run_job(job_uuid)

def run_test():

    job_uuid = str(uuid.uuid4())

    job_config = {
        "job_uuid": job_uuid,
        "replications": 1,
        "job_name": "refactor_test_" + job_uuid,

        "training": {
            "image_sets": [
                {
                    "farm_name": "Saskatoon",
                    "field_name": "Norheim1",
                    "mission_date": "2021-05-26",
                    "test_reserved_images": []
                }
            ]
        },
        "inference": {
            "image_sets": [
                {
                    "farm_name": "Saskatoon",
                    "field_name": "Norheim1",
                    "mission_date": "2021-05-26"
                }
            ]
        }

    }
    job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
    json_io.save_json(job_config_path, job_config)
    job_interface.run_job(job_uuid)



def run_test_dep():
    dataset_sizes = [0] #[500, 5000, 10000, 20000] #20000] #3000] #[15000] #250] #10000]
    methods = ["direct_tiled"] #"transfer"] #"direct_tiled"] #transfer"] #["direct_tiled"] #["transfer"] #"direct_tiled"]

    method_params = {
            "match_method": "bipartite_b_matching",
            "extraction_type": "excess_green_box_combo",
            "patch_size": "image_set_dependent",
            "exclude_target_from_source": True 
    }

    target_datasets = [
        {
            "target_farm_name": "Saskatoon", # "BlaineLake", #"row_spacing", #"UNI", #"row_spacing",
            "target_field_name": "Norheim2", #"HornerWest", #"brown", #"LowN1", #"River", #"brown",
            "target_mission_date": "2021-05-26" #"2021-06-09" #2021-06-01" # "2021-06-07" #"2021-06-01" #-low-res"
        }
    ]

    epoch_patience = {
        256: 30,
        1024: 30,
        4096: 30,
        8192: 30,
        16384: 30
    }
    # run_record = {
    #     "job_uuids": [],
    #     "method_params": method_params,
    #     "methods": methods,
    #     "dataset_sizes": dataset_sizes,
    #     "target_datasets": target_datasets,
    #     "explanation": "test limits of repeated extraction"
    # }

    test_reserved_images = []



    #all_images = ["3", "4", "5", "7", "22", "23", "28", "30", "34", "39", "45", "48", "50", "52"]
    #training_images = ["7"]
    
    #all_images = ["1", "4", "6", "8", "11", "14", "16", "18", "21", "26", "27", "31", "36", "40"]
    #training_images = ["1", "4", "6", "8", "11", "14", "16", "18", "21", "26", "27", "31", "36", "40"] #["27"]
    #test_reserved_images = [image for image in all_images if image not in training_images]

    #test_reserved_images = ["204", "311", "805", "810", "817", "819", "821", "824"]


    for dataset in target_datasets:
        for dataset_size in dataset_sizes:
            for i, method in enumerate(methods):
                job_uuid = str(uuid.uuid4())

                job_config = {
                    "job_uuid": job_uuid,
                    "replications": 1,
                    "job_name": "direct_test_" + job_uuid,
                    "source_construction_params": {
                        "method_name": method,
                        "method_params": method_params,
                        #"size": dataset_size
                    },
                    "target_farm_name": dataset["target_farm_name"],
                    "target_field_name": dataset["target_field_name"],
                    "target_mission_date": dataset["target_mission_date"],
                    "predict_on_completed_only": False, #True, #True,
                    "supplementary_targets": [],
                    "tol_test": 30, #30, #epoch_patience[dataset_size]
                    "test_reserved_images": test_reserved_images,
                    "supplementary_targets": [{
                    #     "target_farm_name": "Biggar",
                    #     "target_field_name": "Dennis1",
                    #     "target_mission_date": "2021-06-04"
                        "target_farm_name": "Saskatoon",
                        "target_field_name": "Norheim1",
                        "target_mission_date": "2021-05-26"
                    }],
                    # "variation_config": {
                    #     "param_configs": ["inference", "inference"],
                    #     "param_names": ["rm_edge_boxes", "patch_border_buffer_percent"],
                    #     "param_values": [[True, None]] #, [False, 0.10], [False, None]]
                    # },
                    "variation_config": {
                        "param_configs": ["training"],
                        "param_names": ["source_construction_params/size"],
                        "param_values": [["max"]] #, [False, 0.10], [False, None]]
                    }

                }
                job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
                json_io.save_json(job_config_path, job_config)
                job_interface.run_job(job_uuid)


    #             run_record["job_uuids"].append(job_uuid)
    
    # run_uuid = str(uuid.uuid4())
    # run_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    # json_io.save_json(run_path, run_record)
    # prepare_report_for_display(run_uuid)
