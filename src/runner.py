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
