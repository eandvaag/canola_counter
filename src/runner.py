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
from io_utils import json_io

dataset_sizes = [256, 1024, 4096, 8192] #, 512, 1024, 2048, 4096]
methods = ["direct", "even_subset", "graph_subset"] #, "even_subset", "graph_subset"]

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
        "target_farm_name": "BlaineLake",
        "target_field_name": "HornerWest",
        "target_mission_date": "2021-06-09"
    }
]




def run_tests():
    run_record = {
        "job_uuids": [],
        "method_params": method_params,
        "methods": methods,
        "dataset_sizes": dataset_sizes,
        "target_datasets": target_datasets
    }

    for dataset in target_datasets:
        for dataset_size in dataset_sizes:
            for method in methods:
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
                    "predict_on_completed_only": True,
                    "supplementary_targets": []
                }
                job_config_path = os.path.join("usr", "data", "jobs", job_uuid + ".json")
                json_io.save_json(job_config_path, job_config)
                job_interface.run_job(job_uuid)

                run_record["job_uuids"].append(job_uuid)

    run_uuid = str(uuid.uuid4())
    run_path = os.path.join("usr", "data", "runs", run_uuid + ".json")
    json_io.save_json(run_path, run_record)




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
                for model_info in job_config["model_info"]:
                    model_uuid = model_info["model_uuid"]

                    metrics_path = os.path.join("usr", "data", "results", 
                                                dataset["target_farm_name"], dataset["target_field_name"], dataset["target_mission_date"],
                                                job_uuid, model_uuid, "metrics.json")

                    metrics = json_io.load_json(metrics_path)
                    point_metrics = metrics["point"]
                    ms_coco_mAP = point_metrics["---"]["Image MS COCO mAP"]
                    ms_coco_mAPs.append(ms_coco_mAP)

                job_ms_coco_mAP = np.mean(ms_coco_mAPs)
                results[method][dataset_size] = job_ms_coco_mAP

    method_colors = {
        "direct": "red",
        "even_subset":"green",
        "graph_subset": "blue"
    }

    for method in results.keys():
        mAP_vals = []
        for dataset_size in results[method].keys():
            dataset_sizes.push(dataset_size)
            mAP_vals.push(results[method][dataset_size])
        
        plt.plot(dataset_sizes, mAP_vals, color=method_colors[method], 
                 marker='o', linestyle='dashed', linewidth=2, markersize=12, label=method)

    plt.legend()
    plt.title("Average MS COCO mAP value by method")
    run_results_dir = os.path.join("usr", "data", "runs", run_uuid)
    if os.path.exists(run_results_dir):
        shutil.rmtree(run_results_dir)
    os.makedirs(run_results_dir)
    plt.savefig(os.path.join(run_results_dir, "ms_coco_plot.png"))


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