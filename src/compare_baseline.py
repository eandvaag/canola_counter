import os
import shutil
import time
import random
import uuid
import urllib3
import logging

import image_set_actions as isa
import server
from models.common import annotation_utils
from io_utils import json_io
import fine_tune_eval

from lock_queue import LockQueue

def test(test_sets, baselines, num_reps):

    for rep_num in range(num_reps):
        for test_set in test_sets:

            org_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
            
            print("\n\nprocessing test_set: {}\n\n".format(org_image_set_dir))
            

            annotations_path = os.path.join(org_image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            num_iterations = 2
            image_names = list(annotations.keys())

            metadata_path = os.path.join(org_image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)
            candidates = image_names.copy()
            training_images = []
            for iteration_number in range(num_iterations):

                candidates = []
                for image_name in annotations.keys():
                    if len(annotations[image_name]["training_regions"]) == 0:
                        candidates.append(image_name)

                chosen_images = random.sample(candidates, 5)

                for chosen_image in chosen_images:
                    annotations[chosen_image]["training_regions"].append([0, 0, metadata["images"][chosen_image]["height_px"], metadata["images"][chosen_image]["width_px"]])
                
                training_images.append(chosen_images)

            # if test_set["farm_name"] == "BlaineLake" and test_set["field_name"] == "Serhienko9S":
            #     training_images = [["8", "19", "21", "38", "44"]]
            # elif test_set["farm_name"] == "BlaineLake" and test_set["field_name"] == "Serhienko15":
            #     training_images = [["11", "12", "18", "19", "29"]]
            # elif test_set["farm_name"] == "SaskatoonEast" and test_set["field_name"] == "Stevenson5SW":
            #     training_images = [["22", "25", "27", "35", "40"]]
            
            


            regions = []
            for image_name in image_names:
                regions.append([[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]])
            # print("regions", regions)
            for baseline in baselines:

                baseline_username = test_set["username"]
                baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                baseline_mission_date = test_set["mission_date"]

                baseline_dir = os.path.join("usr", "data",
                                                        baseline_username, "image_sets",
                                                        baseline_farm_name,
                                                        baseline_field_name,
                                                        baseline_mission_date)
                
                
                        
                print("\n\nprocessing baseline: {}\n\n".format(baseline_dir))

                field_dir = os.path.join("usr", "data",
                                                        baseline_username, "image_sets",
                                                        baseline_farm_name,
                                                        baseline_field_name)                                       
                #                         method["image_set"]["farm_name"], method["image_set"]["field_name"], method["image_set"]["mission_date"])
                # field_dir = os.path.join("usr", "data", method["image_set"]["username"], "image_sets",
                #                             method["image_set"]["farm_name"], method["image_set"]["field_name"])


                if os.path.exists(field_dir):
                    raise RuntimeError("Field dir exists!")
                
                print("copying directory")
                os.makedirs(field_dir)
                shutil.copytree(org_image_set_dir, baseline_dir)

                print("switching to model")
                model_dir = os.path.join(baseline_dir, "model")
                switch_req_path = os.path.join(model_dir, "switch_request.json")
                switch_req = {
                    "model_name": baseline["model_name"],
                    "model_creator": baseline["model_creator"]
                }
                json_io.save_json(switch_req_path, switch_req)

                item = {
                    "username": baseline_username,
                    "farm_name": baseline_farm_name,
                    "field_name": baseline_field_name,
                    "mission_date": baseline_mission_date
                }


                switch_processed = False
                isa.process_switch(item)
                while not switch_processed:
                    print("Waiting for process switch")
                    time.sleep(10)
                    if not os.path.exists(switch_req_path):
                        switch_processed = True


                for iteration_number in range(num_iterations):

                    if iteration_number > 0:

                        annotations_path = os.path.join(baseline_dir, "annotations", "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        for image_name in training_images[iteration_number-1]:
                            annotations[image_name]["training_regions"].append(regions[0][0])

                        annotation_utils.save_annotations(annotations_path, annotations)

                        server.sch_ctx["training_queue"].enqueue(item)
                            

                        train_queue_size = server.sch_ctx["training_queue"].size()
                        print("train_queue_size", train_queue_size)
                        while train_queue_size > 0:
                            item = server.sch_ctx["training_queue"].dequeue()
                            print("running process_train")

                            re_enqueue = server.process_train(item)



                            if re_enqueue:
                                server.sch_ctx["training_queue"].enqueue(item)
                            train_queue_size = server.sch_ctx["training_queue"].size()


                
                    request_uuid = str(uuid.uuid4())
                    request = {
                        "request_uuid": request_uuid,
                        "start_time": int(time.time()),
                        "image_names": image_names,
                        "regions": regions,
                        "save_result": True,
                        "results_name": "fine_tune_" + str(iteration_number),
                        "results_message": ""
                    }

                    request_path = os.path.join(baseline_dir, "model", "prediction", 
                                                "image_set_requests", "pending", request_uuid + ".json")

                    json_io.save_json(request_path, request)
                    print("running process_predict")
                    server.process_predict(item)


def plot_my_results(test_sets, baselines, num_reps):

    for rep_num in range(num_reps):

        for test_set in test_sets:
            methods = []
            for baseline in baselines:
                baseline_username = test_set["username"]
                baseline_farm_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["farm_name"] + "_rep_" + str(rep_num)
                baseline_field_name = "BASELINE_TEST:" + baseline["model_creator"] + ":" + baseline["model_name"] + ":" + test_set["field_name"] + "_rep_" + str(rep_num)
                baseline_mission_date = test_set["mission_date"]
                method = {
                    "image_set": {
                        "username": baseline_username,
                        "farm_name": baseline_farm_name,
                        "field_name": baseline_field_name,
                        "mission_date": baseline_mission_date
                    },
                    "method_label": baseline["num_training_sets"]
                }
                methods.append(
                    method
                )
                
            test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]

            fine_tune_eval.create_global_comparison(methods, 
                                                    "accuracy", 
                                                    os.path.join("baseline_charts", "comparisons", test_set_str, "global", "accuracy", "rep_" + str(rep_num) + ".svg"),
                                                    xpositions="num_annotations", 
                                                    colors=None)
            fine_tune_eval.create_global_comparison(methods, 
                                                    "AP (IoU=.50)", 
                                                    os.path.join("baseline_charts", "comparisons", test_set_str, "global", "AP (IoU=.50)", "rep_" + str(rep_num) + ".svg"),
                                                    xpositions="num_annotations", 
                                                    colors=None)



def run():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.basicConfig(level=logging.INFO)


    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()

    baselines = [

        {
            "model_name": "MORSE_Nasser_2022-05-27",
            "model_creator": "erik",
            "num_training_sets": 1
        },
        # {
        #     "model_name":
        #     "model_creator": "erik",
        # }
        {
            "model_name": "set_of_3",
            "model_creator": "kaylie",
            "num_training_sets": 3
        },
        {
            "model_name": "set_of_6",
            "model_creator": "kaylie",
            "num_training_sets": 6
        },
        {
            "model_name": "set_of_12",
            "model_creator": "kaylie",
            "num_training_sets": 12
        },
        # {
        #     "model_name": "p2irc_symposium_2022",
        #     "model_creator": "kaylie",
        #     "num_training_sets": 16
        # },
        {
            "model_name": "all_stages",
            "model_creator": "kaylie",
            "num_training_sets": 27
        }
    ]

    test_sets = [
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko9S",
            "mission_date": "2022-06-07"
        },
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko11",
            "mission_date": "2022-06-07"
        },
        {
            "username": "erik",
            "farm_name": "BlaineLake",
            "field_name": "Serhienko15",
            "mission_date": "2022-06-14"
        },
        {
            "username": "erik",
            "farm_name": "SaskatoonEast",
            "field_name": "Stevenson5SW",
            "mission_date": "2022-06-13"
        }
    ]

    num_reps = 1

    # test(test_sets, baselines, num_reps)
    plot_my_results(test_sets, baselines, num_reps)


if __name__ == "__main__":
    run()
