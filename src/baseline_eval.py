import os
import glob
import shutil
import uuid
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import image_set_model as ism

from io_utils import json_io, w3c_io


def create_bar_chart():

    chart_data = json_io.load_json("domain_shift_chart_data.json")
    image_names = []
    predicted_counts = []
    anno_counts = []
    count_diffs = []
    colors = []


    for farm_name in chart_data.keys():
        for field_name in chart_data[farm_name].keys():
            for mission_date in chart_data[farm_name][field_name].keys():

                annotations = w3c_io.load_annotations(os.path.join("usr", "data", "kaylie", "image_sets",
                    farm_name, field_name, mission_date, "annotations", "annotations_w3c.json"), {"plant": 0})
                for image_name in chart_data[farm_name][field_name][mission_date].keys():
                    image_names.append(farm_name + "/" + field_name + "/" + mission_date + "/" + image_name)
                    # count_diffs.append(chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1])
                    # tiny_count_diffs.append(tiny_chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1])
                    
                    # count_diffs.append(chart_data[farm_name][field_name][mission_date][image_name]["MS_COCO_mAP"][-1])
                    # tiny_count_diffs.append(tiny_chart_data[farm_name][field_name][mission_date][image_name]["MS_COCO_mAP"][-1])

                    anno_count = len(annotations[image_name]["boxes"])
                    count_diffs.append(abs(chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1]))
                    predicted_counts.append(chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1] + anno_count)
                    anno_counts.append(anno_count)

                    if (farm_name == "Saskatoon" and field_name == "Norheim1" and mission_date == "2021-06-02"):
                        pass
                    elif mission_date[:4] == "2022" \
                        or (farm_name == "row_spacing" and field_name == "brown" and mission_date == "2021-06-01"):
                        
                        # print("testing", farm_name, field_name, mission_date)
                        colors.append("orange")
                    elif annotations[image_name]["status"] == "completed_for_testing":
                        # print("testing from training set", farm_name, field_name, mission_date)
                        colors.append("blue")
                    else:
                        # print("training", farm_name, field_name, mission_date)
                        colors.append("green")


    count_diffs = np.array(count_diffs)
    predicted_counts = np.array(predicted_counts)
    anno_counts = np.array(anno_counts)
    colors = np.array(colors)
    image_names = np.array(image_names)

    inds = np.argsort(count_diffs)

    predicted_counts = predicted_counts[inds]
    anno_counts = anno_counts[inds]
    colors = colors[inds]
    image_names = image_names[inds]

    for i, image_name in enumerate(image_names):
        print("{}: {}".format(i, image_name))

    x = np.arange(len(image_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(26, 6)) #36, 14))
    #fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, anno_counts, width, label='Annotated', color="grey")
    rects2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted', color=colors)



    ax.set_ylabel('Count')
    ax.set_title('Count predictions for model trained on 11 image sets')
    ax.set_xticks(x, np.arange(image_names.size)) #image_names)
    plt.xticks(fontsize=6, rotation=90)
    # ax.legend()
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="grey", lw=4),
                    Line2D([0], [0], color="orange", lw=4),
                    Line2D([0], [0], color="blue", lw=4),
                    Line2D([0], [0], color="green", lw=4),]

    
    #lines = ax.plot(data)
    ax.legend(custom_lines, ['Annotated', 'Test Image From Test Set', 'Test Image From Training Set', 'Training Image From Training Set'])


    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.15)
    plt.savefig("domain_shift_counts.svg")








def eval():
    chart_data = {}
    metrics = [
        "predicted_count_minus_annotated_count", "MS_COCO_mAP"
    ]

    for farm_path in glob.glob(os.path.join("usr", "data", "kaylie", "image_sets", "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                annotations_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                annotations = json_io.load_json(annotations_path)
                image_names = [image_name for image_name in annotations.keys() if annotations[image_name]["status"] == "completed_for_testing" or annotations[image_name]["status"] == "completed_for_training"]


                baseline_src_path = os.path.join("usr", "additional", "baselines","11_sets_1_one_image_per_set.h5") #_no_overlap.h5")
                baseline_dst_path = os.path.join(mission_path, "model", "weights", "best_weights.h5")

                shutil.copyfile(baseline_src_path, baseline_dst_path)


                if farm_name not in chart_data:
                    chart_data[farm_name] = {}
                    # color_lookup[farm_name] = {}
                if field_name not in chart_data[farm_name]:
                    chart_data[farm_name][field_name] = {}
                    # color_lookup[farm_name][field_name] = {}
                if mission_date not in chart_data[farm_name][field_name]:
                    chart_data[farm_name][field_name][mission_date] = {}
                    # color_lookup[farm_name][field_name][mission_date] = colors[color_index]
                    # color_index += 1
                for image_name in image_names:
                    chart_data[farm_name][field_name][mission_date][image_name] = {}
                    for metric in metrics:
                        chart_data[farm_name][field_name][mission_date][image_name][metric] = []

                request_uuid = str(uuid.uuid4())
                request = {
                    "request_uuid": request_uuid,
                    "start_time": int(time.time()),
                    "image_names": image_names,
                    "save_result": True
                }

                request_path = os.path.join("usr", "data", "kaylie", "image_sets",
                                            farm_name, field_name, mission_date, "model", 
                                            "prediction", "image_set_requests", "pending", request_uuid + ".json")     


                json_io.save_json(request_path, request)
                ism.check_predict("kaylie", farm_name, field_name, mission_date)



                results_dir = os.path.join(mission_path, "model", "results")

                cur_result_dir = sorted(glob.glob(os.path.join(results_dir, "*")))[-1]

                results_csv_path = os.path.join(cur_result_dir, "results.csv")

                results_df = pd.read_csv(results_csv_path)

                print("images: {}".format(chart_data[farm_name][field_name][mission_date].keys()))
                for index, row in results_df.iterrows():
                    print("row", row)
                    chart_data[farm_name][field_name][mission_date][str(row["image_name"])]["predicted_count_minus_annotated_count"].append(
                        row["predicted_plant_count"] - row["annotated_plant_count"])
                
                    chart_data[farm_name][field_name][mission_date][str(row["image_name"])]["MS_COCO_mAP"].append(row["MS_COCO_mAP"])



    json_io.save_json(os.path.join("usr", "additional", "chart_data", "11_sets_1_one_image_per_set_chart_data.json"), chart_data)

