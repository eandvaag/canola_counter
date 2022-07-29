import logging
import os
import glob
import shutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pyrsistent import field

from io_utils import json_io, w3c_io
import image_set_model as ism

def test():

    UNI_Dugout_2022_05_30 = [
        [
            -49,
            -1,
            -54,
            -39,
            -76
        ],
        [
            -39,
            0,
            -64,
            -49,
            -43
        ],
        [
            -35,
            0,
            -54,
            -36,
            -44
        ],
        [
            -27,
            9,
            -37,
            -29,
            -35
        ],
        [
            -25,
            5,
            -37,
            -19,
            -30
        ]
    ]

    MORSE_Dugout_2022_05_27 = [
        [
            -116,
            -56,
            -135
        ],
        [
            -71,
            -68,
            -59
        ],
        [
            -58,
            -60,
            -46
        ],
        [
            -45,
            -46,
            -42
        ],
        [
            -44,
            -38,
            -37
        ]

    ]

    Saskatoon_Norheim4_2022_05_24 = [
        [
            -67,
            -99,
            -117,
            -76,
            -81,
            -14,
            -62,
            -98,
            -62,
            5
        ],
        [
            -63,
            -76,
            -65,
            -22,
            -59,
            5,
            -26,
            -71,
            -34,
            3
        ],
        [
            -31,
            -14,
            -19,
            16,
            -34,
            99,
            -9,
            -47,
            -15,
            8
        ],
        [
            -26,
            8,
            -10,
            32,
            -8,
            155,
            6,
            -22,
            5,
            16
        ],
        [
            -27,
            0,
            -25,
            29,
            -18,
            78,
            -4,
            -30,
            -18,
            8
        ]
    ]

    MORSE_Nasser_2022_05_27 = [
        [
            -94,
            -91
        ],
        [
            -80,
            -83
        ],
        [
            -51,
            -65
        ],
        [
            -46,
            -59
        ],
        [
            -35,
            -39
        ]
    ]

    mAP_UNI_Dugout_2022_05_30 = [
        [
            61.9595408439636,
            65.7373428344727,
            56.4121782779694,
            56.5246403217316,
            59.3461513519287
        ],
        [
            67.0270383358002,
            74.466747045517,
            58.6017429828644,
            59.8606884479523,
            64.9820983409882
        ],
        [
            67.8766250610352,
            74.2677330970764,
            60.9893202781677,
            63.2199764251709,
            67.4891293048859
        ],
        [
            67.4691319465637,
            74.247533082962,
            57.9867959022522,
            62.8505825996399,
            66.8014168739319
        ],
        [
            70.8636581897736,
            75.4881739616394,
            63.6513769626617,
            65.5259370803833,
            69.9087977409363
        ]
    ]

    mAP_MORSE_Dugout_2022_05_27 = [
        [
            33.0529510974884,
            44.9342966079712,
            37.054842710495
        ],
        [
            43.8182920217514,
            51.0842740535736,
            53.6222338676453
        ],
        [
            47.8903859853745,
            56.0297846794128,
            59.287029504776
        ],
        [
            47.8504240512848,
            55.683833360672,
            58.2232475280762
        ],
        [
            51.1234879493713,
            59.6147060394287,
            61.7759704589844
        ]
    ]

    mAP_Saskatoon_Norheim4_2022_05_24  = [
        [
            20.6855863332748,
            21.9598576426506,
            14.4822031259537,
            16.9984877109528,
            33.9630395174026,
            10.2299772202969,
            18.7740415334702,
            22.4944397807121,
            5.21742179989815,
            0
        ],
        [
            42.3073053359985,
            40.8205837011337,
            44.4945245981216,
            44.8765426874161,
            48.6010313034058,
            35.4865252971649,
            51.2360811233521,
            45.1865434646606,
            33.0271661281586,
            0
        ],
        [
            52.8983771800995,
            55.450314283371,
            53.5616993904114,
            55.5438876152039,
            60.3344082832336,
            48.6010253429413,
            61.6125702857971,
            51.0525345802307,
            46.8347549438477,
            0
        ],
        [
            53.4228086471558,
            53.3597767353058,
            54.5240044593811,
            53.5442471504211,
            58.0634355545044,
            46.2687015533447,
            62.5221490859985,
            53.156989812851,
            43.4229075908661,
            0
        ],
        [
            55.7208180427551,
            54.9334406852722,
            53.0922055244446,
            58.5757076740265,
            59.3820035457611,
            45.291143655777,
            61.6315484046936,
            54.8845708370209,
            39.7374451160431,
            0
        ]
    ]


    mAP_MORSE_Nasser_2022_05_27 = [

        [
            43.8744306564331,
            44.0011590719223
        ],
        [
            52.3657560348511,
            50.5525052547455
        ],
        [
            58.1359684467316,
            56.9121778011322
        ],
        [
            56.5727293491364,
            55.5345177650452
        ],
        [
            62.2031092643738,
            57.5662851333618
        ]
    ]



    res_lst = [
        [UNI_Dugout_2022_05_30, "red"], 
        [MORSE_Dugout_2022_05_27, "green"],
        [Saskatoon_Norheim4_2022_05_24, "blue"],
        [MORSE_Nasser_2022_05_27, "orange"]
    ]

    mAP_res_lst = [
        [mAP_UNI_Dugout_2022_05_30, "red"], 
        [mAP_MORSE_Dugout_2022_05_27, "green"],
        [mAP_Saskatoon_Norheim4_2022_05_24, "blue"],
        [mAP_MORSE_Nasser_2022_05_27, "orange"]
    ]

    # x = []
    # y = []

    #fig, ax = plt.figure()

    for m, res in enumerate([mAP_res_lst, res_lst]):
        fig, ax = plt.subplots()
        for r in res:
            results = r[0]
            color = r[1]
            for i in range(len(results[0])):
                line = [results[j][i] for j in range(len(results))]
                #for q in results:
                ax.plot([1,2,3,4,5], line, color=color, alpha=0.8)
                # x.append([1,2,3])
                # y.append(q)

            
    
        if m == 0:
            ax.set_ylim([0, 100])
        else:
            ylim = ax.get_ylim()
            max_y = max(abs(ylim[0]), abs(ylim[1]))
            ax.set_ylim([-max_y, max_y])

        plt.xlabel("Number of Image Sets Trained On")
        plt.xticks(ticks=[1,2,3,4,5], labels=["1", "3", "5", "8", "11"])

        if m == 0:
            plt.ylabel("MS COCO mAP")
            plt.savefig("mAP_values.svg")
        else:
            plt.ylabel("Predicted Count Minus Annotated Count")
            plt.savefig("count_differences.svg")


        
        plt.close()


def create_bar_chart():

    # farm_name = "Saskatoon"
    # field_name = "Norheim4"
    # mission_date = "2022-05-24"

    chart_data = json_io.load_json("yolov4_chart_data.json")
    tiny_chart_data = json_io.load_json("yolov4_tiny_chart_data.json")

    image_names = []
    count_diffs = []
    tiny_count_diffs = []
    anno_counts = []

    for farm_name in chart_data.keys():
        for field_name in chart_data[farm_name].keys():
            for mission_date in chart_data[farm_name][field_name].keys():

                annotations = w3c_io.load_annotations(os.path.join("usr", "data", "kaylie", "image_sets",
                    farm_name, field_name, mission_date, "annotations", "annotations_w3c.json"), {"plant": 0})
                for image_name in chart_data[farm_name][field_name][mission_date].keys():
                    image_names.append(image_name)
                    # count_diffs.append(chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1])
                    # tiny_count_diffs.append(tiny_chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1])
                    
                    # count_diffs.append(chart_data[farm_name][field_name][mission_date][image_name]["MS_COCO_mAP"][-1])
                    # tiny_count_diffs.append(tiny_chart_data[farm_name][field_name][mission_date][image_name]["MS_COCO_mAP"][-1])

                    anno_count = len(annotations[image_name]["boxes"])
                    count_diffs.append(chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1] + anno_count)
                    tiny_count_diffs.append(tiny_chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"][-1] + anno_count)
                    anno_counts.append(anno_count)

    x = np.arange(len(image_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 4))
    # rects1 = ax.bar(x - width/2, count_diffs, width, label='YOLOv4')
    # rects2 = ax.bar(x + width/2, tiny_count_diffs, width, label='YOLOv4 Tiny')

    rects1 = ax.bar(x - width/2, anno_counts, width/2, label='Annotated')
    rects2 = ax.bar(x, count_diffs, width/2, label='YOLOv4')
    rects3 = ax.bar(x + width/2, tiny_count_diffs, width/2, label='YOLOv4 Tiny')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Predicted Count Minus Annotated Count')
    # ax.set_title('Count Differences for models trained on 11 image sets')

    # ax.set_ylabel('MS COCO mAP')
    # ax.set_title('MS COCO mAP values for models trained on 11 image sets')

    ax.set_ylabel('Count')
    ax.set_title('Count predictions for models trained on 11 image sets')
    ax.set_xticks(x, image_names)
    ax.legend()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.savefig("count_comparison.svg")

def recreate_plots():

    chart_data = json_io.load_json("yolov4_tiny_chart_data.json")
    metrics = [
        "predicted_count_minus_annotated_count", "MS_COCO_mAP"
    ]
    tick_labels = ["1", "3", "5", "8", "11"] 
    
    baseline_names = ["UNI::LowN1::2021-06-07", "3_sets_0", "5_sets_0", "8_sets_0", "11_sets_0"]
    # baseline_names = ["full_yolov4_UNI::LowN1::2021-06-07", "full_yolov4_3_sets_0", "full_yolov4_5_sets_0", 
    # "full_yolov4_8_sets_0", "full_yolov4_11_sets_0"]

    colors = ["blue", "red", "green", "orange", "purple", "black", "pink"]
    color_lookup = {
        "UNI": {
            "Dugout": {
                "2022-05-30": "blue"
            }
        },
        "MORSE": {
            "Nasser": {
                "2022-05-27": "green"
            },
            "Dugout": {
                "2022-05-27": "red"
            },
        },
        "Saskatoon": {
            "Norheim4": {
                "2022-05-24": "orange"
            },
            "Norheim5": {
                "2022-05-24": "purple"
            }
        }
    }


    for metric in metrics:
        fig, ax = plt.subplots()
        for farm_name in chart_data.keys():
            for field_name in chart_data[farm_name].keys():
                for mission_date in chart_data[farm_name][field_name].keys():

                    # annotations = w3c_io.load_annotations(os.path.join("usr", "data", "kaylie", "image_sets",
                    #     farm_name, field_name, mission_date, "annotations", "annotations_w3c.json"), {"plant": 0})
                    for image_name in chart_data[farm_name][field_name][mission_date].keys():

                        line_data = chart_data[farm_name][field_name][mission_date][image_name][metric]
                        # anno_count = len(annotations[image_name]["boxes"])
                        # line_data = (chart_data[farm_name][field_name][mission_date][image_name]["predicted_count_minus_annotated_count"])
                        # line_data = [(abs(x) / anno_count) * 100 if anno_count > 0 else 0 for x in line_data]
                        ax.plot([i for i in range(len(baseline_names))], line_data, 
                        color=color_lookup[farm_name][field_name][mission_date], alpha=0.8)



    
        if metric == "MS_COCO_mAP":
            ax.set_ylim([0, 100])
        else:
            ylim = ax.get_ylim()
            max_y = max(abs(ylim[0]), abs(ylim[1]))
            ax.set_ylim([-max_y, max_y])
            ax.set_ylim([-220, 220])

        # ax.set_ylim([0, 100])

        plt.xlabel("Number of Image Sets Trained On")
        plt.xticks(ticks=[i for i in range(len(baseline_names))], labels=tick_labels)

        if metric == "MS_COCO_mAP":
            plt.title("YOLOv4 Tiny MS COCO mAP Values")
            plt.ylabel("MS COCO mAP")
            plt.savefig("yolov4_tiny_mAP_values.svg")
        else:
            plt.title("YOLOv4 Tiny Count Differences")
            plt.ylabel("Predicted Count Minus Annotated Count")
            plt.savefig("yolov4_tiny_count_differences.svg")
        # plt.title("YOLOv4 Tiny Percent Error in Count")
        # plt.ylabel("Percent Error in Count")
        # plt.savefig("yolov4_tiny_percent_error.svg")




def generate_plots():
    logging.basicConfig(level=logging.INFO)

    #baseline_names = ["full_yolov4_11_sets_0"] #"UNI::LowN1::2021-06-07", "3_sets_0", "5_sets_0", "8_sets_0"]
    #baseline_names = ["full_yolov4_UNI::LowN1::2021-06-07", "full_yolov4_3_sets_0", "full_yolov4_5_sets_0", "full_yolov4_8_sets_0"]
    baseline_names = ["11_sets_0_no_overlap", "11_sets_0_no_overlap_augmented"]
    
    #tick_labels = ["1", "3", "5", "8"]
    tick_labels = ["11_sets_0_no_overlap", "11_sets_0_no_overlap_augmented"]
    chart_data = {}

    colors = ["blue", "red", "green", "orange", "purple", "black", "pink"]
    color_index = 0
    color_lookup = {}
    metrics = [
        "predicted_count_minus_annotated_count", "MS_COCO_mAP"
    ]

    for farm_path in glob.glob(os.path.join("usr", "data", "kaylie", "image_sets", "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)
                #if farm_name == "MORSE" and field_name == "Dugout" and mission_date == "2022-05-27":
                #if farm_name == "Saskatoon" and field_name == "Norheim4" and mission_date == "2022-05-24": 

                annotations_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                annotations = json_io.load_json(annotations_path)
                image_names = [image_name for image_name in annotations.keys() if annotations[image_name]["status"] == "completed_for_testing"]


                if farm_name not in chart_data:
                    chart_data[farm_name] = {}
                    color_lookup[farm_name] = {}
                if field_name not in chart_data[farm_name]:
                    chart_data[farm_name][field_name] = {}
                    color_lookup[farm_name][field_name] = {}
                if mission_date not in chart_data[farm_name][field_name]:
                    chart_data[farm_name][field_name][mission_date] = {}
                    color_lookup[farm_name][field_name][mission_date] = colors[color_index]
                    color_index += 1
                for image_name in image_names:
                    chart_data[farm_name][field_name][mission_date][image_name] = {}
                    for metric in metrics:
                        chart_data[farm_name][field_name][mission_date][image_name][metric] = []

                for baseline_name in baseline_names:

                    baseline_src_path = os.path.join("usr", "additional", "baselines", baseline_name + ".h5")
                    baseline_dst_path = os.path.join(mission_path, "model", "weights", "best_weights.h5")

                    shutil.copyfile(baseline_src_path, baseline_dst_path)
                    # request = {
                    #     "farm_name": farm_name,
                    #     "field_name": field_name,
                    #     "mission_date": mission_date,
                    #     "image_names": [image_name for image_name in annotations.keys() if annotations[image_name]["status"] == "completed_for_testing"],
                    #     "save_result": True
                    # }

                    # request_uuid = str(uuid.uuid4())
                    # request_path = os.path.join("usr", "requests", "prediction", request_uuid + ".json")

                    # json_io.save_json(request_path, request)

                    ism.predict_on_images(
                        "kaylie",
                        farm_name,
                        field_name,
                        mission_date,
                        image_names,
                        True
                    )

                    os.remove(baseline_dst_path)


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



    # print("chart_data", chart_data)
    # json_io.save_json("yolov4_11_chart_data.json", chart_data)

    # exit()

    for metric in metrics:
        fig, ax = plt.subplots()
        for farm_name in chart_data.keys():
            for field_name in chart_data[farm_name].keys():
                for mission_date in chart_data[farm_name][field_name].keys():
                    for image_name in chart_data[farm_name][field_name][mission_date].keys():

                        line_data = chart_data[farm_name][field_name][mission_date][image_name][metric]
                        ax.plot([i for i in range(len(baseline_names))], line_data, 
                        color=color_lookup[farm_name][field_name][mission_date], alpha=0.8)



    
        if metric == "MS_COCO_mAP":
            ax.set_ylim([0, 100])
        else:
            ylim = ax.get_ylim()
            max_y = max(abs(ylim[0]), abs(ylim[1]))
            ax.set_ylim([-max_y, max_y])

        plt.xlabel("Number of Image Sets Trained On")
        plt.xticks(ticks=[i for i in range(len(baseline_names))], labels=tick_labels)

        if metric == "MS_COCO_mAP":
            plt.ylabel("MS COCO mAP")
            plt.savefig("res_mAP_values.svg")
        else:
            plt.ylabel("Predicted Count Minus Annotated Count")
            plt.savefig("res_count_differences.svg")


