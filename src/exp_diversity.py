import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

from io_utils import json_io
from exp_runner import eval_test_sets, get_mapping_for_test_set, my_plot_colors, eval_single_630_baselines, eval_diverse_630_baselines
#  import fine_tune_eval
from models.common import annotation_utils, inference_metrics

test_set_str_to_number = {
        "Biggar/Dennis1/2021-06-04": 1,
        "Biggar/Dennis2/2021-06-12": 2,
        "Biggar/Dennis3/2021-06-04": 3,
        "BlaineLake/HornerWest/2021-06-09": 4,
        "BlaineLake/Lake/2021-06-09": 5,
        "BlaineLake/River/2021-06-09": 6,
        "BlaineLake/Serhienko9N/2022-06-07": 7,
        "BlaineLake/Serhienko9S/2022-06-14": 8,
        "BlaineLake/Serhienko10/2022-06-14": 9,
        "MORSE/Dugout/2022-05-27": 10,
        "MORSE/Nasser/2022-05-27": 11,
        "row_spacing/brown/2021-06-01": 12,
        "row_spacing/brown/2021-06-08": 13,
        "row_spacing/nasser/2021-06-01": 14, 
        "row_spacing/nasser2/2022-06-02": 15,
        "Saskatoon/Norheim1/2021-05-26": 16,
        "Saskatoon/Norheim1/2021-06-02": 17,
        "Saskatoon/Norheim2/2021-05-26": 18,
        "Saskatoon/Norheim4/2022-05-24": 19,
        "Saskatoon/Norheim5/2022-05-24": 20,
        "SaskatoonEast/Stevenson5NW/2022-06-20": 21,
        "UNI/Brown/2021-06-05": 22,
        "UNI/Dugout/2022-05-30": 23,
        "UNI/LowN1/2021-06-07": 24,
        "UNI/LowN2/2021-06-07": 25,
        "UNI/Sutherland/2021-06-05": 26,
        "UNI/Vaderstad/2022-06-16": 27
}













def get_diversity_results(test_sets, single_baselines, diverse_baselines, out_dir):

    mappings = {}
    results = {
        "single": [],
        "diverse": []
    }
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": []
        # }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    for i in range(2):
        if i == 0:
            baselines = single_baselines
            result_key = "single"
        else:
            baselines = diverse_baselines
            result_key = "diverse"

        for baseline in baselines:
            rep_accuracies = []
            rep_mean_abs_dics = []
            rep_mean_abs_dids = []
            for rep_num in range(5):

                model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                test_set_accuracies = []
                test_set_mean_abs_dics = []
                test_set_mean_abs_dids = []
                abs_dics = []
                # abs_dids = []
                for test_set in test_sets:
                    print(test_set)
                    # abs_dics = []
                    abs_dids = []
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
                    

                    metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
                    metadata = json_io.load_json(metadata_path)
                    camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
                    camera_specs = json_io.load_json(camera_specs_path)


                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                    # rep_accuracies = []
                    # for rep_num in range(3):
                    # model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    # excel_path = os.path.join(result_dir, "metrics.xlsx")
                    # df = pd.read_excel(excel_path, sheet_name=0)
                    # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                    # rep_accuracies.append(rep_accuracy)


                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)


                            anno_count = annotations[image_name]["boxes"].shape[0]
                            pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]
                            abs_dics.append(abs(anno_count - pred_count))

                            height_px = metadata["images"][image_name]["height_px"]
                            width_px = metadata["images"][image_name]["width_px"]
                            area_px = height_px * width_px

                            gsd = inference_metrics.get_gsd(camera_specs, metadata)
                            area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                            annotated_count_per_square_metre = anno_count / area_m2
                            predicted_count_per_square_metre = pred_count / area_m2

                            abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                            abs_dids.append(abs_did)
                            

                    test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images)
                    test_set_accuracies.append(test_set_accuracy)

                    # remove #
                    # mean_abs_dic = np.mean(abs_dics)
                    # test_set_mean_abs_dics.append(mean_abs_dic)
                    # end remove #

                    mean_abs_did = np.mean(abs_dids)
                    test_set_mean_abs_dids.append(mean_abs_did)


                rep_mean_abs_dic = np.mean(abs_dics) #test_set_mean_abs_dics) #abs_dics)
                rep_mean_abs_dics.append(rep_mean_abs_dic)

                rep_mean_abs_did = np.mean(test_set_mean_abs_dids) #test_set_mean_abs_dids) #abs_dics)
                rep_mean_abs_dids.append(rep_mean_abs_did)


                rep_accuracy = np.mean(test_set_accuracies)
                rep_accuracies.append(rep_accuracy)

            # if i == 1:
            print(baseline["model_name"], rep_accuracies)

            baseline_mean_abs_dic = float(np.mean(rep_mean_abs_dics))
            baseline_mean_abs_dic_std = float(np.std(rep_mean_abs_dics))

            baseline_mean_abs_did = float(np.mean(rep_mean_abs_dids))
            baseline_mean_abs_did_std = float(np.std(rep_mean_abs_dids))


            baseline_accuracy = float(np.mean(rep_accuracies))
            baseline_accuracy_std = float(np.std(rep_accuracies))
        

        
                    # baseline_accuracy = np.mean(rep_accuracies)
                    # #  baseline_variance = np.std(rep_accuracies)
                    # baseline_accuracies.append(baseline_accuracy)


                    # results[test_set_str][result_key].append(
                    #     (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



            # overall_baseline_accuracy = np.mean(baseline_accuracies)



            results[result_key].append(
                (baseline["model_label"], 
                baseline_accuracy, #overall_baseline_accuracy,
                baseline_accuracy_std, 
                baseline_mean_abs_dic,
                baseline_mean_abs_dic_std,
                baseline_mean_abs_did,
                baseline_mean_abs_did_std
                ))
                
                #np.min(baseline_accuracies),
                #np.max(baseline_accuracies)))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results_path = os.path.join(out_dir, "results.json")
    json_io.save_json(results_path, results)

    # return results





def get_size_results(test_sets, training_set_sizes, single_image_set_model_str, out_dir):

    mappings = {}
    results = {
        "set_of_27": [],
        single_image_set_model_str: []
    }
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": []
        # }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)



    model_names = []
    for training_set_size in training_set_sizes:
        model_names.append("set_of_27_" + str(training_set_size) + "_patches")
    for training_set_size in training_set_sizes:
        model_names.append(single_image_set_model_str + str(training_set_size) + "_patches")

    # for i in range(2):
    #     if i == 0:
    #         baselines = single_baselines
    #         result_key = "single"
    #     else:
    #         baselines = diverse_baselines
    #         result_key = "diverse"

    #     for baseline in baselines:
    # for model_name in model_names:

    for i in range(2):
        if i == 0:
            result_key = "set_of_27"
        else:
            result_key = single_image_set_model_str
        for training_set_size in training_set_sizes:
            rep_accuracies = []
            rep_mean_abs_dics = []
            for rep_num in range(5):

                model_name = result_key + "_" + str(training_set_size) + "_patches_rep_" + str(rep_num)
                # baseline["model_name"] + "_rep_" + str(rep_num)
                
                test_set_accuracies = []
                abs_dics = []
                for test_set in test_sets:
                    print(test_set)
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                    # rep_accuracies = []
                    # for rep_num in range(3):
                    # model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    # excel_path = os.path.join(result_dir, "metrics.xlsx")
                    # df = pd.read_excel(excel_path, sheet_name=0)
                    # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                    # rep_accuracies.append(rep_accuracy)


                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)


                            anno_count = annotations[image_name]["boxes"].shape[0]
                            pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]
                            abs_dics.append(abs(anno_count - pred_count))

                    

                    test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images)
                    test_set_accuracies.append(test_set_accuracy)



                rep_mean_abs_dic = np.mean(abs_dics)
                rep_mean_abs_dics.append(rep_mean_abs_dic)


                rep_accuracy = np.mean(test_set_accuracies)
                rep_accuracies.append(rep_accuracy)

            # if i == 1:
            # print(baseline["model_name"], rep_accuracies)

            baseline_mean_abs_dic = float(np.mean(rep_mean_abs_dics))
            baseline_mean_abs_dic_std = float(np.std(rep_mean_abs_dics))

            baseline_accuracy = float(np.mean(rep_accuracies))
            baseline_accuracy_std = float(np.std(rep_accuracies))
        

        
                    # baseline_accuracy = np.mean(rep_accuracies)
                    # #  baseline_variance = np.std(rep_accuracies)
                    # baseline_accuracies.append(baseline_accuracy)


                    # results[test_set_str][result_key].append(
                    #     (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



            # overall_baseline_accuracy = np.mean(baseline_accuracies)


            results[result_key].append(
                (
                # baseline["model_label"], 
                training_set_size,
                baseline_accuracy, #overall_baseline_accuracy,
                baseline_accuracy_std, 
                baseline_mean_abs_dic,
                baseline_mean_abs_dic_std
                )
            )
                
                #np.min(baseline_accuracies),
                #np.max(baseline_accuracies)))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results_path = os.path.join(out_dir, "size_results.json")
    json_io.save_json(results_path, results)











def p(out_dir):

    results_path = os.path.join(out_dir, "size_results.json")
    results = json_io.load_json(results_path)

    # print("set_of_27")
    for y, z in zip(results["set_of_27"], results["row_spacing_brown_2021-06-01"]):
        print(z[0])
        print("row_spacing_brown_2021-06-01", z[1])
        print("set_of_27", y[1])
        print("diff", y[1] - z[1])


    # print("set_of_27")
    # for z in results["set_of_27"]:
    #     print(z[0])




def create_training_set_size_plot(out_dir):

    results_path = os.path.join(out_dir, "size_results.json")
    results = json_io.load_json(results_path)

    for k in results.keys():
        if k != "set_of_27":
            single_image_set_model_str = k
            break

    plot_colors = {
        "set_of_27": my_plot_colors[0],
        single_image_set_model_str: my_plot_colors[1]
    }
    labels = {
        "set_of_27": "Diverse",
        single_image_set_model_str: single_image_set_model_str
    }

    # fig = plt.figure(figsize=(10,10))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    

    # single_tuples.sort(key=lambda x: x[1], reverse=False)
    # diverse_tuples.sort(key=lambda x: x[1], reverse=False)

    for k in results.keys():
        axs[0].plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=plot_colors[k], label=labels[k])

        axs[0].scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=plot_colors[k])
        axs[0].fill_between([x[0] for x in results[k]],
                            [x[1] - x[2] for x in results[k]],
                            [x[1] + x[2] for x in results[k]],
                            edgecolor=plot_colors[k], 
                            facecolor=plot_colors[k], 
                            alpha=0.15)
        



    # axs[0].scatter(np.arange(len(single_tuples)), [x[1] for x in single_tuples], color=my_plot_colors[1], marker="_", zorder=2, label="Single Image Set Model")
    # axs[0].scatter(np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), [x[1] for x in diverse_tuples], color=my_plot_colors[0], marker="_", zorder=2, label="Diverse Model")
    # for i, x in enumerate(single_tuples):
    #     axs[0].plot([i, i], [x[1] - x[2], x[1] + x[2]], color=my_plot_colors[1])

    # for i, x in enumerate(diverse_tuples):
    #     axs[0].plot([len(single_tuples)+i, len(single_tuples)+i], [x[1] - x[2], x[1] + x[2]], color=my_plot_colors[0])

    axs[0].set_ylabel("Instance-Based Accuracy")

    # axs[0].set_ylim(bottom=0, top=1)
    # axs[0].set_xticks([])

    # single_tuples.sort(key=lambda x: x[3], reverse=True)
    # diverse_tuples.sort(key=lambda x: x[3], reverse=True)



    for k in results.keys():
        axs[1].plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=plot_colors[k], label=labels[k])
        axs[1].scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=plot_colors[k])
        axs[1].fill_between([x[0] for x in results[k]],
                            [x[3] - x[4] for x in results[k]],
                            [x[3] + x[4] for x in results[k]],
                            edgecolor=plot_colors[k], 
                            facecolor=plot_colors[k], 
                            alpha=0.15)
        

    # axs[1].scatter(np.arange(len(single_tuples)), [x[3] for x in single_tuples], color=my_plot_colors[1], marker="_", zorder=2, label="Single Image Set Model")
    # axs[1].scatter(np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), [x[3] for x in diverse_tuples], color=my_plot_colors[0], marker="_", zorder=2, label="Diverse Model")
    # for i, x in enumerate(single_tuples):
    #     axs[1].plot([i, i], [x[3] - x[4], x[3] + x[4]], color=my_plot_colors[1])

    # for i, x in enumerate(diverse_tuples):
    #     axs[1].plot([len(single_tuples)+i, len(single_tuples)+i], [x[3] - x[4], x[3] + x[4]], color=my_plot_colors[0])


    axs[1].set_ylabel("Mean Absolute Difference in Count")
    axs[1].set_ylim(bottom=0)
    # axs[1].set_xticks([])


    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=11)

    fig.suptitle("Effect of Training Set Size and Diversity on Test Performance")
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2, top=0.86)

    out_path = os.path.join(out_dir, "combined_diversity_size_plot.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)










def create_accuracy_plot(out_dir):

    results_path = os.path.join(out_dir, "results.json")
    results = json_io.load_json(results_path)


    # for test_set_str in results:
    single_tuples = results["single"]
    diverse_tuples = results["diverse"]

    single_tuples.sort(key=lambda x: x[1])
    diverse_tuples.sort(key=lambda x: x[1])

    labels = []
    for single_tuple in single_tuples:
        labels.append(single_tuple[0])
    for diverse_tuple in diverse_tuples:
        labels.append(diverse_tuple[0])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.32, 0.05, 0.66, 0.9])

    ax.scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2)
    ax.scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2)
    for i, x in enumerate(single_tuples):
        ax.plot([x[1] - x[2], x[1] + x[2]], [i, i], color=my_plot_colors[1])

    for i, x in enumerate(diverse_tuples):
        ax.plot([x[1] - x[2], x[1] + x[2]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Instance-Based Accuracy")
    ax.set_title("Effect of Training Set Diversity on Accuracy")

    out_path = os.path.join(out_dir, "instance_based_accuracy.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)



def create_mean_abs_dic_plot(out_dir):

    results_path = os.path.join(out_dir, "results.json")
    results = json_io.load_json(results_path)


    # for test_set_str in results:
    single_tuples = results["single"]
    diverse_tuples = results["diverse"]

    single_tuples.sort(key=lambda x: x[3], reverse=True)
    diverse_tuples.sort(key=lambda x: x[3], reverse=True)

    labels = []
    for single_tuple in single_tuples:
        labels.append(single_tuple[0])
    for diverse_tuple in diverse_tuples:
        labels.append(diverse_tuple[0])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.32, 0.05, 0.66, 0.9])

    ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2)
    ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2)
    for i, x in enumerate(single_tuples):
        ax.plot([x[3] - x[4], x[3] + x[4]], [i, i], color=my_plot_colors[1])

    for i, x in enumerate(diverse_tuples):
        ax.plot([x[3] - x[4], x[3] + x[4]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Mean Absolute Difference in Count")
    ax.set_title("Effect of Training Set Diversity on Predicted Count")

    ax.set_xlim(left=0)

    out_path = os.path.join(out_dir, "mean_abs_dic.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)



def create_combined_plot(out_dir):


    results_path = os.path.join(out_dir, "results.json")
    results = json_io.load_json(results_path)

    single_tuples = results["single"]
    diverse_tuples = results["diverse"]


    labels = []
    for single_tuple in single_tuples:
        labels.append(single_tuple[0])
    for diverse_tuple in diverse_tuples:
        labels.append(diverse_tuple[0])

    # fig = plt.figure(figsize=(10,10))
    # fig, axs = plt.subplots(1, 3, figsize=(18, 8)) #12, 8))

    fig, axs = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={"width_ratios": [2, 2, 0.4]}) #3]})
    
    single_tuples.sort(key=lambda x: x[1], reverse=False)
    diverse_tuples.sort(key=lambda x: x[1], reverse=False)


    axs[0].scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2, label="Single Image Set Model")
    axs[0].scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2, label="Diverse Model")
    for i, x in enumerate(single_tuples):
        axs[0].plot([x[1] - x[2], x[1] + x[2]], [i, i], color=my_plot_colors[1])

    for i, x in enumerate(diverse_tuples):
        axs[0].plot([x[1] - x[2], x[1] + x[2]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    # axs[0].set_xlabel("Instance-Based Accuracy")
    axs[0].set_title("Instance-Based Accuracy", pad=30)
    # axs[1].set_title("Effect of Training Set Diversity on Predicted Count")
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_xlim(left=0, right=1)
    # xticks = [x[0] for x in single_tuples] + [x[0] for x in diverse_tuples]
    all_els = single_tuples + diverse_tuples
    # axs[0].set_xticks(np.arange(len(xticks))) #[])
    # axs[0].set_xticklabels(xticks, rotation=90)
    for i, x in enumerate(all_els):
        if x[0] == "Diverse":
            continue
        axs[0].annotate(str(test_set_str_to_number[x[0]]), (x[1] + x[2] + 0.03, i), va="center") #, rotation=90)

    # axs[0].xticks(rotation=90)
    axs[0].set_yticks([])

    single_tuples.sort(key=lambda x: x[3], reverse=True)
    diverse_tuples.sort(key=lambda x: x[3], reverse=True)


    axs[1].scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2, label="Single Image Set Model")
    axs[1].scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2, label="Diverse Model")
    for i, x in enumerate(single_tuples):
        axs[1].plot([x[3] - x[4], x[3] + x[4]], [i, i], color=my_plot_colors[1])

    for i, x in enumerate(diverse_tuples):
        axs[1].plot([x[3] - x[4], x[3] + x[4]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    # axs[1].set_xlabel("Mean Absolute Difference in Count")
    axs[1].set_title("Mean Absolute Difference in Count", pad=30)
    # axs[1].set_title("Effect of Training Set Diversity on Predicted Count")
    axs[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[1].set_xlim(left=0, right=185) #, right=1)
    # xticks = [x[0] for x in single_tuples] + [x[0] for x in diverse_tuples]
    all_els = single_tuples + diverse_tuples
    # axs[0].set_xticks(np.arange(len(xticks))) #[])
    # axs[0].set_xticklabels(xticks, rotation=90)
    for i, x in enumerate(all_els):
        if x[0] == "Diverse":
            continue
        axs[1].annotate(str(test_set_str_to_number[x[0]]), (x[3] + x[4] + 5, i), va="center") #, rotation=90)

    # axs[0].xticks(rotation=90)
    axs[1].set_yticks([])


    textstr = ""
    
    z = {}
    for k in test_set_str_to_number:
        z[test_set_str_to_number[k]] = k

    for i in range(len(list(test_set_str_to_number.keys()))):
        if len(str(i+1)) == 2:
            pad = "....."
        else:
            pad = "......"
        textstr += str(i+1) + pad + z[i+1] + "\n"


    axs[2].text(0.12, 0.97, textstr, transform=axs[2].transAxes, fontsize=10, family="monospace",
        verticalalignment='top', linespacing=1.8) #, bbox=props)

    axs[2].axis("off")






    # axs[1].scatter(np.arange(len(single_tuples)), [x[3] for x in single_tuples], color=my_plot_colors[1], marker="_", zorder=2, label="Single Image Set Model")
    # axs[1].scatter(np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), [x[3] for x in diverse_tuples], color=my_plot_colors[0], marker="_", zorder=2, label="Diverse Model")
    # for i, x in enumerate(single_tuples):
    #     axs[1].plot([i, i], [x[3] - x[4], x[3] + x[4]], color=my_plot_colors[1])

    # for i, x in enumerate(diverse_tuples):
    #     axs[1].plot([len(single_tuples)+i, len(single_tuples)+i], [x[3] - x[4], x[3] + x[4]], color=my_plot_colors[0])




    # # ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2)
    # # ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2)
    # # for i, x in enumerate(single_tuples):
    # #     ax.plot([x[3] - x[4], x[3] + x[4]], [i, i], color=my_plot_colors[1])

    # # for i, x in enumerate(diverse_tuples):
    # #     ax.plot([x[3] - x[4], x[3] + x[4]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    # # ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
    # # ax.set_yticklabels(labels)



    # axs[1].set_ylabel("Mean Absolute Difference in Count")
    # # axs[1].set_title("Effect of Training Set Diversity on Predicted Count")

    # axs[1].set_ylim(bottom=0)
    # axs[1].set_xticks([])


    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.778, 0.88), fontsize=12) #loc='upper right', fontsize=11)

    fig.suptitle("Effect of Training Set Diversity on Test Performance", fontsize=16)
    plt.tight_layout()

    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1,
    #                 right=0.9,
    #                 top=0.9,
    #                 wspace=0.4,
    #                 hspace=0.4)

    plt.subplots_adjust(top=0.84) #86) #wspace=0.2, top=0.86) #left=0.05, right=0.95, wspace=0.4, top=0.86) 

    out_path = os.path.join(out_dir, "combined_diversity_plot.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)






def create_combined_plot_less_wide(out_dir):


    results_path = os.path.join(out_dir, "results.json")
    results = json_io.load_json(results_path)

    single_tuples = results["single"]
    diverse_tuples = results["diverse"]


    labels = []
    for single_tuple in single_tuples:
        labels.append(single_tuple[0])
    for diverse_tuple in diverse_tuples:
        labels.append(diverse_tuple[0])

    # fig = plt.figure(figsize=(10,10))
    # fig, axs = plt.subplots(1, 3, figsize=(18, 8)) #12, 8))

    fig, axs = plt.subplots(1, 2, figsize=(14, 12)) #8))#, gridspec_kw={"width_ratios": [2, 2, 0.4]}) #3]})
    
    single_tuples.sort(key=lambda x: x[1], reverse=False)
    diverse_tuples.sort(key=lambda x: x[1], reverse=False)


    axs[0].scatter([x[1] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2, label="Single Image Set Model")
    axs[0].scatter([x[1] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2, label="Diverse Model")
    for i, x in enumerate(single_tuples):
        axs[0].plot([x[1] - x[2], x[1] + x[2]], [i, i], color=my_plot_colors[1])

    for i, x in enumerate(diverse_tuples):
        axs[0].plot([x[1] - x[2], x[1] + x[2]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    # axs[0].set_xlabel("Instance-Based Accuracy")
    axs[0].set_title("Instance-Based Accuracy", pad=30)
    # axs[1].set_title("Effect of Training Set Diversity on Predicted Count")
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_xlim(left=0, right=1)
    # xticks = [x[0] for x in single_tuples] + [x[0] for x in diverse_tuples]
    all_els = single_tuples + diverse_tuples
    # axs[0].set_xticks(np.arange(len(xticks))) #[])
    # axs[0].set_xticklabels(xticks, rotation=90)
    for i, x in enumerate(all_els):
        if x[0] == "Diverse":
            continue
        if test_set_str_to_number[x[0]] == 27:
            char = "\u0394"
        else:
            char = chr(64+test_set_str_to_number[x[0]])
        axs[0].annotate(char, (x[1] + x[2] + 0.03, i), va="center") #, rotation=90)

    # axs[0].xticks(rotation=90)
    axs[0].set_yticks([])

    single_tuples.sort(key=lambda x: x[5], reverse=True)
    diverse_tuples.sort(key=lambda x: x[5], reverse=True)


    axs[1].scatter([x[5] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2, label="Single Image Set Model")
    axs[1].scatter([x[5] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2, label="Diverse Model")
    for i, x in enumerate(single_tuples):
        axs[1].plot([x[5] - x[6], x[5] + x[6]], [i, i], color=my_plot_colors[1])

    for i, x in enumerate(diverse_tuples):
        axs[1].plot([x[5] - x[6], x[5] + x[6]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    # axs[1].set_xlabel("Mean Absolute Difference in Count")
    # axs[1].set_title("Mean Absolute Difference in Count", pad=30)
    axs[1].set_title("Mean Absolute Difference in Density", pad=30)
    # axs[1].set_title("Effect of Training Set Diversity on Predicted Count")
    axs[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ### add ###
    # axs[1].set_xlim(left=0, right=185) #, right=1)
    ### ###


    # xticks = [x[0] for x in single_tuples] + [x[0] for x in diverse_tuples]
    all_els = single_tuples + diverse_tuples
    # axs[0].set_xticks(np.arange(len(xticks))) #[])
    # axs[0].set_xticklabels(xticks, rotation=90)
    for i, x in enumerate(all_els):
        if x[0] == "Diverse":
            continue
        if test_set_str_to_number[x[0]] == 27:
            char = "\u0394"
        else:
            char = chr(64+test_set_str_to_number[x[0]])
        axs[1].annotate(char, (x[5] + x[6] + 1.4, i), va="center")
        # axs[1].annotate(str(test_set_str_to_number[x[0]]), (x[3] + x[4] + 5, i), va="center") #, rotation=90)

    # axs[0].xticks(rotation=90)
    axs[1].set_yticks([])


    textstr = ""
    # textstr2 = ""
    
    z = {}
    for k in test_set_str_to_number:
        z[test_set_str_to_number[k]] = k

    for i in range(14): #range(len(list(test_set_str_to_number.keys()))):
        # if len(str(i+1)) == 2:
        #     pad = "....."
        # else:
        #     pad = "......"
        pad = "......"

        if i < 13:
            # if len(str(i+15)) == 2:
            #     pad2 = "....."
            # else:
            #     pad2 = "......"
            pad2 = "......"

            # part2 = str(i+15) + pad2 + z[i+15]
            if i == 12:
                char = "\u0394"
            else:
                char = chr(65+i+14)
            part2 = char + pad2 + z[i+15]
            spacing = " " * (50 - len(z[i+1]))
        else:
            part2 = ""
            spacing = ""



        # textstr += str(i+1) + pad +z[i+1] + spacing + part2
        textstr += chr(65+i) + pad +z[i+1] + spacing + part2
        if i < 13:
            textstr += "\n"

    # for i in range(14, 27): #range(len(list(test_set_str_to_number.keys()))):
    #     if len(str(i+1)) == 2:
    #         pad = "....."
    #     else:
    #         pad = "......"
    #     textstr2 += str(i+1) + pad + z[i+1] + "\n"

    print(textstr)

    props = dict(boxstyle='round,pad=0.8', facecolor='white') #, alpha=0.5)
    # axs[0].text(0, 0, textstr, transform=axs[0].transAxes, fontsize=14,
    #     verticalalignment='top', bbox=props)
    

    plt.gcf().text(0.20, 0.28, textstr,  fontsize=10, family="monospace", 
                   linespacing=1.8, verticalalignment='top', bbox=props)


    # axs[1, 0].text(0.32, 0.97, textstr, transform=axs[1, 0].transAxes, fontsize=10, family="monospace",
    #     verticalalignment='top', linespacing=1.8) #, bbox=props)

    # axs[1, 0].axis("off")

    # axs[1, 1].text(0.32, 0.97, textstr2, transform=axs[1, 1].transAxes, fontsize=10, family="monospace",
    #     verticalalignment='top', linespacing=1.8) #, bbox=props)

    # axs[1, 1].axis("off")






    # axs[1].scatter(np.arange(len(single_tuples)), [x[3] for x in single_tuples], color=my_plot_colors[1], marker="_", zorder=2, label="Single Image Set Model")
    # axs[1].scatter(np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), [x[3] for x in diverse_tuples], color=my_plot_colors[0], marker="_", zorder=2, label="Diverse Model")
    # for i, x in enumerate(single_tuples):
    #     axs[1].plot([i, i], [x[3] - x[4], x[3] + x[4]], color=my_plot_colors[1])

    # for i, x in enumerate(diverse_tuples):
    #     axs[1].plot([len(single_tuples)+i, len(single_tuples)+i], [x[3] - x[4], x[3] + x[4]], color=my_plot_colors[0])




    # # ax.scatter([x[3] for x in single_tuples], np.arange(len(single_tuples)), color=my_plot_colors[1], marker="|", zorder=2)
    # # ax.scatter([x[3] for x in diverse_tuples], np.arange(len(single_tuples), len(single_tuples)+len(diverse_tuples)), color=my_plot_colors[0], marker="|", zorder=2)
    # # for i, x in enumerate(single_tuples):
    # #     ax.plot([x[3] - x[4], x[3] + x[4]], [i, i], color=my_plot_colors[1])

    # # for i, x in enumerate(diverse_tuples):
    # #     ax.plot([x[3] - x[4], x[3] + x[4]], [len(single_tuples)+i, len(single_tuples)+i], color=my_plot_colors[0])

    # # ax.set_yticks(np.arange(0, len(single_tuples)+len(diverse_tuples)))
    # # ax.set_yticklabels(labels)



    # axs[1].set_ylabel("Mean Absolute Difference in Count")
    # # axs[1].set_title("Effect of Training Set Diversity on Predicted Count")

    # axs[1].set_ylim(bottom=0)
    axs[1].set_xlim(left=0, right=46.5)
    # axs[1].set_xticks([])

    # legend_elements = [Line2D([0], [0], color=my_plot_colors[0], lw=10, label="Diverse Model"),
    #                 Line2D([0], [0], color=my_plot_colors[1], lw=10, label="Single Image Set Model")
    #                 ]
    legend_elements = [
    Patch(color=my_plot_colors[0], label="Diverse Model"),
    Patch(color=my_plot_colors[1], label="Single Image Set Model")
    ]

    # handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=legend_elements, handlelength=0.05, loc=(0.39, 0.88), fontsize=14)
            #    bbox_to_anchor=(1, 1), loc="upper center", fontsize=14) #loc=(0.25, 0.88), fontsize=14) #loc=(0.778, 0.88), fontsize=12) #loc='upper right', fontsize=11)
    # lgnd.legendHandles[0]._sizes = [100]
    # lgnd.legendHandles[1]._sizes = [100]


    fig.suptitle("Effect of Training Set Diversity on Test Performance", fontsize=16)
    plt.tight_layout()

    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1,
    #                 right=0.9,
    #                 top=0.9,
    #                 wspace=0.4,
    #                 hspace=0.4)

    plt.subplots_adjust(top=0.82, bottom=0.3) #, bottom=0.1) #86) #wspace=0.2, top=0.86) #left=0.05, right=0.95, wspace=0.4, top=0.86) 

    out_path = os.path.join(out_dir, "combined_diversity_plot_mean_mean_abs_did.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)




def get_3150_results(set_numbers, test_sets, out_dir):


    mappings = {}
    results = {}

    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": []
        # }
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    for set_number in set_numbers:
        results[set_number] = []
        rep_accuracies = []
        rep_mean_abs_dics = []
        for rep_num in range(20):

            model_name = "set_of_" + str(set_number) + "_3150_patches_rep_" + str(rep_num)
            
            test_set_accuracies = []
            abs_dics = []
            for test_set in test_sets:
                print(test_set)
                test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                # rep_accuracies = []
                # for rep_num in range(3):
                # model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                # excel_path = os.path.join(result_dir, "metrics.xlsx")
                # df = pd.read_excel(excel_path, sheet_name=0)
                # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                # rep_accuracies.append(rep_accuracy)


                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)
                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)
                assessment_images = []
                for image_name in annotations.keys():
                    if len(annotations[image_name]["test_regions"]) > 0:
                        assessment_images.append(image_name)


                        anno_count = annotations[image_name]["boxes"].shape[0]
                        pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]
                        abs_dics.append(abs(anno_count - pred_count))

                

                test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images)
                test_set_accuracies.append(test_set_accuracy)



            rep_mean_abs_dic = float(np.mean(abs_dics))
            rep_mean_abs_dics.append(rep_mean_abs_dic)


            rep_accuracy = float(np.mean(test_set_accuracies))
            rep_accuracies.append(rep_accuracy)

        # if i == 1:
        # print(baseline["model_name"], rep_accuracies)

        # baseline_mean_abs_dic = float(np.mean(rep_mean_abs_dics))
        # baseline_mean_abs_dic_std = float(np.std(rep_mean_abs_dics))

        # baseline_accuracy = float(np.mean(rep_accuracies))
        # baseline_accuracy_std = float(np.std(rep_accuracies))



                # baseline_accuracy = np.mean(rep_accuracies)
                # #  baseline_variance = np.std(rep_accuracies)
                # baseline_accuracies.append(baseline_accuracy)


                # results[test_set_str][result_key].append(
                #     (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



        # overall_baseline_accuracy = np.mean(baseline_accuracies)

        results[set_number] = [
            rep_accuracies,
            rep_mean_abs_dics
        ]

        # results[set_number].append(
        #     # (
        #     rep_accuracies,
        #     rep_mean_abs_dics
        #     # baseline["model_label"], 
        #     # baseline_accuracy, #overall_baseline_accuracy,
        #     # baseline_accuracy_std, 
        #     # baseline_mean_abs_dic,
        #     # baseline_mean_abs_dic_std
        #     # )
        # )
            
            #np.min(baseline_accuracies),
            #np.max(baseline_accuracies)))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results_path = os.path.join(out_dir, "results_3150.json")
    json_io.save_json(results_path, results)


def create_3150_plot(out_dir):

    results_path = os.path.join(out_dir, "results_3150.json")
    results = json_io.load_json(results_path)
    json_io.print_json(results)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4)) #14, 12))

    set_nums = sorted([int(x) for x in results.keys()])

    for set_num in set_nums:
        print("{}: {} {} {} {}".format(set_num, 
                                 np.std(results[str(set_num)][0]),
                                 np.mean(results[str(set_num)][0]), 
                                 np.mean(results[str(set_num)][1]),
                                 np.std(results[str(set_num)][1])))

    for set_num in results.keys():
        axs[0].scatter([int(set_num)] * len(results[set_num][1]), results[set_num][0], 
                       color=my_plot_colors[0], marker="o", zorder=2, alpha=0.4) #, label="Single Image Set Model")

    axs[0].plot(set_nums, [np.mean(results[x][0]) for x in results.keys()], 
                   color=my_plot_colors[0], zorder=1)
    
    axs[0].set_ylabel("Instance-Based Accuracy") #, pad=30)
    axs[0].set_xlabel("Number of Sampled Training Image Sets")
    axs[0].set_ylim(bottom=0, top=1)
    axs[0].set_xticks(set_nums)


    for set_num in results.keys():
        axs[1].scatter([int(set_num)] * len(results[set_num][1]), results[set_num][1], 
                       color=my_plot_colors[0], marker="o", zorder=2, alpha=0.4) #, label="Single Image Set Model")
        
    axs[1].plot(set_nums, [np.mean(results[x][1]) for x in results.keys()], 
                   color=my_plot_colors[0], zorder=1)
    
    axs[1].set_ylabel("Mean Absolute Difference In Count")
    axs[1].set_xlabel("Number of Sampled Training Image Sets")
    axs[1].set_ylim(bottom=0)
    axs[1].set_xticks(set_nums)

    plt.suptitle("Effect of Training Set Diversity on Test Performance: Varying the Number of Sampled Image Sets")

    plt.tight_layout()

    out_path = os.path.join(out_dir, "effect_of_training_sets.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)



def runner_630():


    single_baselines = []
    for baseline in eval_single_630_baselines:

        b_name = baseline[:len(baseline)-len("_630_patches")]
        pieces = b_name.split("_")
        label = "_".join(pieces[:-2]) + "/" + pieces[-2] + "/" + pieces[-1]


        single_baselines.append({
            "model_name": baseline,
            "model_creator": "eval",
            "model_label": label
        })

    diverse_baselines = []
    for baseline in eval_diverse_630_baselines: #["set_of_27_1000_patches"]: #eval_diverse_630_baselines:
        diverse_baselines.append({
            "model_name": baseline,
            "model_creator": "eval",
            "model_label": "Diverse"
        })        
    out_dir = os.path.join("eval_charts", "diversity", "630_patch_models")

    # get_diversity_results(eval_test_sets, single_baselines, diverse_baselines, out_dir)
    # create_accuracy_plot(out_dir)
    # create_mean_abs_dic_plot(out_dir)
    create_combined_plot_less_wide(out_dir)

    # training_set_sizes = [250, 630, 1000, 2000, 3906]
    # get_size_results(eval_test_sets, training_set_sizes, "row_spacing_brown_2021-06-01", out_dir)
    # create_training_set_size_plot(out_dir)
    # p(out_dir)




def runner_3150():
    set_numbers = [5, 10, 15, 20, 25]
    out_dir = os.path.join("eval_charts", "diversity", "3150_patch_models")
    get_3150_results(set_numbers, eval_test_sets, out_dir)
    create_3150_plot(out_dir)



if __name__ == "__main__":
    runner_630()
    # runner_3150()