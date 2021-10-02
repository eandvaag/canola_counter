import numpy as np
import os
import shutil
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

import model_eval
from io_utils import json_io



def draw_boxes_on_img(img_array,
                      pred_boxes,
                      pred_classes,
                      pred_scores,
                      settings,
                      gt_boxes=None,
                      patch_coords=None,
                      display_class=True,
                      display_score=True):

    out_array = np.copy(img_array)

    if patch_coords is not None:
        for patch_coord in patch_coords:
            cv2.rectangle(out_array, (patch_coord[1], patch_coord[0]), (patch_coord[3], patch_coord[2]), (255, 0, 255), 1)

    if gt_boxes is not None:
        for gt_box in gt_boxes:
            cv2.rectangle(out_array, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (255, 0, 0), 1)

    for pred_box, pred_class, pred_score in zip(pred_boxes, pred_classes, pred_scores):

        cv2.rectangle(out_array, (max(pred_box[1], 0), max(pred_box[0], 0)),
                                 (min(pred_box[3], out_array.shape[1]), min(pred_box[2], out_array.shape[0])),
                                 (0, 255, 0), 1)

        if display_class or display_class:
            if display_class and display_score:
                label = settings.rev_class_map[pred_class] + ": " + str(round(pred_score, 2))
            elif display_class:
                label = pred_class
            elif display_score:
                label = str(round(pred_score, 2))

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_array, (pred_box[1], int(pred_box[0] - text_h)), (int(pred_box[1] + text_w), pred_box[0]), (0, 255, 0), -1)
            cv2.putText(out_array, label, (pred_box[1], pred_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out_array


def output_patch_predictions(model, patch_dir, settings):

    model.generate_predictions(patch_dir, skip_if_found=True)

    pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
    pred_data, is_annotated = json_io.read_patch_predictions(model, patch_dir, read_patches=True)
    out_dir = os.path.join(pred_dir, "patch_predictions")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    for pred in tqdm.tqdm(pred_data, desc="Outputting patch predictions"):

        gt_boxes = pred["patch_abs_boxes"] if is_annotated else None

        pred_patch = draw_boxes_on_img(pred["patch"], 
                                       pred["pred_patch_abs_boxes"],
                                       pred["pred_classes"], 
                                       pred["pred_scores"],
                                       settings,
                                       gt_boxes=gt_boxes,
                                       display_class=True,#False,
                                       display_score=True)#False)


        pred_patch_name = os.path.basename(pred["patch_path"])[:-4] + "_predictions.png"

        cv2.imwrite(os.path.join(out_dir, pred_patch_name), pred_patch)





def output_img_predictions(model, patch_dir, settings):

    model.generate_predictions(patch_dir, skip_if_found=True)

    pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
    pred_data, is_annotated = json_io.read_img_predictions(model, patch_dir, settings)
    out_dir = os.path.join(pred_dir, "image_predictions")
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    for img_path in tqdm.tqdm(pred_data.keys(), "Outputting image predictions"):

        img_array = cv2.imread(img_path)

        gt_boxes = pred_data[img_path]["img_abs_boxes"] if is_annotated else None

        pred_img = draw_boxes_on_img(img_array,
                                     pred_data[img_path]["nms_pred_img_abs_boxes"],
                                     pred_data[img_path]["nms_pred_classes"],
                                     pred_data[img_path]["nms_pred_scores"],
                                     settings,
                                     gt_boxes=gt_boxes,
                                     patch_coords=pred_data[img_path]["patch_coords"],
                                     display_class=False,
                                     display_score=False)
        pred_img_name = os.path.basename(img_path)[:-4] + "_predictions.png"

        cv2.imwrite(os.path.join(out_dir, pred_img_name), pred_img)






def output_patch_count_diff_histogram(model, patch_dir, settings):

    model_eval.evaluate(model, patch_dir, settings)

    pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
    pred_stats = json_io.load_json(os.path.join(pred_dir, "prediction_stats.json"))

    out_dir = os.path.join(pred_dir, "prediction_stats_vis")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, "patch_plant_count_diff_histogram.svg")

    patch_paths = pred_stats["patch_results"].keys()
    data = [pred_stats["patch_results"][patch_path]["pred_minus_actual"] for patch_path in patch_paths]

    plt.figure()
    plt.hist(data, color="palegreen")
    plt.title("Patch Plant Counts: Predicted Minus Actual")
    plt.xlabel("Count Difference")
    plt.ylabel("Number of patches")
    plt.savefig(save_path)
    plt.close()




def output_loss_plot(model):

    model_dir = model.config.model_dir
    loss_record_path = os.path.join(model_dir, "loss_record.json")
    save_path = os.path.join(model_dir, "loss_plot.svg")

    loss_record = json_io.load_json(loss_record_path)

    train_loss_values = [loss_record[epoch]["train_loss"] for epoch in loss_record.keys()]
    val_loss_values = [loss_record[epoch]["val_loss"] for epoch in loss_record.keys()]
    epochs = [int(epoch) for epoch in loss_record.keys()]

    plt.figure()
    plt.plot(epochs, train_loss_values, "steelblue", label="Training")
    plt.plot(epochs, val_loss_values, "tomato", label="Validation")
    plt.title(model.config.model_name + " ('" + model.config.instance_name + "') Loss Values")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()



def bar_chart_img_results(out_path, models, patch_dirs, settings):

    data = []

    for i, (model, patch_dir) in enumerate(zip(models, patch_dirs)):
        model_eval.evaluate(model, patch_dir, settings)

        pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
        pred_stats = json_io.load_json(os.path.join(pred_dir, "prediction_stats.json"))


        img_paths = sorted(pred_stats["image_results"].keys())
        img_names = [os.path.basename(img_path)[:-4] for img_path in img_paths]
        pred_counts = [pred_stats["image_results"][img_path]["num_pred"] for img_path in img_paths]


        if i == 0:
            gt_counts = [pred_stats["image_results"][img_path]["num_actual"] for img_path in img_paths]
            data.append(go.Bar(name="Ground Truth", x=img_names, y=gt_counts))

        data.append(go.Bar(name=model.config.model_name + " ('" + model.config.instance_name + "')", x=img_names, y=pred_counts))

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group',
        title="Ground Truth and Predicted Counts",
        xaxis_title="Image",
        yaxis_title="Count",
        font=dict(
            family="verdana",
            size=18,
            color='darkslategray'
        )
    )
    fig.write_html(out_path)
