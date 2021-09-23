
import numpy as np
import os
import cv2

from io_utils import json_io


def draw_boxes_on_img(img_array,
				   	  pred_boxes,
				   	  pred_classes,
				   	  pred_scores,
				   	  score_threshold=0.5,
				   	  gt_boxes=None,
				   	  patch_coords=None):

	out_array = np.copy(img_array)

	print("img_array.shape", img_array.shape)
	#print("patch_coords", patch_coords.dtype)
	if patch_coords is not None:
		for patch_coord in patch_coords:
			#print("patch_coord", patch_coord)
			cv2.rectangle(out_array, (patch_coord[1], patch_coord[0]), (patch_coord[3], patch_coord[2]), (255, 0, 255), 1)

	#print("pred_boxes", pred_boxes.dtype)
	#print("pred_boxes.shape", pred_boxes.shape)
	for pred_box, pred_class, pred_score in zip(pred_boxes, pred_classes, pred_scores):
		if pred_score >= score_threshold:
			cv2.rectangle(out_array, (max(pred_box[1], 0), max(pred_box[0], 0)),
						   		 	 (min(pred_box[3], out_array.shape[1]), min(pred_box[2], out_array.shape[0])),
						   			 (0, 255, 0), 1)
	return out_array


def output_patch_predictions(pred_dir):

	pred_data = json_io.read_patch_predictions(pred_dir, read_patches=True)
	pred_patch_dir = os.path.join(pred_dir, "patch_predictions")
	os.makedirs(pred_patch_dir)

	for pred in pred_data:

		pred_patch = draw_boxes_on_img(pred["patch"], 
									   pred["pred_patch_abs_boxes"],
									   pred["pred_classes"], 
									   pred["pred_scores"])
		pred_patch_name = os.path.basename(pred["patch_path"])[:-4] + "_predictions.png"

		cv2.imwrite(os.path.join(pred_patch_dir, pred_patch_name), pred_patch)





def output_img_predictions(pred_dir, nms_iou_threshold):

	pred_data = json_io.read_img_predictions(pred_dir, nms_iou_threshold=nms_iou_threshold)
	pred_img_dir = os.path.join(pred_dir, "image_predictions")
	os.makedirs(pred_img_dir)


	for img_path in pred_data.keys():

		img_array = cv2.imread(img_path)

		pred_img = draw_boxes_on_img(img_array,
									 pred_data[img_path]["pred_boxes"],#["sel_boxes"],
				 	   			     pred_data[img_path]["pred_classes"],#["sel_classes"],
									 pred_data[img_path]["pred_scores"],#["sel_scores"])
									 patch_coords=pred_data[img_path]["patch_coords"])
		pred_img_name = os.path.basename(img_path)[:-4] + "_predictions.png"

		cv2.imwrite(os.path.join(pred_img_dir, pred_img_name), pred_img)