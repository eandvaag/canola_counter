import argparse
import os

import run_settings
import dataset
import extract_patches as ep
import model_interface
import model_vis

from io_utils import json_io











def main(config):

	settings = run_settings.Settings(config)


	train_dataset = dataset.Dataset(settings.train_img_paths, "train")
	val_dataset = dataset.Dataset(settings.val_img_paths, "val")
	test_dataset = dataset.Dataset(settings.test_img_paths, "test")

	#extractor = ep.BoxExtractor()
	#train_tf_record_path = extractor.extract(train_dataset, settings, annotate_patches=True)
	#val_tf_record_path = extractor.extract(val_dataset, settings, annotate_patches=True)

	extractor = ep.TileExtractor()
	#extractor = ep.BoxExtractor()
	train_patches_dir = extractor.extract(train_dataset, settings, annotate_patches=True)
	val_patches_dir = extractor.extract(val_dataset, settings, annotate_patches=True)


	#model = model_interface.RetinaNet(settings)
	model = model_interface.RetinaNet(settings, model_uuid="6ba8a185-437a-45b0-a406-f258fde5d9f4")#, model_uuid="b3ede4f9-a5c8-49e2-bfe5-a133fdeb95e8")
	
	#model.train(train_patches_dir, val_patches_dir)
	#exit()
	train_pred_dir = model.generate_predictions(train_patches_dir, found_behaviour="skip")
	val_pred_dir = model.generate_predictions(val_patches_dir, found_behaviour="skip")

	#model_vis.output_patch_predictions(train_pred_dir)
	#model_vis.output_img_predictions(train_pred_dir, nms_iou_threshold=0.25)

	model_vis.output_patch_predictions(val_pred_dir)
	#model_vis.output_img_predictions(val_pred_dir, nms_iou_threshold=0.25)








if __name__ == "__main__":


	parser = argparse.ArgumentParser(description="A tool for detecting plants in UAV images")
	parser.add_argument('input', type=str, help="")
	args = parser.parse_args()


	config = json_io.load_json(args.input)

	main(config)