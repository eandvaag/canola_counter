import argparse
import os

import run_settings
import dataset
import extract_patches as ep
import model_interface
import model_vis
import model_eval

from io_utils import json_io

import logging





def main(config):

    logging.basicConfig(level=logging.INFO)

    settings = run_settings.Settings(config)


    train_dataset = dataset.Dataset(settings.train_img_paths, "train")
    val_dataset = dataset.Dataset(settings.val_img_paths, "val")
    test_dataset = dataset.Dataset(settings.test_img_paths, "test")


    #extractor = ep.BoxExtractor()
    #extractor = ep.TileExtractor()
    extractor = ep.JitterBoxExtractor()

    train_patch_dir = extractor.extract(train_dataset, settings, annotate_patches=True)
    val_patch_dir = extractor.extract(val_dataset, settings, annotate_patches=True)

    #model = model_interface.RetinaNet(settings, 'exothermic-screen')
    #model = model_interface.CenterNet(settings, 'relaxed-canyon')

    model = model_interface.CenterNet(settings)

    model.train(train_patch_dir, val_patch_dir)
    #model = model_interface.CenterNet(settings)#, 'relaxed-canyon')
    #model_vis.output_patch_predictions(model, train_patch_dir, settings)
    #model_vis.output_img_predictions(model, train_patch_dir, settings)

    #model_vis.output_patch_predictions(model, val_patch_dir, settings)
    #model_vis.output_img_predictions(model, val_patch_dir, settings)

    #model_vis.output_loss_plot(model)
    #model_eval.evaluate_patches(train_pred_dir, settings)

    #retinanet = model_interface.RetinaNet(settings, 'exothermic-screen')
    #centernet = model_interface.CenterNet(settings, 'relaxed-canyon')
    #model_vis.bar_chart_img_results(
    #    "/home/eaa299/Documents/work/2021/my_model_comparison_chart_tiles.html",
    #    [retinanet, centernet], [train_patch_dir, train_patch_dir], settings)



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="A tool for detecting plants in UAV images")
    parser.add_argument('input', type=str, help="")
    args = parser.parse_args()


    config = json_io.load_json(args.input)

    main(config)