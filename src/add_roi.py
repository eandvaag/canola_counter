import glob
import os
import shutil
import zipfile

from models.common import annotation_utils
from io_utils import json_io

from models.common import inference_metrics

def add_roi():

    for username_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(username_path)

        if username == "erik":
            overlay_appearance_path = os.path.join(username_path, "overlay_appearance.json")
            overlay_appearance = json_io.load_json(overlay_appearance_path)
            overlay_appearance["draw_order"] = ["region_of_interest", "training_region", "test_region", "annotation", "prediction"]
            overlay_appearance["style"]["region_of_interest"] = "strokeRect"
            overlay_appearance["colors"]["region_of_interest"] = "#07eb4b"
            json_io.save_json(overlay_appearance_path, overlay_appearance)



            for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
                farm_name = os.path.basename(farm_path)
                for field_path in glob.glob(os.path.join(farm_path, "*")):
                    field_name = os.path.basename(field_path)
                    for mission_path in glob.glob(os.path.join(field_path, "*")):
                        print(mission_path)
                        mission_date = os.path.basename(mission_path)

                        tags_path = os.path.join(mission_path, "annotations", "tags.json")
                        if os.path.exists(os.path.join(mission_path, "annotations")):
                            # print("adding tags to", tags_path)
                            json_io.save_json(tags_path, {})

                        annotations_path = os.path.join(mission_path, "annotations", "annotations.json")
                        if os.path.exists(annotations_path):
                            annotations = json_io.load_json(annotations_path)

                            for image_name in annotations.keys():
                                annotations[image_name]["regions_of_interest"] = []

                            json_io.save_json(annotations_path, annotations)

                        results_dir = os.path.join(mission_path, "model", "results")
                        if os.path.exists(results_dir):
                            for result_dir in glob.glob(os.path.join(results_dir, "*")):

                                annotations_path = os.path.join(result_dir, "annotations.json")
                                annotations = json_io.load_json(annotations_path)

                                for image_name in annotations.keys():
                                    annotations[image_name]["regions_of_interest"] = []

                                json_io.save_json(annotations_path, annotations)


                                tags_path = os.path.join(result_dir, "tags.json")
                                json_io.save_json(tags_path, {})


                                inference_metrics.create_areas_spreadsheet(result_dir)

                                raw_outputs_zip_path = os.path.join(result_dir, "raw_outputs.zip")
                                raw_outputs_dir = os.path.join(result_dir, "raw_outputs")

                                with zipfile.ZipFile(raw_outputs_zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(raw_outputs_dir)

                                annotations_path = os.path.join(raw_outputs_dir, "annotations.json")
                                annotations = json_io.load_json(annotations_path)
                                for image_name in annotations.keys():
                                    annotations[image_name]["regions_of_interest"] = []

                                json_io.save_json(annotations_path, annotations)

                                shutil.make_archive(raw_outputs_dir, 'zip', raw_outputs_dir)
                                shutil.rmtree(raw_outputs_dir)

