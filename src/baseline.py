import logging
import os

from io_utils import json_io
import image_set_model as ism

def resume_baseline_request(baseline_name):

    logging.basicConfig(level=logging.INFO)
    ism.handle_resume_direct_baseline_request(baseline_name)


def run_baseline_request():

    logging.basicConfig(level=logging.INFO)

    request = {
        # "baseline_name": "UNI::LowN1::2021-06-07::" +
        #                  "BlaineLake::River::2021-06-09::" +
        #                  "BlaineLake::HornerWest::2021-06-09::" +
        #                  "Saskatoon::Norheim1::2021-05-26::" +
        #                  "UNI::Sutherland::2021-06-05::" +
        #                  "UNI::Brown::2021-06-05::" + 
        #                  "row_spacing::brown_low_res::2021-06-01::" +
        #                  "row_spacing::nasser_low_res::2020-06-08::" +
        #                  "Saskatoon::Norheim2::2021-05-26::" +
        #                  "Saskatoon::Norheim4::2022-05-24",
        "baseline_name": "all_test",
        #"baseline_name": "UNI::Sutherland::2021-06-05",
        
        "image_sets": [
        {
            "farm_name": "UNI",
            "field_name": "LowN1",
            "mission_date": "2021-06-07"
        },
        {
            "farm_name": "BlaineLake",
            "field_name": "River",
            "mission_date": "2021-06-09"
        },
        {
            "farm_name": "BlaineLake",
            "field_name": "HornerWest",
            "mission_date": "2021-06-09"
        },
        {
            "farm_name": "Saskatoon",
            "field_name": "Norheim1",
            "mission_date": "2021-05-26"
        },
        {
            "farm_name": "UNI",
            "field_name": "Sutherland",
            "mission_date": "2021-06-05"
        },
        {
            "farm_name": "UNI",
            "field_name": "Brown",
            "mission_date": "2021-06-05"
        },
        {
            "farm_name": "row_spacing",
            "field_name": "brown_low_res",
            "mission_date": "2021-06-01"
        },
        {
            "farm_name": "row_spacing",
            "field_name": "nasser_low_res",
            "mission_date": "2020-06-08"
        },
        {
            "farm_name": "Saskatoon",
            "field_name": "Norheim2",
            "mission_date": "2021-05-26"
        },
        {
            "farm_name": "Saskatoon",
            "field_name": "Norheim4",
            "mission_date": "2022-05-24"
        },
        ]
    }

    for image_set in request["image_sets"]:
        annotations_path = os.path.join("usr", "data", "image_sets",
                           image_set["farm_name"], image_set["field_name"], image_set["mission_date"],
                           "annotations", "annotations_w3c.json")
        annotations = json_io.load_json(annotations_path)
        image_set["images"] = [image_name for image_name in annotations.keys() if annotations[image_name]["status"] == "completed_for_training"]


    ism.handle_direct_baseline_request(request)

if __name__ == "__main__":
    run_baseline_request()