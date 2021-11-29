import os

from io_utils import json_io

def add_entry_to_inference_record(entry):

    farm_name = entry["farm_name"]
    field_name = entry["field_name"]
    mission_date = entry["mission_date"]
    dataset_name = entry["dataset_name"]
    model_uuid = entry["model_uuid"]
    value = entry["value"]

    usr_data_root = os.path.join("usr", "data")
    inference_lookup_path = os.path.join(usr_data_root, "records", "inference_lookup.json")
    inference_lookup = json_io.load_json(inference_lookup_path)

    d = inference_lookup["inference_runs"]
    if farm_name not in d:
        d[farm_name] = {}
    
    d = d[farm_name]
    if field_name not in d:
        d[field_name] = {}

    d = d[field_name]
    if mission_date not in d:
        d[mission_date] = {}

    d = d[mission_date]
    if dataset_name not in d:
        d[dataset_name] = {}

    d = d[dataset_name]
    d[model_uuid] = value

    json_io.save_json(inference_lookup_path, inference_lookup)


def remove_entries_from_inference_record(entries):

    usr_data_root = os.path.join("usr", "data")
    inference_lookup_path = os.path.join(usr_data_root, "records", "inference_lookup.json")
    inference_lookup = json_io.load_json(inference_lookup_path)

    for entry in entries:
        farm_name = entry["farm_name"]
        field_name = entry["field_name"]
        mission_date = entry["mission_date"]
        dataset_name = entry["dataset_name"]
        model_uuid = entry["model_uuid"]


        try:
            inference_entries = inference_lookup["inference_runs"][farm_name][field_name][mission_date][dataset_name]
        except KeyError:
            continue

        if model_uuid in inference_entries:
            del inference_entries[model_uuid]

    json_io.save_json(inference_lookup_path, inference_lookup)