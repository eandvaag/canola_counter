import json
import os
import numpy as np


from io_utils import xml_io


def save_json(path, data):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def load_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def print_json(data):
    print(json.dumps(data, indent=4, sort_keys=True))