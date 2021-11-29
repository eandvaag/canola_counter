import os

from models.common import model_interface, group_interface
import image_set

def handle_request(req):

    req_type = req["request_type"]
    req_args = req["request_args"]

    if req_type == "register_image_set":
        image_set.register_image_set(req_args)

    elif req_type == "create_model":
        model_interface.create_model(req_args)

    elif req_type == "train_model":
        model_interface.train_model(req_args)

    elif req_type == "run_inference":
        model_interface.run_inference(req_args)

    elif req_type == "run_group":
        group_interface.run_group(req_args)

    elif req_type == "resume_group":
        group_interface.resume_group(req_args)

    elif req_type == "destroy_group":
        group_interface.destroy_group(req_args)

    else:
        raise RuntimeError("Unknown request type: '{}'.".format(req_type))