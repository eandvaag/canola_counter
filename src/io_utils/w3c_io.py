import uuid
import os

import math as m
import numpy as np

from io_utils import xml_io, json_io


def load_annotations(w3c_path, class_map):

    annotations = json_io.load_json(w3c_path)
    prefix_len = len("xywh=pixel:")

    #boxes = {}
    #classes = {}
    ret_annotations = {}

    for image_name in annotations.keys():
        ret_annotations[image_name] = {
            "status": annotations[image_name]["status"],
            "boxes": [],
            "classes": []
        }
        #boxes[image_name] = []
        #classes[image_name] = []
        for annotation in annotations[image_name]["annotations"]:
            px_str = annotation["target"]["selector"]["value"]
            cls_str = annotation["body"][0]["value"]
            xywh = [int(round(float(x))) for x in px_str[prefix_len:].split(",")]

            min_y = xywh[1]
            min_x = xywh[0]
            max_y = min_y + xywh[3]
            max_x = min_x + xywh[2]
            ret_annotations[image_name]["boxes"].append([min_y, min_x, max_y, max_x])
            ret_annotations[image_name]["classes"].append(class_map[cls_str])



        ret_annotations[image_name]["boxes"] = np.array(ret_annotations[image_name]["boxes"])
        ret_annotations[image_name]["classes"] = np.array(ret_annotations[image_name]["classes"])

    return ret_annotations #boxes, classes

            
def save_annotations(annotations_path, predictions, config):

    annotations = {}
    reverse_class_map = {v: k for k, v in config.arch["class_map"].items()}

    for image_name in predictions["image_predictions"].keys():

        annotations[image_name] = {}
        annotations[image_name]["annotations"] = []

        for i in range(len(predictions["image_predictions"][image_name]["pred_image_abs_boxes"])):

            annotation_uuid = str(uuid.uuid4())

            pred_class_num = predictions["image_predictions"][image_name]["pred_classes"][i]
            pred_score = predictions["image_predictions"][image_name]["pred_scores"][i]

            min_y = predictions["image_predictions"][image_name]["pred_image_abs_boxes"][i][0]
            min_x = predictions["image_predictions"][image_name]["pred_image_abs_boxes"][i][1]
            max_y = predictions["image_predictions"][image_name]["pred_image_abs_boxes"][i][2]
            max_x = predictions["image_predictions"][image_name]["pred_image_abs_boxes"][i][3]

            h = max_y - min_y
            w = max_x - min_x

            box_str = ",".join([str(min_x), str(min_y), str(w), str(h)])

            w3c_annotation = {
                "type": "Annotation",
                "body": [{
                    "type": "TextualBody",
                    "purpose": "class",
                    "value": reverse_class_map[pred_class_num]
                },
                {
                    "type": "TextualBody",
                    "purpose": "score",
                    "value": "%.2f" % pred_score
                }],
                "target": {
                    "source": "",
                    "selector": {
                        "type": "FragmentSelector",
                        "conformsTo": "http://www.w3.org/TR/media-frags/",
                        "value": "xywh=pixel:" + box_str
                    }
                },
                "@context": "http://www.w3.org/ns/anno.jsonld",
                "id": annotation_uuid
            }


            annotations[image_name]["annotations"].append(w3c_annotation)


    json_io.save_json(annotations_path, annotations)




def convert_xml_files_to_w3c(xml_dir, class_map):

    xml_files = os.listdir(xml_dir)

    reverse_class_map = {v:k for k,v in class_map.items()}

    res = {}

    for xml_file in xml_files:

        extensionless_name = xml_file[:-4]
        res[extensionless_name] = []

        xml_path = os.path.join(xml_dir, xml_file)

        boxes, classes = xml_io.load_boxes_and_classes(xml_path, class_map)

        for (box, cls_num) in zip(boxes, classes):

            if reverse_class_map[cls_num] == "plant":
                annotation_uuid = str(uuid.uuid4())

                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                box_str = ",".join([str(box[1]), str(box[0]), str(box_w), str(box_h)])

                w3c_annotation = {
                    "type": "Annotation",
                    "body": [{
                        "type": "TextualBody",
                        "purpose": "class",
                        "value": reverse_class_map[cls_num]
                    }],
                    "target": {
                        "source": "",
                        "selector": {
                            "type": "FragmentSelector",
                            "conformsTo": "http://www.w3.org/TR/media-frags/",
                            "value": "xywh=pixel:" + box_str
                        }
                    },
                    "@context": "http://www.w3.org/ns/anno.jsonld",
                    "id": annotation_uuid
                }

                res[extensionless_name].append(w3c_annotation)
            else:
                print("skipping weed")

    return res



def get_completed_images(annotations):
    return [image_name for image_name in annotations.keys() \
            if annotations[image_name]["status"] == "completed"]


def get_num_annotations(annotations, require_completed=True):
    num_annotations = 0
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed" or not require_completed:
            boxes = annotations[image_name]["boxes"]
            num_annotations += np.shape(boxes)[0]
    return num_annotations


def get_patch_size(annotations):
    

    median_box_area = get_median_box_area(annotations)

    #(40000 / 288) (90000 / 2296) 


    #slope = (90000 - 40000) / (2296 - 288)
    #patch_area = slope * (median_box_area - 288) + 40000

    patch_area = median_box_area * (90000 / 2296)
    patch_size = round(m.sqrt(patch_area))
    print("patch_size", patch_size)
    return patch_size
    

def get_median_box_area(annotations):
    
    box_areas = []
    for img_name in annotations.keys():
        boxes = annotations[img_name]["boxes"]
        if boxes.size > 0:
            img_box_areas = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])).tolist()
            box_areas.extend(img_box_areas)

    if len(box_areas) == 0:
        raise RuntimeError("No annotations found.") 

    return np.median(box_areas)
    