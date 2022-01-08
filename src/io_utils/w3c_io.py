import uuid
import os

from io_utils import xml_io




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

    return res


