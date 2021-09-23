from xml.etree import ElementTree as ET
import numpy as np


def load_boxes_and_classes(xml_path, class_map):

    boxes = []
    classes = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        obj_class = obj.find('name').text
        bnd_box = obj.find('bndbox')
        x_min = int(bnd_box.find('xmin').text)
        y_min = int(bnd_box.find('ymin').text)
        x_max = int(bnd_box.find('xmax').text)
        y_max = int(bnd_box.find('ymax').text)

        box = np.array([y_min, x_min, y_max, x_max])
        boxes.append(box)
           
        classes.append(class_map[obj_class])
            

    return np.array(boxes), np.array(classes)