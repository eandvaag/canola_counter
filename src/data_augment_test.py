import os
import tqdm
import tensorflow as tf
import numpy as np
import cv2

from io_utils import tf_record_io
from models.common import box_utils
from models.common import data_augment
from models.common import model_vis


class DataLoader:

    def __init__(self, tf_record_paths):
        self.tf_record_paths = tf_record_paths


    def create_dataset(self):

        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)

        dataset_size = np.sum([1 for _ in dataset])

        #dataset = dataset.take(1)

        autotune = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(autotune)

        return dataset, dataset_size


    def read_tf_sample(self, tf_sample):

        sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
        img_path, img, boxes, classes = self._preprocess(sample)

        return img_path, img, boxes, classes


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
        #print(img)
        #img = tf.io.read_file(filename=img_path)
        #img = tf.image.decode_image(contents=img, channels=3, dtype=tf.dtypes.float32)

        h, w = img.shape[:2]
        boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
        classes = (tf.sparse.to_dense(sample["patch_classes"])).numpy().astype(np.float32)


        #if self.augment:
        #    img, boxes = data_augment.apply_augmentations(self.data_augmentations, img, boxes)

        #boxes = self._box_preprocess(boxes)

        return img_path, img, boxes, classes


def output_result(img, boxes, classes, out_path):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes = np.stack([
            boxes[:, 1] * img.shape[1],
            boxes[:, 0] * img.shape[0],
            boxes[:, 3] * img.shape[1],
            boxes[:, 2] * img.shape[0]
    ], axis=-1).astype(np.int32)

    #print("boxes", boxes)
    out = model_vis.draw_boxes_on_image(img,
                                    boxes,
                                    classes,
                                    np.ones(shape=classes.shape),
                                    None,
                                    display_class=False,
                                    display_score=False)

    cv2.imwrite(out_path, out)



if __name__ == "__main__":


    #tf_record_path = "/home/eaa299/Documents/work/2021/plant_detection/plant_detection" + \
    #                 "/src/usr/data/image_sets/canola_row_spacing_2/2020_06_08/patches/" + \
    #                 "02916d2e-aac0-45bb-9658-caa33082462b/patches-with-boxes-record.tfrec"

    tf_record_path = "/home/eaa299/workspace/plant_detection/plant_detection/src/usr/data/models/" + \
                    "6dbabda4-6756-4fb1-aa0b-e94b06754635/source_patches/0/training/annotated-patches-record.tfrec"

    tf_record_path = "/home/eaa299/workspace/plant_detection/plant_detection/src/usr/data/kaylie/" + \
                    "image_sets/MORSE/Dugout/2022-05-27/model/prediction/images/UNI_MORSEfung_D_May26_101a/patches-record.tfrec"
    out_dir = "/home/eaa299/workspace/augmentation/" #Documents/work/2021/augmentation_test"
    
    data_loader = DataLoader([tf_record_path])
    dataset, dataset_size = data_loader.create_dataset()
    steps = 5
    max_steps = 5

    for step, tf_sample in enumerate(tqdm.tqdm(dataset, total=steps, desc="Generating augmented images")):

        if step == max_steps:
            break

        img_path, img, boxes, classes = data_loader.read_tf_sample(tf_sample)
        img_name = os.path.basename(img_path)
        img_name_split = img_name.split(".")
        img_id = img_name_split[0]

        output_result(img, boxes, classes, os.path.join(out_dir, img_id + "_original.png"))


        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                     [{"type": "flip_horizontal", "parameters": {"probability": 1.0}}], img, boxes, classes)
        # output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_flip_horizontal.png"))

        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                     [{"type": "flip_vertical", "parameters": {"probability": 1.0}}], img, boxes, classes)
        # output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_flip_vertical.png"))

        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                     [{"type": "rotate", "parameters": {"probability": 1.0, "angle_limit": [90, 90]}}], img, boxes, classes)
        # output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_rotate_90.png"))

        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                     [{"type": "rotate", "parameters": {"probability": 1.0, "angle_limit": [45, 45]}}], img, boxes, classes)
        # output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_rotate_45.png"))

        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                     [{"type": "rotate", "parameters": {"probability": 1.0, "angle_limit": [15, 15]}}], img, boxes, classes)
        # output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_rotate_15.png"))

        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                     [{"type": "rotate_90", "parameters": {"probability": 1.0}}], img, boxes, classes)
        # output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_rotate90.png"))        

        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                  [{"type": "affine", "parameters": 
        #                                  {"probability": 1.0, 
        #                                  "scale": 1.0, 
        #                                  "translate_percent": [0, 0], 
        #                                  "rotate": [-360, 360], 
        #                                  "shear": [-40, 40]}}],
        #                                  img, boxes, classes)
        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #                                  [{"type": "affine", "parameters": 
        #                                  {"probability": 1.0, 
        #                                  "scale": 1.0, 
        #                                  "translate_percent": (-0.3, 0.3), 
        #                                  "rotate": 0, 
        #                                  "shear": 0}}],
        #                                  img, boxes, classes)


        #aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #    [{"type": "brightness",
        #      "parameters": {"probability": 1.0, "limit": [0, 0]}}],
        #      img, boxes, classes)
        # aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
        #    [{"type": "brightness_contrast",
        #      "parameters": {"probability": 1.0, "brightness_limit": [-0.2, 0.2], "contrast_limit": [-0.2, 0.2]}}],
        #      img, boxes, classes)
        #print(aug_img)

        aug_img, aug_boxes, aug_classes = data_augment.apply_augmentations(
            [{"type": "rgb_shift",
              "parameters": {"probability": 1.0, "r_shift_limit": [-30, 30], "g_shift_limit": [-30, 30], "b_shift_limit": [-30, 30]}}],
              img, boxes, classes)
        # shear: -40, 40, rotate: -360, 360
        output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_rgb_shift.png"))        

        output_result(aug_img, aug_boxes, aug_classes, os.path.join(out_dir, img_id + "_augmented.png"))        


        # aug_img, aug_boxes = data_augment.flip_horizontal(img, boxes)
        # output_result(aug_img, aug_boxes, classes, os.path.join(out_dir, img_id + "_flip_horizontal.png"))

        # aug_img, aug_boxes = data_augment.flip_vertical(img, boxes)
        # output_result(aug_img, aug_boxes, classes, os.path.join(out_dir, img_id + "_flip_vertical.png"))

        # aug_img, aug_boxes = data_augment.rotate_90(img, boxes, 0)
        # output_result(aug_img, aug_boxes, classes, os.path.join(out_dir, img_id + "_rotate_90_k0.png"))

        # aug_img, aug_boxes = data_augment.rotate_90(img, boxes, 1)
        # output_result(aug_img, aug_boxes, classes, os.path.join(out_dir, img_id + "_rotate_90_k1.png"))

        # aug_img, aug_boxes = data_augment.rotate_90(img, boxes, 2)
        # output_result(aug_img, aug_boxes, classes, os.path.join(out_dir, img_id + "_rotate_90_k2.png"))

        # aug_img, aug_boxes = data_augment.rotate_90(img, boxes, 3)
        # output_result(aug_img, aug_boxes, classes, os.path.join(out_dir, img_id + "_rotate_90_k3.png"))

        # aug_img = data_augment.adjust_brightness(img, 0.1)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_brightness_01.png"))

        # aug_img = data_augment.adjust_brightness(img, 0.2)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_brightness_02.png"))

        # aug_img = data_augment.adjust_brightness(img, 0.3)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_brightness_03.png"))

        # aug_img = data_augment.adjust_brightness(img, -0.1)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_brightness_-01.png"))

        # aug_img = data_augment.adjust_brightness(img, -0.2)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_brightness_-02.png"))

        # aug_img = data_augment.adjust_brightness(img, -0.3)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_brightness_-03.png"))


        # aug_img = data_augment.adjust_saturation(img, 0.5)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_saturation_05.png"))

        # aug_img = data_augment.adjust_saturation(img, 1)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_saturation_1.png"))

        # aug_img = data_augment.adjust_saturation(img, 2)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_saturation_2.png"))

        # aug_img = data_augment.adjust_saturation(img, 3)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_saturation_3.png"))

        
        # aug_img = data_augment.adjust_saturation(img, 2)
        # aug_img = data_augment.adjust_brightness(aug_img, 0.2)
        # output_result(aug_img, boxes, classes, os.path.join(out_dir, img_id + "_saturation_2_brightness_02.png"))
