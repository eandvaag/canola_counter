import argparse
import os
import glob
import numpy as np
import cv2


import image_utils
from image_set import Image

def segment(image_path, out_path, threshold):
    image = Image(image_path)

    image_array = image.load_image_array()


    image_array_fp = np.float32(image_array) / 255
    exg_array = (2 * image_array_fp[:,:,1]) - image_array_fp[:,:,0] - image_array_fp[:,:,2]
    
    binary_mask = exg_array > threshold     
    #binary_mask = exg_array < threshold

    out_array = np.copy(image_array)

    # if gt_boxes is not None:
    #     shapes = np.zeros_like(image_array, np.uint8)
    #     for gt_box in gt_boxes:
    #         cv2.rectangle(shapes, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (255, 0, 0), -1)
    #     alpha = 0.25
    #     mask = shapes.astype(bool)
    #     out_array[mask] = cv2.addWeighted(image_array, alpha, shapes, 1-alpha, 0)[mask]
    binary_array = np.zeros(shape=out_array.shape, dtype=np.uint8)

    print("image_array.shape: {}, binary_mask.shape: {}, binary_array.shape: {}".format(
        image_array.shape, binary_mask.shape, binary_array.shape
    ))
    binary_array[:,:,:][binary_mask] = (255, 0, 0)
    #binary_array[:,:,:][np.logical_not(binary_mask)] = (255, 255, 255)

    alpha = 0.45
    #alpha = 0.50
    out_array[binary_mask] = cv2.addWeighted(image_array, alpha, binary_array, 1-alpha, 0)[binary_mask]


    #out_path = "usr/data/tmp_out_test.png"
    # img_dzi_path = "tmp_out_dzi"
    cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))

    desired_height = 650
    scale = desired_height / out_array.shape[0]
    desired_width = int(out_array.shape[1] * scale)
    resized = cv2.resize(out_array, (desired_width, desired_height))

    #img_dzi_path = out_path[:-4]
    #os.system("./MagickSlicer/magick-slicer.sh '" + out_path + "' '" + img_dzi_path + "'")

    low_res_out_path = out_path[:-4] + "_low_res.png"
    cv2.imwrite(low_res_out_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    #cv2.imwrite(out_path, cv2.cvtColor(binary_array, cv2.COLOR_RGB2BGR))
    # print("executing conv command")
    # conv_cmd =  "./MagickSlicer/magick-slicer.sh '" + out_path + "' '" + img_dzi_path + "'"
    # exec(conv_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    parser.add_argument("image_name", type=str)
    parser.add_argument("threshold", type=float)
    args = parser.parse_args()

    image_set_dir = os.path.join("usr", "data", "image_sets",
                              args.farm_name, args.field_name, args.mission_date)

    image_path = glob.glob(os.path.join(image_set_dir, "images", args.image_name + ".*"))[0]

    out_path = os.path.join(image_set_dir, "segmentations", args.image_name + ".png")

    segment(image_path, out_path, args.threshold)