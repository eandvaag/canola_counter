import os
import numpy as np
import tensorflow as tf


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_patch_tf_records_for_img(img, patch_data, out_dir, is_annotated):

    patch_tf_records = []

    for i in range(len(patch_data["patches"])):

        patch_tf_record = {
            "img_path": bytes_feature(img.img_path),
            "patch_path": bytes_feature(os.path.join(out_dir, patch_data["patch_names"][i])),
            #"scenario_uuid": bytes_feature(scenario_uuid),
            "patch_coords": float_feature_list(patch_data["patch_coords"][i])
        }
        if is_annotated:
            patch_tf_record.update({
                "patch_normalized_boxes": float_feature_list(list(np.array(patch_data["patch_normalized_boxes"][i]).astype(np.float32).flatten())),
                "patch_abs_boxes": int_feature_list(list(np.array(patch_data["patch_abs_boxes"][i]).astype(np.int64).flatten())),
                "img_abs_boxes": int_feature_list(list(np.array(patch_data["img_abs_boxes"][i]).astype(np.int64).flatten())),
                "patch_classes": int_feature_list(np.array(patch_data["patch_classes"][i]).astype(np.int64))

            })
        patch_tf_records.append(tf.train.Example(features=tf.train.Features(feature=patch_tf_record)))

    return patch_tf_records


def create_patch_tf_prediction_records(patch_data):

    patch_tf_records = []

    for i in range(len(patch_data)):

        patch_tf_record = {
            "img_path": bytes_feature(patch_data[i]["img_path"]),
            "patch_path": bytes_feature(patch_data[i]["patch_path"]),
            #"scenario_uuid": bytes_feature(patch_data[i]["scenario_uuid"]),
            "patch_coords": int_feature_list(list(np.array(patch_data[i]["patch_coords"]).astype(np.int64))),
            "predicted_abs_boxes": float_feature_list(list(np.array(patch_data[i]["predicted_abs_boxes"]).astype(np.int64).flatten())),
            "predicted_classes": int_feature_list(list(np.array(patch_data[i]["predicted_classes"]).astype(np.int64))),
            "scores": float_feature_list(list(np.array(patch_data[i]["predicted_classes"]).astype(np.float32)))
        }
        patch_tf_records.append(tf.train.Example(features=tf.train.Features(feature=patch_tf_record)))

    return patch_tf_records







def output_patch_tf_records(out_dir, patch_tf_records):

    record_path = os.path.join(out_dir, "record.tfrec")
    with tf.io.TFRecordWriter(record_path) as writer:
        for patch_tf_record in patch_tf_records:
            writer.write(patch_tf_record.SerializeToString())


def parse_sample_from_tf_record(tf_sample, is_annotated):

    schema = {
        "img_path": tf.io.FixedLenFeature([], tf.string),
        "patch_path": tf.io.FixedLenFeature([], tf.string),
        #"scenario_uuid": tf.io.FixedLenFeature([], tf.string),
        "patch_coords": tf.io.VarLenFeature(tf.float32)
    }
    if is_annotated:
        schema.update({
            "patch_normalized_boxes": tf.io.VarLenFeature(tf.float32),
            "patch_abs_boxes": tf.io.VarLenFeature(tf.int64),
            "img_abs_boxes": tf.io.VarLenFeature(tf.int64),
            "patch_classes": tf.io.VarLenFeature(tf.int64)
        })

    sample = tf.io.parse_single_example(tf_sample, schema)
    #sample = tf.io.parse_example([tf_sample], schema)
    return sample