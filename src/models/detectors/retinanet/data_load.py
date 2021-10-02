from abc import ABC
import numpy as np
import tensorflow as tf
import cv2

import models.detectors.common.box_utils as box_utils
import models.detectors.common.data_augment as data_augment

from models.detectors.retinanet.encode import LabelEncoder

from io_utils import tf_record_io

class DatasetWrapper():

    def __init__(self, images, boxes, labels):

        self.images = images
        self.boxes = boxes
        self.labels = labels

    def generator(self):

        for (image, box, label) in zip(self.images, self.boxes, self.labels):

            yield {'image': image,
                    'objects': {
                        'bbox': box,
                        'label': label
                    }
                }

def load_all_data(tf_record_path, config):

    images = []
    bboxes = []
    labels = []
    num_images = 0

    img_dtype = tf.dtypes.float32
    bbox_dtype = tf.dtypes.float32
    label_dtype = tf.dtypes.int64

    dataset = tf.data.TFRecordDataset(filenames=[tf_record_path])
    for sample in dataset:
        sample = tf_record_io.parse_sample_from_tf_record(sample, is_annotated=True)
        #image_raw = tf.io.read_file(filename=sample["patch_path"])
        #image = tf.io.decode_image(contents=image_raw, channels=config["input_img_channels"], dtype=tf.float32, expand_animations=False)
        img_path = bytes.decode((sample["patch_path"]).numpy())
        image = cv2.imread(img_path)
        patch_normalized_boxes = tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4))
        patch_classes = tf.sparse.to_dense(sample["patch_classes"])

        images.append(image)
        bboxes.append(patch_normalized_boxes)
        labels.append(patch_classes)
        num_images += 1

    img_shape = images[0].shape

    dataset_wrapper = DatasetWrapper(images, bboxes, labels)
    dataset = tf.data.Dataset.from_generator(dataset_wrapper.generator, 
        output_types={'image': img_dtype, 'objects': {'bbox': bbox_dtype, 'label': label_dtype}}, 
        output_shapes={'image': img_shape, 'objects': {'bbox': (None, 4), 'label': (None, )}}).repeat(config["num_epochs"])

    return dataset, num_images 


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio








def preprocess_data(sample, augment):
    image = sample["image"]
    bbox = box_utils.swap_xy_tf(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    if augment:
        image, bbox = random_flip_horizontal(image, bbox)

    image, image_shape, _ = resize_and_pad_image(image, jitter=None)
    
    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = box_utils.convert_to_xywh_tf(bbox)
    return image, bbox, class_id


def prepare_dataset_for_training(dataset, config, shuffle, augment):

    autotune = tf.data.experimental.AUTOTUNE
    label_encoder = LabelEncoder()

    dataset = dataset.map(lambda x: preprocess_data(x, augment), num_parallel_calls=autotune)
    if shuffle:
        dataset = dataset.shuffle(8 * config.train_batch_size)
    dataset = dataset.padded_batch(batch_size=config.train_batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
    dataset = dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.prefetch(autotune)
    return dataset



class DataLoader(ABC):

    def __init__(self, tf_record_path, config):
        super().__init__()
        self.tf_record_path = tf_record_path
        self.input_img_min_side = config.input_img_min_side
        self.input_img_max_side = config.input_img_max_side
        self.smallest_fmap_stride = config.smallest_fmap_stride
        self.input_img_channels = config.input_img_channels

    def _img_preprocess(self, img):

        min_side = self.input_img_min_side
        max_side = self.input_img_max_side
        stride = self.smallest_fmap_stride


        img_shape = tf.cast(tf.shape(img)[:2], dtype=tf.float32)

        ratio = min_side / tf.reduce_min(img_shape)
        if ratio * tf.reduce_max(img_shape) > max_side:
            ratio = max_side / tf.reduce_max(img_shape)

        img_shape = ratio * img_shape
        img = tf.image.resize(img, tf.cast(img_shape, dtype=tf.int32))
        padded_img_shape = tf.cast(tf.math.ceil(img_shape / stride) * stride, dtype=tf.int32)
        img = tf.image.pad_to_bounding_box(img, 0, 0, padded_img_shape[0], padded_img_shape[1])

        ratio = [1.0 / ratio, 1.0 / ratio]

        return img, img_shape, ratio

class InferenceDataLoader(DataLoader):

    def __init__(self, tf_record_path, config):
        super().__init__(tf_record_path, config)
        self.batch_size = config.inference_batch_size

    def create_dataset(self):
        dataset = tf.data.TFRecordDataset(filenames=[self.tf_record_path])
        dataset_size = np.sum([1 for _ in dataset])
        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset, dataset_size
    
    def read_batch_data(self, batch_data, is_annotated):
        batch_imgs = []
        batch_info = []
        batch_ratios = []
        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=is_annotated)
            img, ratio = self._preprocess(sample)
            batch_imgs.append(img)
            batch_ratios.append(ratio)
            batch_info.append(sample)

        batch_imgs = tf.stack(batch_imgs, axis=0)
        return batch_imgs, batch_ratios, batch_info


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())
        img = cv2.imread(img_path)
        #img_raw = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        img, _, ratio = self._img_preprocess(img)
        return img, ratio


class TrainDataLoader(DataLoader):

    def __init__(self, tf_record_path, config, shuffle, augment):
        super().__init__(tf_record_path, config)
        self.batch_size = config.train_batch_size
        self.label_encoder = LabelEncoder()
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config.data_augmentations


    def create_batched_dataset(self):

        dataset = tf.data.TFRecordDataset(filenames=[self.tf_record_path])
        dataset_size = np.sum([1 for _ in dataset])
        if self.shuffle:
            dataset = dataset.shuffle(dataset_size)
        dataset = dataset.batch(batch_size=self.batch_size)

        autotune = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(autotune)

        return dataset, dataset_size


    def read_batch_data(self, batch_data):

        batch_imgs = []
        batch_boxes = []
        batch_classes = []

        most_boxes = 0

        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            img, boxes, classes = self._preprocess(sample)
            num_boxes = boxes.shape[0]
            batch_imgs.append(img)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

            if num_boxes > most_boxes:
                most_boxes = num_boxes

        padded_batch_boxes = []
        padded_batch_classes = []
        for boxes, classes in zip(batch_boxes, batch_classes):
            padded_boxes = tf.pad(boxes, [[0, most_boxes - boxes.shape[0]], [0, 0]], "CONSTANT", constant_values=1e-8)
            padded_classes = tf.pad(classes, [[0, most_boxes - boxes.shape[0]]], "CONSTANT", constant_values=-1)
            padded_batch_boxes.append(padded_boxes)
            padded_batch_classes.append(padded_classes)

        batch_imgs = tf.stack(values=batch_imgs, axis=0)
        padded_batch_boxes = tf.stack(padded_batch_boxes, axis=0)
        padded_batch_classes = tf.stack(padded_batch_classes, axis=0)

        return self.label_encoder.encode_batch(batch_imgs, padded_batch_boxes, padded_batch_classes)





    def _box_preprocess(self, img_shape, boxes):
        boxes = tf.stack(
            [
                boxes[:, 0] * img_shape[1],
                boxes[:, 1] * img_shape[0],
                boxes[:, 2] * img_shape[1],
                boxes[:, 3] * img_shape[0]
            ],
            axis=-1
        )
        boxes = box_utils.convert_to_xywh_tf(boxes)
        return boxes


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())
        #img_raw = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        img = cv2.imread(img_path)
        boxes = box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))
        classes = tf.sparse.to_dense(sample["patch_classes"])


        if self.augment:
            img, boxes = data_augment.apply_augmentations(self.data_augmentations, img, boxes)

        img, img_shape, ratio = self._img_preprocess(img)
        boxes = self._box_preprocess(img_shape, boxes)
        classes = tf.cast(classes, dtype=tf.int32)

        return img, boxes, classes

