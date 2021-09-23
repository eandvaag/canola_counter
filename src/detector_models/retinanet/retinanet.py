import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import tqdm
import cv2
from abc import ABC

import extract_patches as ep

from io_utils import json_io
from io_utils import tf_record_io


def swap_xy(boxes):
    """Swaps the order of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )







def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)



class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            #print("gt_boxes[{}]: {}".format(i, gt_boxes[i]))
            #print("cls_ids[{}]: {}".format(i, cls_ids[i]))
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            #print("label[{}]: {}".format(i, label[i]))
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()




class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes,
        confidence_threshold,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )







def _get_dataset_size(dataset):
    size = 0
    for _ in dataset:
        size += 1
    return size





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




def random_flip_horizontal(image, boxes):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes





def preprocess_data(sample, augment):
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
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
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


def prepare_dataset_for_training(dataset, config, shuffle, augment):

    autotune = tf.data.experimental.AUTOTUNE
    label_encoder = LabelEncoder()

    dataset = dataset.map(lambda x: preprocess_data(x, augment), num_parallel_calls=autotune)
    if shuffle:
        dataset = dataset.shuffle(8 * config["batch_size"])
    dataset = dataset.padded_batch(batch_size=config["batch_size"], padding_values=(0.0, 1e-8, -1), drop_remainder=True)
    dataset = dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.prefetch(autotune)
    return dataset



class DataLoader(ABC):

    def __init__(self, tf_record_path, config):
        super().__init__()
        self.tf_record_path = tf_record_path
        self.input_img_min_side = config["input_img_min_side"]
        self.input_img_max_side = config["input_img_max_side"]
        self.smallest_fmap_stride = config["smallest_fmap_stride"]
        self.input_img_channels = config["input_img_channels"]

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

        return img, img_shape, ratio

class PredictionDataLoader(DataLoader):


    def create_dataset(self):
        dataset = tf.data.TFRecordDataset(filenames=[self.tf_record_path])
        dataset_size = _get_dataset_size(dataset)
        #dataset = dataset.batch(1)

        return dataset, dataset_size
    
    def read_sample(self, tf_sample, is_annotated):
        #tf_sample = tf_sample[0]
        #for tf_sample in data:
        #print("tf_sample", tf_sample)
        sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=is_annotated)
        #print("sample['scenario_uuid']", sample["scenario_uuid"])
        #print("sample['scenario_uuid']", bytes.decode(sample["scenario_uuid"].numpy()))
        img, ratio = self._preprocess(sample)
        return tf.expand_dims(img, axis=0), ratio, sample

    def _preprocess(self, sample):
        #img_path = sample["patch_path"]
        img_path = bytes.decode((sample["patch_path"]).numpy())
        img = cv2.imread(img_path)
        #img_raw = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        #img = tf.cast(img, dtype=tf.float32)
        #print("img", img)
        img, _, ratio = self._img_preprocess(img)
        img = tf.keras.applications.resnet.preprocess_input(img)
        return img, ratio


class TrainDataLoader(DataLoader):

    def __init__(self, tf_record_path, config):
        super().__init__(tf_record_path, config)
        self.batch_size = config["batch_size"]
        self.label_encoder = LabelEncoder()



    def create_batched_dataset(self):

        dataset = tf.data.TFRecordDataset(filenames=[self.tf_record_path])
        dataset_size = _get_dataset_size(dataset)
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
            #print("sample", sample)
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
            #padded_classes = tf.pad(classes, [[0, most_boxes - boxes.shape[0]], [0, 0]], "CONSTANT", constant_values=-1)
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
        boxes = convert_to_xywh(boxes)
        return boxes


    def _preprocess(self, sample):
        #sample = tf_record_io.parse_sample_from_tf_record(sample, True)
        
        img_path = sample["patch_path"]
        img_raw = tf.io.read_file(filename=img_path)
        img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        img, img_shape, ratio = self._img_preprocess(img)
        
        #patch_normalized_boxes = tf.sparse.to_dense(sample["patch_normalized_boxes"])
        boxes = swap_xy(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))
        classes = tf.sparse.to_dense(sample["patch_classes"])

        #boxes = []
        #classes = []
        #num_boxes = len(patch_normalized_boxes) // 4


        #for i in range(num_boxes):
        #    y_min = patch_normalized_boxes[i * 4]
        #    x_min = patch_normalized_boxes[(i * 4) + 1]
        #    y_max = patch_normalized_boxes[(i * 4) + 2]
        #    x_max = patch_normalized_boxes[(i * 4) + 3]
        #    class_id = patch_classes[i]
        #
        #    boxes.append([x_min, y_min, x_max, y_max])
        #    classes.append(class_id)


        #boxes = np.array(boxes)
        classes = tf.cast(classes, dtype=tf.int32)

        #boxes = swap_xy(boxes)
        # boxes = tf.stack(
        #     [
        #         boxes[:, 0] * img_shape[1],
        #         boxes[:, 1] * img_shape[0],
        #         boxes[:, 2] * img_shape[1],
        #         boxes[:, 3] * img_shape[0]
        #     ],
        #     axis=-1
        # )
        boxes = self._box_preprocess(img_shape, boxes)
        #print("boxes", boxes)
        #print("classes", classes)
        return img, boxes, classes









def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )



class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output



def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head



class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        #print("cls_labels", cls_labels)
        #print("cls_predictions", cls_predictions)
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        #print("(1) clf_loss", clf_loss)
        #print("(1) box_loss", box_loss)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        #print("(2) clf_loss", clf_loss)
        #print("(2) box_loss", box_loss)        
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        #print("normalizer", normalizer)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        #print("(3) clf_loss", clf_loss)
        #print("(3) box_loss", box_loss)
        #print("loss", loss)
        return loss

def load_inference_model(model_dir, config):

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(config["num_classes"])
    retinanet = RetinaNet(config["num_classes"], resnet50_backbone)

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    retinanet.compile(loss=loss_fn, optimizer=optimizer)

    weights_dir = os.path.join(model_dir, "weights")
    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    retinanet.load_weights(latest_checkpoint)

    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = retinanet(image, training=False)
    detections = DecodePredictions(num_classes=config["num_classes"], 
                                   confidence_threshold=config["confidence_threshold"])(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    return inference_model



def generate_predictions(patches_dir, model_dir, found_behaviour="skip"):

    config = json_io.load_json(os.path.join(model_dir, "model_config.json"))

    tf_record_path = os.path.join(patches_dir, "record.tfrec")
    patch_info = ep.parse_patches_dir(patches_dir)
    is_annotated = patch_info["is_annotated"]

    data_loader = PredictionDataLoader(tf_record_path, config)
    dataset, dataset_size = data_loader.create_dataset()

    pred_dir = os.path.join(model_dir, os.path.basename(patches_dir) + "-predictions")
    pred_path = os.path.join(pred_dir, "predictions.json")


    if os.path.exists(pred_path):
        if found_behaviour == "skip":
            return pred_dir
        elif found_behaviour == "replace":
            shutil.rmtree(pred_dir)
    
    os.makedirs(pred_dir)


    inference_model = load_inference_model(model_dir, config)

    prediction_data = {"predictions": []}

    for step, sample in enumerate(tqdm.tqdm(dataset, total=dataset_size, desc="Generating patch predictions")):

        patch, ratio, sample_data = data_loader.read_sample(sample, is_annotated=is_annotated)
        detections = inference_model.predict(patch)
        num_detections = detections.valid_detections[0]
        classes = detections.nmsed_classes[0][:num_detections]
        scores = detections.nmsed_scores[0][:num_detections]
        patch_abs_boxes = swap_xy(detections.nmsed_boxes[0][:num_detections] / ratio)
        prediction_data["predictions"].append({
            "img_path": bytes.decode((sample_data["img_path"]).numpy()),
            "patch_path": bytes.decode((sample_data["patch_path"]).numpy()),
            #"scenario_uuid": bytes.decode((sample_data["scenario_uuid"]).numpy()),
            "patch_coords": tf.sparse.to_dense(sample_data["patch_coords"]).numpy().tolist(),
            "pred_patch_abs_boxes": patch_abs_boxes.numpy().tolist(),
            "pred_classes": classes.tolist(),
            "pred_scores": scores.tolist()
        })

    json_io.save_json(pred_path, prediction_data)
    return pred_dir


def train(train_patches_dir, val_patches_dir, model_dir):

    config = json_io.load_json(os.path.join(model_dir, "model_config.json"))


    train_tf_record_path = os.path.join(train_patches_dir, "record.tfrec")
    val_tf_record_path = os.path.join(val_patches_dir, "record.tfrec")

    #train_data_loader = TrainDataLoader(train_tf_record_path, config)
    #train_dataset, train_dataset_size = train_data_loader.create_batched_dataset()

    #val_data_loader = TrainDataLoader(val_tf_record_path, config)
    #val_dataset, val_dataset_size = val_data_loader.create_batched_dataset()

    #print("Found {} train images and {} validation images.".format(train_dataset_size, val_dataset_size))


    resnet50_backbone = get_backbone()
    retinanet = RetinaNet(config["num_classes"], resnet50_backbone)

    weights_dir = os.path.join(model_dir, "weights")

    #if load_model:
    #    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    #    retinanet.load_weights(latest_checkpoint)

    loss_fn = RetinaNetLoss(config["num_classes"])
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)


    train_dataset, train_dataset_size = load_all_data(train_tf_record_path, config)
    val_dataset, val_dataset_size = load_all_data(val_tf_record_path, config)

    train_dataset = prepare_dataset_for_training(train_dataset, config, shuffle=True, augment=True)
    val_dataset = prepare_dataset_for_training(val_dataset, config, shuffle=False, augment=False)

    train_steps_per_epoch = train_dataset_size // config["batch_size"]
    val_steps_per_epoch = val_dataset_size // config["batch_size"]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(weights_dir, "epoch-{epoch}"),
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=3, restore_best_weights=True)
    ]

    retinanet.compile(loss=loss_fn, optimizer=optimizer)
    retinanet.fit(train_dataset,
              validation_data=val_dataset,
              batch_size=config["batch_size"],
              epochs=config["num_epochs"],
              callbacks=callbacks,
              steps_per_epoch=train_steps_per_epoch,
              validation_steps=val_steps_per_epoch,
              verbose=1)

    return






    train_loss_metric = tf.metrics.Mean()
    val_loss_metric = tf.metrics.Mean()
    steps_per_epoch = int(tf.math.ceil(train_dataset_size / config["batch_size"]))

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = retinanet(batch_images, training=True)
            #print(batch_labels)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
        #print("loss_value", loss_value)
        gradients = tape.gradient(target=loss_value, sources=retinanet.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, retinanet.trainable_variables))
        #print("loss_value", loss_value)
        train_loss_metric.update_state(values=loss_value)



    cardinality = np.sum([1 for i in train_dataset])
    for epoch in range(config["num_epochs"]):
        for step, batch_data in enumerate(tqdm.tqdm(train_dataset, total=cardinality)):
            step_start_time = time.time()
            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
            train_step(batch_images, batch_labels)
            step_end_time = time.time()

            print("Epoch: {}/{}, step: {}/{}, loss: {}, time_cost: {:.3f}s".format(epoch,
                                                                                  config["num_epochs"],
                                                                                  step,
                                                                                  steps_per_epoch,
                                                                                  train_loss_metric.result(),
                                                                                  step_end_time - step_start_time))
        train_loss_metric.reset_states()




        for step, batch_data in enumerate(tqdm.tqdm(train_dataset, desc="Evaluating training set")):
            batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
            pred = retinanet(batch_images, training=False)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
            train_loss_metric.update_state(values=loss_value)
        print("Epoch: {}/{}: validation loss: {}".format(epoch, config["num_epochs"], train_loss_metric.result()))
        train_loss_metric.reset_states()

        for step, batch_data in enumerate(tqdm.tqdm(val_dataset, desc="Evaluating validation set")):
            batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
            pred = retinanet(batch_images, training=False)
            loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
            val_loss_metric.update_state(values=loss_value)
        print("Epoch: {}/{}: validation loss: {}".format(epoch, config["num_epochs"], val_loss_metric.result()))
        val_loss_metric.reset_states()



        if epoch % config["save_frequency"] == 0:
            retinanet.save_weights(filepath=os.path.join(weights_dir, "epoch-{}".format(epoch)), save_format="tf")

    retinanet.save_weights(filepath=os.path.join(weights_dir, "epoch-{}".format(epoch)), save_format="tf")