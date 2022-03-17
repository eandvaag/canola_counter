
import tensorflow as tf


def get_model():
    weights = 'imagenet'
    model = tf.keras.applications.ResNet50(
        weights=weights,
        include_top=False, 
        input_shape=[None, None, 3],
        pooling="max"
    )
    #c3_output, c4_output, c5_output = [
    #    model.get_layer(layer_name).output
    #    for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    #]
    return model




# class DataLoader:

#     def __init__(self, tf_record_paths, input_image_shape, batch_size):
#         self.tf_record_paths = tf_record_paths
#         self.batch_size = batch_size
#         self.input_image_shape = input_image_shape


#     def create_batched_dataset(self):

#         dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)

#         num_images = np.sum([1 for _ in dataset])

#         dataset = dataset.batch(batch_size=self.batch_size)

#         autotune = tf.data.experimental.AUTOTUNE
#         dataset = dataset.prefetch(autotune)

#         return dataset, num_images

#     def read_batch_data(self, batch_data):

#         batch_images = []

#         for tf_sample in batch_data:
#             sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=False)
#             image = self._preprocess(sample)
#             batch_images.append(image)

#         batch_images = tf.stack(values=batch_images, axis=0)
#         #batch_images = tf.ragged.constant(batch_images)
#         #batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
#         return batch_images

#     def _preprocess(self, sample):
#         image_path = bytes.decode((sample["patch_path"]).numpy())
#         image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
#         image = tf.convert_to_tensor(image, dtype=tf.float32)
#         image = tf.image.resize(images=image, size=self.input_image_shape[:2])
#         return image