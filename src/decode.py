from abc import ABC



class DetectorDecoder(ABC, tf.keras.layers.Layer):


	def __init__(self, config):
		super(DetectorDecoder, self).__init__()





class RetinaNetDecoder(DetectorDecoder):
	
	def __init__(self, config):
		super(RetinaNetDecoder, self).__init__()
		self.num_classes = config["num_classes"]
		self.nms_iou_threshold = config["nms_iou_threshold"]
		self.max_detections_per_class = config["max_detections_per_class"]
		self.max_detections = config["max_detections"]

		self._anchor_box = AnchorBox()
		self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)