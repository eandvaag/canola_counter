

class Decoder:

	def __init__(self, config, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.num_classes = self.config.arch["num_classes"]
		self.xy_scales = self.config.arch["xy_scales"]
		self.strides = self.config.arch["strides"]
		self.anchors = self.config.arch["anchors"]


	def __call__(self, conv_outputs):

		decoded_fm = []
		for i, conv_output in enumerate(conv_outputs):
			conv_shape = tf.shape(conv_output)
			batch_size = conv_shape[0]
			output_size = conv_shape[1]

			# second last dimension is the number of anchors per stage (which is always 3)
			# last dimension is box predictions (4) + objectness (1) + num_classes
			# basically this reshape command just divides the volume so that there is an axis for the anchors
			conv_output = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + self.num_classes))

			conv_raw_dxdy, conv_raw_dwdh, conv_raw_obj, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.num_classes), axis=-1)

			x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=0), [output_size, 1])
			y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=1), [1, output_size])
			xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)		# [gx, gy, 1, 2]

			xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
			xy_grid = tf.cast(xy_grid, tf.float32)

			pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.xy_scales[i]) - 0.5 * (self.xy_scales[i] - 1) + xy_grid) * self.strides[i]
			pred_wh = tf.exp(conv_raw_dwdh) * self.anchors[i]
			max_scale = tf.cast(output_size * self.strides[i], dtype=tf.float32)

			pred_wh = tf.clip_by_value(pred_wh, clip_value_min=0.0, clip_value_max=max_scale)
			pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
			pred_obj = tf.sigmoid(conv_raw_obj)
			pred_prob = tf.sigmoid(conv_raw_prob)

			decoded_fm.append(tf.concat([pred_xywh, pred_obj, pred_prob], axis=-1))
		return decoded_fm





