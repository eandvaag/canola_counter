import tensorflow as tf

from models.detectors.centernet.backbones.resnet import resnet_50_backbone, \
														resnet_101_backbone, \
														resnet_152_backbone


def get_backbone(config):

	if config.backbone_name == "resnet_50":
		return resnet_50_backbone()
	elif config.backbone_name == "resnet_101":
		return resnet_101_backbone()
	elif config.backbone_name == "resnet_152":
		return resnet_152_backbone()
	else:
		raise RuntimeError("Unknown backbone: '{}'.".format(config.backbone_name))




def build_head(head_conv, output_filters):

	head = tf.keras.Sequential()

	if head_conv > 0:
		head.add(tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"))
		head.add(tf.keras.layers.ReLU())
	
	head.add(tf.keras.layers.Conv2D(filters=output_filters, kernel_size=(1, 1), strides=1, padding="same"))

	return head



class UpConvResNet(tf.keras.layers.Layer):

	def __init__(self, backbone, **kwargs):
		super(UpConvResNet, self).__init__(name="UpConvResNet", **kwargs)
		self.backbone = backbone
		self.transposed_conv_layers = self.__make_transposed_conv_layers(num_layers=3, 
                                                         			   num_filters=[256, 256, 256], 
                                                         			   num_kernels=[4, 4, 4])

	def call(self, images, training=False):
		images = tf.keras.applications.resnet.preprocess_input(images)
		c5_output = self.backbone(images, training=training)
		output = self.transposed_conv_layers(c5_output, training=training)
		return output

	def __make_transposed_conv_layers(self, num_layers, num_filters, num_kernels):
	    layers = tf.keras.Sequential()
	    for i in range(num_layers):
	        layers.add(tf.keras.layers.Conv2DTranspose(filters=num_filters[i],
	                                                kernel_size=num_kernels[i],
	                                                strides=2,
	                                                padding="same",
	                                                use_bias=False))
	        layers.add(tf.keras.layers.BatchNormalization())
	        layers.add(tf.keras.layers.ReLU())
	    return layers




class CenterNet(tf.keras.Model):

	def __init__(self, config):
		super(CenterNet, self).__init__(name="CenterNet")
		backbone = get_backbone(config)
		self.upconvresnet = UpConvResNet(backbone)
		self.heatmap_head = build_head(config.head_conv, config.num_classes)
		self.reg_head = build_head(config.head_conv, 2)
		self.wh_head = build_head(config.head_conv, 2)

	def call(self, image, training=None):
		features = self.upconvresnet(image, training=training)
		heatmap_outputs = self.heatmap_head(features, training=training)
		reg_outputs = self.reg_head(features, training=training)
		wh_outputs = self.wh_head(features, training=training)
		
		return tf.concat(values=[heatmap_outputs, reg_outputs, wh_outputs], axis=-1)