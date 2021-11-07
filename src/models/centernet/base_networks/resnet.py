import tensorflow as tf

from models.centernet.backbones.resnet import resnet18_backbone, \
                                              resnet34_backbone, \
                                              resnet50_backbone, \
                                              resnet101_backbone, \
                                              resnet152_backbone, \
                                              keras_resnet50_backbone, \
                                              keras_resnet101_backbone, \
                                              keras_resnet152_backbone


def get_backbone(backbone_config):

    backbone_type = backbone_config["backbone_type"]

    if backbone_config["keras_prebuilt"]:

        load_imagenet_weights = backbone_config["imagenet_pretrained"]
        
        if backbone_type == "resnet50":
            return keras_resnet50_backbone(load_imagenet_weights)

        elif backbone_type == "resnet101":
            return keras_resnet101_backbone(load_imagenet_weights)

        elif backbone_type == "resnet152":
            return keras_resnet152_backbone(load_imagenet_weights)

    else:

        if backbone_type == "resnet18":
            return resnet18_backbone()

        elif backbone_type == "resnet34":
            return resnet34_backbone()

        elif backbone_type == "resnet50":
            return resnet50_backbone()            

        elif backbone_type == "resnet101":
            return resnet101_backbone()

        elif backbone_type == "resnet152":
            return resnet152_backbone()


    raise RuntimeError("Invalid backbone configuration: '{}'.".format(backbone_config))



class ModifiedResNet(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(ModifiedResNet, self).__init__(name="ModifiedResNet", **kwargs)
        self.backbone = get_backbone(config.backbone_config)

        # Uses the approach found in "Simple Baselines for Human Pose Estimation and Tracking"
        self.transposed_conv_layers = self.__make_transposed_conv_layers(num_layers=3, 
                                                                         filters=[256, 256, 256], 
                                                                         kernel_sizes=[4, 4, 4])

    def call(self, images, training=False):
        images = tf.keras.applications.resnet.preprocess_input(images)
        c5_output = self.backbone(images, training=training)
        output = self.transposed_conv_layers(c5_output, training=training)
        return output

    def __make_transposed_conv_layers(self, num_layers, filters, kernel_sizes):
        layers = tf.keras.Sequential()
        for i in range(num_layers):
            layers.add(tf.keras.layers.Conv2DTranspose(filters=filters[i],
                                                       kernel_size=kernel_sizes[i],
                                                       strides=2,
                                                       padding="same",
                                                       use_bias=False))
            layers.add(tf.keras.layers.BatchNormalization())
            layers.add(tf.keras.layers.ReLU())
        return layers