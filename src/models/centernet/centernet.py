import tensorflow as tf

from models.centernet.base_networks import resnet, efficientdet



def get_base_network(config):

    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    efficient_dets = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

    if config.backbone_config["backbone_type"] in resnets:
        return resnet.ModifiedResNet(config)
    elif config.backbone_config["backbone_type"] in efficient_dets:
        return efficientdet.ModifiedEfficientDet(config)
    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config.backbone_config["backbone_type"]))



def build_head(head_conv, output_filters):

    head = tf.keras.Sequential()

    # additional layers for the network head -- this is not referred to in the original 'Objects as Points' paper.
    if head_conv > 0:
        head.add(tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"))
        head.add(tf.keras.layers.ReLU())
    
    head.add(tf.keras.layers.Conv2D(filters=output_filters, kernel_size=(1, 1), strides=1, padding="same"))

    return head







class CenterNet(tf.keras.Model):

    def __init__(self, config):
        super(CenterNet, self).__init__(name="CenterNet")
        self.base_network = get_base_network(config)
        self.heatmap_head = build_head(config.head_conv, config.num_classes)
        self.reg_head = build_head(config.head_conv, 2)
        self.wh_head = build_head(config.head_conv, 2)

    def call(self, image, training=None):
        features = self.base_network(image, training=training)
        heatmap_outputs = self.heatmap_head(features, training=training)
        reg_outputs = self.reg_head(features, training=training)
        wh_outputs = self.wh_head(features, training=training)
        
        return tf.concat(values=[heatmap_outputs, reg_outputs, wh_outputs], axis=-1)
