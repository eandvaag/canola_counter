import tensorflow as tf


def keras_resnet50_backbone(load_imagenet_weights):

    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.ResNet50(
        weights=weights,
        include_top=False, 
        input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        model.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def keras_resnet101_backbone(load_imagenet_weights):
    
    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.ResNet101(
        weights=weights,
        include_top=False,
        input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        model.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def keras_resnet152_backbone(load_imagenet_weights):

    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.ResNet152(
        weights=weights,
        include_top=False,
        input_shape=[None, None, 3]
    )

    c3_output, c4_output, c5_output = [
        model.get_layer(layer_name).output
        for layer_name in ["conv3_block8_out", "conv4_block36_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output]
    )