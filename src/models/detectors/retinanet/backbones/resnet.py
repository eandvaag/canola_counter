import tensorflow as tf


def resnet_50_backbone():

    model = tf.keras.applications.ResNet50(
        weights="imagenet",
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


def resnet_101_backbone():

    model = tf.keras.applications.ResNet101(
        weights="imagenet",
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


def resnet_152_backbone():

    model = tf.keras.applications.ResNet152(
        weights="imagenet",
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