import tensorflow as tf




def resnet_50_backbone():

    model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False, 
        input_shape=[None, None, 3]
    )

    return model


def resnet_101_backbone():

    model = tf.keras.applications.ResNet101(
        weights="imagenet",
        include_top=False, 
        input_shape=[None, None, 3]
    )

    return model


def resnet_152_backbone():

    model = tf.keras.applications.ResNet152(
        weights="imagenet",
        include_top=False, 
        input_shape=[None, None, 3]
    )

    return model