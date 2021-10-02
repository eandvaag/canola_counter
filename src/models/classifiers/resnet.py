import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filters, strides=1):
        """
            Uses Option (B) described in ResNet paper.
            Projection shortcuts are used for increasing dimensions,
            and other shortcuts are identity.
        """

        super(BasicBlock, self).__init__()



        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=strides,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=(1, 1),
                                       strides=strides,
                                       use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.downsample = tf.keras.layers.Lambda(lambda x: x)


    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output


class ResNetTypeI(tf.keras.Model):
    
    def __init__(self, blocks_per_layer, include_top=True):

        super(ResNetTypeI, self).__init__()
        self.include_top = include_top

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            use_bias=False)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.conv2 = ResNetTypeI.__res_block_layer(filters=64,
                                                   blocks=blocks_per_layer[0])
        self.conv3 = ResNetTypeI.__res_block_layer(filters=128,
                                                   blocks=blocks_per_layer[1],
                                                   l1_strides=2)
        self.conv4 = ResNetTypeI.__res_block_layer(filters=256,
                                                   blocks=blocks_per_layer[2],
                                                   l1_strides=2)
        self.conv5 = ResNetTypeI.__res_block_layer(filters=512,
                                                   blocks=blocks_per_layer[3],
                                                   l1_strides=2)
        if include_top:
            raise NotImplementedError()
        



    @staticmethod
    def __res_block_layer(filters, blocks, l1_strides=1):

        res_block = tf.keras.Sequential()
        res_block.add(BasicBlock(filters, strides=l1_strides))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filters, strides=1))

        return res_block

    def call(self, inputs, training=None, **kwargs):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)

        if self.include_top:
            raise NotImplementedError()

        return x







def resnet_18(include_top=True):
    return ResNetTypeI(blocks_per_layer=[2, 2, 2, 2], include_top=include_top)