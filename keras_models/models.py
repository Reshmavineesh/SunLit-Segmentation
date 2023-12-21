import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Add,
    Input,
    Layer,
    Conv2D,
    Dropout,
    MaxPool2D,
    Activation,
    concatenate,
    Concatenate,
    MaxPooling2D,
    UpSampling2D,
    Convolution2D,
    Conv2DTranspose,
    AveragePooling2D,
    BatchNormalization,
)


def create_layer(x, filters, dropout_rate=0.2, weight_decay=1e-4):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters,
        kernel_size=3,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
        padding="same",
    )(x)
    x = Dropout(dropout_rate)(x)
    return x


def create_dense_block(
    x, growth_rate=16, num_layers=4, dropout_rate=0.2, weight_decay=1e-4
):
    for i in range(num_layers):
        l = create_layer(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, l])
    return x


def create_transition_down(x, dropout_rate=0.2, weight_decay=1e-4):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        x.get_shape().as_list()[-1],
        1,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
        padding="same",
    )(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPool2D()(x)
    return x


def create_transition_up(x, filters, weight_decay=1e-4):
    x = Conv2DTranspose(
        filters,
        3,
        strides=(2, 2),
        padding="same",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    return x


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.name_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )


def Upsampling(inputs, scale):
    return UpSampling2D(size=(scale, scale))(inputs)


def ResidualUnit(inputs, n_filters=48, filter_size=3):
    net = Conv2D(n_filters, filter_size, padding="same")(inputs)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Conv2D(n_filters, filter_size, padding="same")(net)
    net = BatchNormalization()(net)
    return Add()([inputs, net])


def FullResolutionResidualUnit(
    pool_stream, res_stream, n_filters_3, n_filters_1, pool_scale
):
    G = Concatenate()(
        [pool_stream, MaxPooling2D(pool_size=(pool_scale, pool_scale))(res_stream)]
    )

    net = Conv2D(n_filters_3, 3, padding="same")(G)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Conv2D(n_filters_3, 3, padding="same")(net)
    net = BatchNormalization()(net)
    pool_stream_out = Activation("relu")(net)

    net = Conv2D(n_filters_1, 1, padding="same")(pool_stream_out)
    net = Upsampling(net, scale=pool_scale)
    res_stream_out = Add()([res_stream, net])

    return pool_stream_out, res_stream_out


def SegNet(input_shape):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (3, 3), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (3, 3), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D()(conv_2)

    conv_3 = Convolution2D(128, (3, 3), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (3, 3), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D()(conv_4)

    conv_5 = Convolution2D(256, (3, 3), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (3, 3), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (3, 3), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D()(conv_7)

    conv_8 = Convolution2D(512, (3, 3), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (3, 3), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (3, 3), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D()(conv_10)

    conv_11 = Convolution2D(512, (3, 3), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (3, 3), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (3, 3), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D()(conv_13)

    # decoder

    unpool_1 = MaxUnpooling2D()([pool_5, mask_5])

    conv_14 = Convolution2D(512, (3, 3), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (3, 3), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (3, 3), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D()([conv_16, mask_4])

    conv_17 = Convolution2D(512, (3, 3), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (3, 3), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (3, 3), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D()([conv_19, mask_3])

    conv_20 = Convolution2D(256, (3, 3), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (3, 3), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (3, 3), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D()([conv_22, mask_2])

    conv_23 = Convolution2D(128, (3, 3), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (3, 3), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D()([conv_24, mask_1])

    conv_25 = Convolution2D(64, (3, 3), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(1, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)

    outputs = Activation("sigmoid")(conv_26)

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model


def FRRNA(input_shape):
    inputs = Input(shape=input_shape)
    net = Conv2D(48, 5, padding="same")(inputs)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = ResidualUnit(net, n_filters=48, filter_size=3)
    net = ResidualUnit(net, n_filters=48, filter_size=3)
    net = ResidualUnit(net, n_filters=48, filter_size=3)

    pool_stream = MaxPooling2D(pool_size=(2, 2))(net)
    res_stream = Conv2D(32, 1, padding="same")(net)

    for _ in range(3):
        pool_stream, res_stream = FullResolutionResidualUnit(
            pool_stream=pool_stream,
            res_stream=res_stream,
            n_filters_3=96,
            n_filters_1=32,
            pool_scale=2,
        )

    pool_stream = MaxPooling2D(pool_size=(2, 2))(pool_stream)

    for _ in range(4):
        pool_stream, res_stream = FullResolutionResidualUnit(
            pool_stream=pool_stream,
            res_stream=res_stream,
            n_filters_3=192,
            n_filters_1=32,
            pool_scale=4,
        )

    pool_stream = MaxPooling2D(pool_size=(2, 2))(pool_stream)

    for _ in range(2):
        pool_stream, res_stream = FullResolutionResidualUnit(
            pool_stream=pool_stream,
            res_stream=res_stream,
            n_filters_3=384,
            n_filters_1=32,
            pool_scale=8,
        )

    pool_stream = MaxPooling2D(pool_size=(2, 2))(pool_stream)

    for _ in range(2):
        pool_stream, res_stream = FullResolutionResidualUnit(
            pool_stream=pool_stream,
            res_stream=res_stream,
            n_filters_3=384,
            n_filters_1=32,
            pool_scale=16,
        )

    for _ in range(4):
        pool_stream = Upsampling(pool_stream, scale=2)

    net = Concatenate()([pool_stream, res_stream])
    net = Conv2D(48, (1, 1), activation=None)(net)
    net = ResidualUnit(net, n_filters=48, filter_size=3)
    net = ResidualUnit(net, n_filters=48, filter_size=3)
    net = ResidualUnit(net, n_filters=48, filter_size=3)

    net = Conv2D(1, (1, 1), activation=None, name="logits")(net)

    outputs = Activation("sigmoid")(net)
    model = Model(inputs=inputs, outputs=outputs, name="FRRNA")

    return model


def FCDN103(
    n_classes,
    input_shape,
    growth_rate=16,
    num_filters=48,
    num_layers_per_block=4,
    dropout_rate=0,
    weight_decay=0,
):
    inp = Input(shape=input_shape)
    x = Conv2D(
        num_filters,
        3,
        padding="same",
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(weight_decay),
    )(inp)

    skips = []
    for i, num_layers in enumerate(num_layers_per_block[:-1]):
        skips.append(create_dense_block(x, 16, num_layers, dropout_rate, weight_decay))
        x = create_transition_down(skips[i], dropout_rate, weight_decay)

    x = create_dense_block(
        x, growth_rate, num_layers_per_block[-1], dropout_rate, weight_decay
    )

    skips.reverse()
    for i, num_layers in enumerate(reversed(num_layers_per_block[:-1])):
        x = create_transition_up(x, num_layers * growth_rate, weight_decay)
        x = concatenate([x, skips[i]])
        x = create_dense_block(x, growth_rate, num_layers, dropout_rate, weight_decay)

    x = Conv2D(
        n_classes,
        1,
        padding="same",
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = Activation("sigmoid")(x)

    model = Model(inp, x, name="FCDN103")
    return model


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer="he_normal",
    )(block_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def ResNet101_deeplabV3(image_size, num_classes=1):
    model_input = Input(shape=(image_size, image_size, 3))
    preprocessed = applications.resnet50.preprocess_input(model_input)
    resnet101 = applications.ResNet101(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet101.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet101.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model_output = Activation("sigmoid")(x)
    return Model(inputs=model_input, outputs=model_output)
