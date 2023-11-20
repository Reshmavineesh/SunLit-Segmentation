from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D, Concatenate, MaxPooling2D


def Upsampling(inputs, scale):
    return UpSampling2D(size=(scale, scale))(inputs)


def ResidualUnit(inputs, n_filters=48, filter_size=3):
    net = Conv2D(n_filters, filter_size, padding="same")(inputs)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Conv2D(n_filters, filter_size, padding="same")(net)
    net = BatchNormalization()(net)
    return Add()([inputs, net])


def FullResolutionResidualUnit(pool_stream, res_stream, n_filters_3, n_filters_1, pool_scale):    
    G = Concatenate()([pool_stream, MaxPooling2D(pool_size=(pool_scale, pool_scale))(res_stream)])

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
    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model