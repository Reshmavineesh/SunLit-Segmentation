from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Conv2D
from tensorflow.keras.layers import concatenate, MaxPool2D, Input, Conv2DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def create_layer(x, filters, dropout_rate=0.2, weight_decay=1e-4):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=3, kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    return x

def create_dense_block(x, growth_rate=16, num_layers=4, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(num_layers):
        l = create_layer(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, l])
    return x

def create_transition_down(x, dropout_rate=0.2, weight_decay=1e-4):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(x.get_shape().as_list()[-1], 1, kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPool2D()(x)
    return x

def create_transition_up(x, filters, weight_decay=1e-4):
    x = Conv2DTranspose(filters, 3, strides=(2, 2), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    return x

def FCDN103(n_classes, input_shape, growth_rate=16, num_filters=48, num_layers_per_block=4, dropout_rate=0, weight_decay=0):
    inp = Input(shape=input_shape)
    x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(inp)

    skips = []
    for i, num_layers in enumerate(num_layers_per_block[:-1]):
        skips.append(create_dense_block(x, 16, num_layers, dropout_rate, weight_decay))
        x = create_transition_down(skips[i], dropout_rate, weight_decay)
    
    x = create_dense_block(x, growth_rate, num_layers_per_block[-1], dropout_rate, weight_decay)
    
    skips.reverse()
    for i, num_layers in enumerate(reversed(num_layers_per_block[:-1])):
        x = create_transition_up(x, num_layers * growth_rate, weight_decay)
        x = concatenate([x, skips[i]])
        x = create_dense_block(x, growth_rate, num_layers, dropout_rate, weight_decay)
    
    x = Conv2D(n_classes, 1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    x = Activation('sigmoid')(x)
    
    model = Model(inp, x)
    return model
