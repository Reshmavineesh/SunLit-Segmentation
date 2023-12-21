from keras_models.models import SegNet, FRRNA, FCDN103, ResNet101_deeplabV3

input_shape = (256, 256, 3)
n_layers_per_block = [4, 5, 7, 10, 12, 15]
n_classes = 1

# Segnet
model = SegNet(input_shape)
print(model.summary())

# FRRNA
model = FRRNA(input_shape)
print(model.summary())


# FCDN103
model = FCDN103(
    n_classes,
    input_shape,
    num_layers_per_block=n_layers_per_block,
    dropout_rate=0.2,
    weight_decay=1e-4,
)
print(model.summary())


# DeepLab Resnet10
model = ResNet101_deeplabV3(input_shape[0])
print(model.summary())
