import os
from src import utils
import tensorflow as tf
from src.FCDense103 import FCDN103


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


image_path = "dataset/dataset_pistachio/color_images/"
mask_path = "dataset/dataset_pistachio/masks/"
n_train = 1100
n_test = 125

train_images, train_masks, test_images, test_masks = utils.load_dataset(
    image_path, mask_path, n_train, n_test
)
print(f"Train Images:\t{train_images.shape}")
print(f"Train Masks:\t{train_masks.shape}")
print(f"Test Images:\t{test_images.shape}")
print(f"Test Masks:\t{test_masks.shape}")


EPOCHS = 60
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
loss = utils.binary_cross_entropy
metrics = ["accuracy"]
input_shape = train_images.shape[1:]
n_layers_per_block = [4, 5, 7, 10, 12, 15]
n_classes = 1

train_attr = f"""
Learning rate: {LEARNING_RATE}
Batch Size: {BATCH_SIZE}
Optimizer: {optimizer.get_config()['name']}
Loss Fn: {loss.get_config()['name']}
"""

model = FCDN103(
    n_classes,
    input_shape,
    num_layers_per_block=n_layers_per_block,
    dropout_rate=0.2,
    weight_decay=1e-4,
)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit(
    train_images,
    train_masks,
    validation_split=0.25,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)
