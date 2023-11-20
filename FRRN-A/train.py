import os
from src import utils
import tensorflow as tf
from src.frrna import FRRNA


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


EPOCHS = 40
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
loss = utils.binary_cross_entropy
metrics = ["accuracy"]
input_shape = train_images.shape[1:]

train_attr = f"""
Learning rate: {LEARNING_RATE}
Batch Size: {BATCH_SIZE}
Optimizer: {optimizer.get_config()['name']}
Loss Fn: {loss.get_config()['name']}
"""

model = FRRNA(input_shape)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit(
    train_images,
    train_masks,
    validation_split=0.25,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)
