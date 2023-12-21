import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import utils
import pickle
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras_models.models import SegNet, FRRNA, FCDN103


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="Choose model SegNet | FRRNA | FCDN103")
parser.add_argument("--dataset", type=str, help="Choose dataset Pistachio | Tomato")
parser.add_argument("--epochs", type=int, default=100, help="epochs (default 100)")
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate (default 0.001)"
)
parser.add_argument(
    "--batch-size", type=int, default=16, help="batch size (default 16)"
)

args = parser.parse_args()

model_selected = args.model
dataset_selected = args.dataset
epochs_selected = args.epochs
lr_selected = args.lr

if dataset_selected.lower() == "p128":
    image_path = "dataset/dataset_pistachio_128/color_images/"
    mask_path = "dataset/dataset_pistachio_128/masks/"
elif dataset_selected.lower() == "t128":
    image_path = "dataset/dataset_tomato_128/color_images/"
    mask_path = "dataset/dataset_tomato_128/masks/"
elif dataset_selected.lower() == "p256":
    image_path = "dataset/dataset_pistachio/color_images/"
    mask_path = "dataset/dataset_pistachio/masks/"
elif dataset_selected.lower() == "t256":
    image_path = "dataset/dataset_tomato/color_images/"
    mask_path = "dataset/dataset_tomato/masks/"
else:
    print("Invalid dataset selected: use p128 / t128 / p256 / t256")

n_val = 300
train_images, train_masks, test_images, test_masks = utils.load_dataset(
    image_path, mask_path, n_val
)
print(f"Train Images:\t{train_images.shape}")
print(f"Train Masks:\t{train_masks.shape}")
print(f"Test Images:\t{test_images.shape}")
print(f"Test Masks:\t{test_masks.shape}")


EPOCHS = epochs_selected
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
early_stopper = EarlyStopping(
    monitor="val_loss", patience=25, verbose=1, restore_best_weights=True
)
optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
loss = utils.binary_cross_entropy
metrics = ["accuracy"]
input_shape = train_images.shape[1:]
train_attr = {
    "lr": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "optimizer": optimizer.get_config()["name"],
    "loss_fn": loss.get_config()["name"],
}

if model_selected.lower() == "segnet":
    model = SegNet(input_shape)
elif model_selected.lower() == "frrna":
    model = FRRNA(input_shape)
elif model_selected.lower() == "fcdn103":
    n_layers_per_block = [4, 5, 7, 10, 12, 15]
    n_classes = 1
    model = FCDN103(
        n_classes,
        input_shape,
        num_layers_per_block=n_layers_per_block,
        dropout_rate=0.2,
        weight_decay=1e-4,
    )
else:
    print("Invalid model selected: use segnet / frrna / fcdn103")

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(
    train_images,
    train_masks,
    validation_split=0.25,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopper],
)
model.save(f'{model.name}.h5')

exports = {
    "history": history,
    "image_path": image_path,
    "mask_path": mask_path,
    "test_images": test_images,
    "test_masks": test_masks,
    "train_attr": train_attr
}
with open("model_variables.pkl", 'wb') as file:
    pickle.dump(exports, file)







'''
# Testing trained model


import pickle

with open("model_variables.pkl", 'rb') as file:
    model_vars = pickle.load(exports, file)
history = model_vars["history"]
image_path = model_vars["image_path"]
mask_path = model_vars["mask_path"]
test_images = model_vars["test_images"]
test_masks = model_vars["test_masks"]
train_attr = model_vars["train_att"]

'''