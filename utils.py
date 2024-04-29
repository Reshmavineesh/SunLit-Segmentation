import os
import numpy as np
from random import sample
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy


def load_dataset(image_path, mask_path, n_val):
    color_imgs = os.listdir(image_path)
    masks_imgs = os.listdir(mask_path)
    color_imgs.sort()
    masks_imgs.sort()

    total_images = len(color_imgs)
    split = total_images - n_val

    images = []
    for image in color_imgs[:split]:
        data = imread(image_path + image)
        images.append(data)

    masks = []
    for image in masks_imgs[:split]:
        data = imread(mask_path + image)
        masks.append(data)

    train_images = np.stack(images)
    train_masks = np.stack(masks) / 255

    images = []
    for image in color_imgs[split:]:
        data = imread(image_path + image)
        images.append(data)

    masks = []
    for image in masks_imgs[split:]:
        data = imread(mask_path + image)
        masks.append(data)

    test_images = np.stack(images)
    test_masks = np.stack(masks) / 255

    del images, masks
    return train_images, train_masks, test_images, test_masks


# Loss Functions
binary_cross_entropy = BinaryCrossentropy()


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_loss_get_config():
    return {"name": "soft_dice"}


dice_loss.get_config = dice_loss_get_config


def randomOutput(image_path, mask_path, model):
    input_name = sample(os.listdir(image_path), 1)[0]
    mask_name = input_name.replace(".jpg", ".png")
    input_image = imread(image_path + input_name)
    input_image = np.asarray([input_image])
    mask_image = imread(mask_path + mask_name)
    output_image = model.predict(input_image)[0][:, :, 0]
    return input_image, mask_image, output_image


def analyzer(
    history, model, img_path, msk_path, test_images, test_masks, train_attr={}
):
    result = model.evaluate(test_images, test_masks)
    fig, axs = plt.subplots(6, 3, figsize=(20, 30))

    axs[0, 0].plot(history.history["loss"], label="Training Loss")
    axs[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axs[0, 0].set_title("Model Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(history.history["accuracy"], label="Training Accuracy")
    axs[0, 1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0, 1].set_title("Model Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()

    train_log = f"Loss: {result[0]:.2f}\nAccuracy: {result[1]:.2f}"
    if train_attr != {}:
        train_log += f"\nLearning Rate: {train_attr['lr']}"
        train_log += f"\nBatch Size: {train_attr['batch_size']}"
        train_log += f"\nOptimizer: {train_attr['optimizer']}"
        train_log += f"\nLoss Fn: {train_attr['loss_fn']}"
    axs[0, 2].text(0.5, 0.5, train_log, ha="right", va="center", fontsize=12)
    axs[0, 2].axis("off")

    for i in range(1, 6):
        prediction = randomOutput(img_path, msk_path, model)
        axs[i, 0].imshow(prediction[0][0])
        axs[i, 1].imshow(prediction[1], cmap="gray")
        axs[i, 2].imshow(prediction[2], cmap="gray")

    axs[-1, 0].set_xlabel("Input", fontsize=12)
    axs[-1, 1].set_xlabel("Ground Truth", fontsize=12)
    axs[-1, 2].set_xlabel("Predicted", fontsize=12)
    plt.show()
