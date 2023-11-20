import os
import numpy as np
from random import sample
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy


def load_dataset(image_path, mask_path, n_train, n_test):
    color_imgs = os.listdir(image_path)
    masks_imgs = os.listdir(mask_path)
    color_imgs.sort()
    masks_imgs.sort()

    images = []
    for image in color_imgs[:n_train]:
        data = imread(image_path + image)
        images.append(data)

    masks = []
    for image in masks_imgs[:n_train]:
        data = imread(mask_path + image)
        masks.append(data)

    train_images = np.stack(images)
    train_masks = np.stack(masks) / 255

    images = []
    for image in color_imgs[n_train : n_train + n_test]:
        data = imread(image_path + image)
        images.append(data)

    masks = []
    for image in masks_imgs[n_train : n_train + n_test]:
        data = imread(mask_path + image)
        masks.append(data)

    test_images = np.stack(images)
    test_masks = np.stack(masks) / 255

    del images, masks
    return train_images, train_masks, test_images, test_masks


# Loss Functions
binary_cross_entropy = BinaryCrossentropy()


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def analyzer(
    history, model, img_path, msk_path, test_images, test_masks, train_attr=""
):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if train_attr != "":
        plt.text(
            0.5,
            0.8,
            train_attr,
            horizontalalignment="left",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    result = model.evaluate(test_images, test_masks)
    print(f"Loss:\t\t{result[0]*100:.2f}%\nAccuracy:\t{result[1]*100:.2f}%")

    image = sample(os.listdir(img_path), 1)[0]
    mask = image.replace(".jpg", "_L.png")
    input = imread(img_path + image)
    input = np.asarray([input])
    truth = imread(msk_path + mask)
    output = model.predict(input)[0][:, :, 0]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input[0])
    axs[0].set_title("Image 1")
    axs[1].imshow(truth, cmap="gray")
    axs[1].set_title("Image 2")
    axs[2].imshow(output, cmap="gray")
    axs[2].set_title("Image 3")
    for ax in axs:
        ax.axis("off")
    plt.show()
