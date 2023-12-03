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


def randomOutput(image_path, mask_path, model):
    input_name = sample(os.listdir(image_path), 1)[0]
    mask_name = input_name.replace(".jpg", "_L.png")
    input_image = imread(image_path + input_name)
    input_image = np.asarray([input_image])
    mask_image = imread(mask_path + mask_name)
    output_image = model.predict(input_image)[0][:, :, 0]
    return input_image, mask_image, output_image


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

    fig, axs = plt.subplots(5, 3, figsize=(12, 20))
    for i in range(5):
        prediction = randomOutput(img_path, msk_path, model)
        axs[i, 0].imshow(prediction[0][0])
        axs[i, 1].imshow(prediction[1], cmap="gray")
        axs[i, 2].imshow(prediction[2], cmap="gray")
    
    plt.subplots_adjust(bottom=0.3)
    axs[-1, 0].set_xlabel('Input', fontsize=12)
    axs[-1, 1].set_xlabel('Truth', fontsize=12)
    axs[-1, 2].set_xlabel('Output', fontsize=12)
    fig.text(0.5, 0.25, 'Custom Text', ha='center', fontsize=12)
    plt.show()
