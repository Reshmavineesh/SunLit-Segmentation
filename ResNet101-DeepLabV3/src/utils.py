import os
import torch
from PIL import Image
from skimage.io import imread
import matplotlib.pyplot as plt
import torch.nn.functional as nnf
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader


class LeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.color_images = os.listdir(os.path.join(root_dir, "color_images"))

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "color_images", self.color_images[idx])
        mask_name = os.path.join(
            self.root_dir, "masks", self.color_images[idx].replace(".jpg", "_L.png")
        )
        image = Image.open(img_name)
        mask = Image.open(mask_name)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.tensor(mask, dtype=torch.int64)
        return image, mask


def load_dataset(dataset_root, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LeafDataset(dataset_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_data_loader, val_data_loader


def test_model(model, image_path, mask_path, transform, DEVICE):
    input_image = Image.open(image_path)
    input_image = transform(input_image).unsqueeze(0)
    input_image = input_image.to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(input_image)["out"]
    softmax = nnf.softmax(output, dim=1)

    _, predicted_class = torch.max(softmax, 1)

    color_mapping = [0, 255]
    predicted_mask = predicted_class[0].cpu().numpy()
    segmented_image = Image.fromarray((predicted_mask * 255).astype("uint8"))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Color Image")
    plt.imshow(imread(image_path))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Original Mask")
    plt.imshow(imread(mask_path), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmentation Output")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")
    plt.show()
