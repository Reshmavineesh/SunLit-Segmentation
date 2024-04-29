import torch
from src import utils
from src import resnet
from tqdm import tqdm

dataset_dir = "dataset/dataset_pistachio_128"
train_set, val_set = utils.load_dataset(dataset_dir, batch_size=16)

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet.model
model.to(DEVICE)

optimizer = resnet.optimizer("sgd", lr=0.0001, weight_decay=1e-4)
criterion = resnet.cross_entropy


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(tqdm(train_set), 0):
        inputs, masks = data
        masks = masks.squeeze(1)
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_train += resnet.accuracy(outputs, masks)
        total_train += 1

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_set), 0):
            inputs, masks = data
            masks = masks.squeeze(1)
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            outputs = model(inputs)["out"]
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            correct_val += resnet.accuracy(outputs, masks)
            total_val += 1

    val_loss = running_loss / total_val
    val_acc = correct_val / total_val
    validation_losses.append(val_loss)
    validation_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
    )
