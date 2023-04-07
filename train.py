import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata

from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, DatasetDict

import flor
from flor import MTK as Flor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 2
batch_size = 8
learning_rate = 0.001

# Data loader
data = load_dataset("imagenet-1k")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {"train", "validation", "test"}  # type: ignore

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152").to(device)  # type: ignore
Flor.checkpoints(model)


def my_collate(batch):
    new_batch = []
    for i, record in enumerate(batch):
        if len(record["image"].shape) == 3:
            h, w, c = record["image"].shape
            im = record["image"].reshape((3, h, w))
        elif len(record["image"].shape) == 2:
            h, w = record["image"].shape
            im = record["image"].reshape((1, h, w))
            im = im.repeat(3, 1, 1)
        else:
            raise

        im = feature_extractor(im, return_tensors="pt")["pixel_values"]
        new_batch.append(
            {
                "label": record["label"],
                "image": im,
            }
        )

    out = torchdata.default_collate(new_batch)
    out["image"] = out["image"].reshape(batch_size, 3, 224, 224)
    return out


train_loader = torchdata.DataLoader(dataset=data["train"].with_format("torch"), batch_size=batch_size, shuffle=True, collate_fn=my_collate)  # type: ignore
val_loader = torchdata.DataLoader(dataset=data["validation"].with_format("torch"), batch_size=batch_size, shuffle=False, collate_fn=my_collate)  # type: ignore
test_loader = torchdata.DataLoader(dataset=data["test"].with_format("torch"), batch_size=batch_size, shuffle=False, collate_fn=my_collate)  # type: ignore

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Flor.checkpoints(optimizer)

# Train the model
total_step = len(train_loader)
num_steps = 1000
for epoch in Flor.loop(range(num_epochs)):
    model.train()
    for i, batch in Flor.loop(enumerate(train_loader)):
        # Move tensors to the configured device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i, num_steps, flor.log("loss", loss.item())
                )
            )
            if i == num_steps:
                # bootleg sampling
                break

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        print(f"evaluating for 1000 rounds")
        for i, batch in enumerate(val_loader):
            # Move tensors to the configured device
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)
            left, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 0:
                print(i, correct, total)
                if i == 1000:
                    break

        print(
            "Accuracy of the network on the 8000 test images: {} %".format(
                flor.log("val_acc", 100 * correct / total)
            )
        )

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    print(f"evaluating for {len(val_loader)} rounds")
    for i, batch in enumerate(val_loader):
        # Move tensors to the configured device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(images)
        left, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 0:
            print(i, correct, total)

    print(
        f"Accuracy of the network on the {len(val_loader) * batch_size} test images: {flor.log('acc', 100 * correct / total)}"
    )
