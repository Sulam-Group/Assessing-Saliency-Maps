from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

# DEFINE TRANSFORMS
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
        ]
    ),
}

# LOAD IMAGES
HOME = "$YOUR_HOME_PATH"
DATA_DIR = os.path.join(HOME, "$YOUR_DATA_PATH")

usplits = ["train", "val"]
split_datasets = {
    usplit: datasets.ImageFolder(
        os.path.join(DATA_DIR, usplit), data_transforms[usplit]
    )
    for usplit in usplits
}
dataloaders = {
    usplit: torch.utils.data.DataLoader(
        split_datasets[usplit], batch_size=4, shuffle=True, num_workers=4
    )
    for usplit in usplits
}
split_sizes = {usplit: len(split_datasets[usplit]) for usplit in usplits}
class_names = split_datasets["train"].classes

# SELECT DEVICE
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# DEFINE TRAINING FUNCTION
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / split_sizes[phase]
            epoch_acc = running_corrects.double() / split_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# LOAD PRETRAINED CONVNET
model_ft = models.inception_v3(pretrained=True)
model_ft.aux_logits = False
num_ftrs = model_ft.fc.in_features
# DEFINE OUTPU NUMBER OF CLASSES
model_ft.fc = nn.Linear(num_ftrs, 2)

# LOAD THE MODEL ONTO GPU IF PRESENT, CPU IF NOT
model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)

exp_lr_echeduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=1e-1)

model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_echeduler, num_epochs=10
)

torch.save(
    model_ft.state_dict(),
    os.path.join(HOME, "$YOUR_OUTPUT_PATH"),
)
