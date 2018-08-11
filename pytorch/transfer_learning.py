from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import namedtuple

def load_data():
  data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
  }
  data_dir = "hymenoptera_data"
  image_datasets = {
      x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
      for x in ['train', 'val']}
  dataloaders = {
      x: torch.utils.data.DataLoader(
          image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
      for x in ['train', 'val']}
  data_type = namedtuple("data_type", ["classes", "training", "validation"])
  return data_type(image_datasets['train'].classes, dataloaders['train'], dataloaders['val'])

def imshow(img):
  img = img.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  img = std * img + mean
  img = np.clip(img, 0, 1)
  plt.imshow(img)
  plt.show()

def show_images(data):
  images, classes = next(iter(data.training))
  print(' '.join('%5s'%data.classes[label] for label in classes))
  imshow(torchvision.utils.make_grid(images))

def train_model(model, data, num_epochs = 25):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    for phase in ['train', 'val']:
      if phase == "train":
        scheduler.step()
        model.train()
        phase_data = data.training
      else:
        model.eval()
        phase_data = data.validation
      running_loss = 0.0
      running_corrects = 0
      count = 0.0
      for inputs, labels in phase_data:
        count += labels.shape[0]
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          if phase == 'train':
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      epoch_loss = running_loss / count
      epoch_acc = running_corrects.double() / count
      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
          phase, epoch_loss, epoch_acc))
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
  print('Best val Acc: {:4f}'.format(best_acc))
  model.load_state_dict(best_model_wts)

def visualize_model(model, data, num_images=6):
  was_training = model.training
  model.eval()
  images_so_far = 0
  fig = plt.figure()
  with torch.no_grad():
    for i, (inputs, labels) in enumerate(data.validation):
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      for j in range(inputs.size()[0]):
        images_so_far += 1
        ax = plt.subplot(num_images//2, 2, images_so_far)
        ax.axis('off')
        ax.set_title('predicted: {}'.format(data.classes[preds[j]]))
        imshow(inputs.data[j])
        if images_so_far == num_images:
          model.train(mode=was_training)
          return
  model.train(mode=was_training)

if __name__ == "__main__":
  model = models.resnet18(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 2)
  data = load_data()
# show_images(data)
  train_model(model, data)
  visualize_model(model, data)
