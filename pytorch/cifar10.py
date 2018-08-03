import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

def load_data():
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_set = torchvision.datasets.CIFAR10(
      root="./cifar_data",
      train=True,
      download=True,
      transform=transform)
  train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=4,
      shuffle=True,
      num_workers=2)
  test_set = torchvision.datasets.CIFAR10(
      root="./cifar_data",
      train=False,
      download=True,
      transform=transform)
  test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=4,
      shuffle=False,
      num_workers=2)
  classes = (
      'plane', 'car', 'bird', 'cat', 'deer',
      'dog', 'frog', 'horse', 'ship', 'truck')

  data_type = namedtuple("data_type", ["classes", "training", "test"])
  return data_type(classes, train_loader, test_loader)

def imshow(img):
  img = img / 2.0 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def show_images(data):
  images, labels = next(iter(data.training))
  print(' '.join('%5s'%data.classes[label] for label in labels)
  imshow(torchvision.utils.make_grid(images))

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def train_net(net, data):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  print("Starting training")
  for epoch in range(2):
    running_loss = 0.0
    for i, train_data in enumerate(data.training, 0):
      inputs, labels = train_data
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if i % 2000 == 1999:
        print("[%d, %5d] loss: %.3f"%(epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
  print("Finished training")


def test_net(net, data):
  correct = 0
  total = 0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  print("Starting test")
  with torch.no_grad():
    for test_data in data.test:
      images, labels = test_data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      c = (predicted == labels).squeeze()
      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

  print("Accuracy: %d %%"%(100 * correct / total))
  for i in range(10):
    print("Accuracy of %5s: %2d %%"%(data.classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == "__main__":
  net = Net()
  data = load_data()
  train_net(net, data)
  test_net(net, data)
