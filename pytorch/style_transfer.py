from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
from collections import namedtuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMSIZE = 512 if torch.cuda.is_available() else 128
NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
RESIZE_OP = transforms.Compose([transforms.Resize(IMSIZE), transforms.ToTensor()])
TO_PIL_OP = transforms.ToPILImage()
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1000000

def load_image(image_path):
  image = Image.open(image_path)
  image = RESIZE_OP(image).unsqueeze(0)
  return image.to(DEVICE, torch.float)

def imshow(tensor, fig):
  plt.figure(fig)
  image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
  image = image.squeeze(0)      # remove the fake batch dimension
  image = TO_PIL_OP(image)
  plt.imshow(image)
  plt.pause(0.001)

def gram_matrix(input):
  b, f, h, w = input.size()
  features = input.view(b * f, h * w)
  G = torch.mm(features, features.t())
  return G.div(b * f * h * w)

class ContentLoss(nn.Module):
  def __init__(self, target):
    super(ContentLoss, self).__init__()
    self.target = target.detach()

  def forward(self, input):
    self.loss = F.mse_loss(input, self.target)
    return input

class StyleLoss(nn.Module):
  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()

  def forward(self, input):
    G = gram_matrix(input)
    self.loss = F.mse_loss(G, self.target)
    return input

class Normalization(nn.Module):
  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1, 1, 1)
    self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
    return (img - self.mean) / self.std

def create_style_model_losses(cnn, content_img, style_img):
  cnn = copy.deepcopy(cnn)
  normalization = Normalization(NORMALIZATION_MEAN, NORMALIZATION_STD).to(DEVICE)
  content_losses = []
  style_losses = []
  model = nn.Sequential(normalization)
  i = 0
  last_loss_index = 0
  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = "conv_{}".format(i)
    elif isinstance(layer, nn.ReLU):
      name = "relu_{}".format(i)
      layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
      name = "pool_{}".format(i)
    elif isinstance(layer, nn.BatchNorm2d):
      name = "bn_{}".format(i)
    else:
      raise RuntimeError("Unknown layer: {}".format(layer.__class__.__name__))
    model.add_module(name, layer)
    if name in CONTENT_LAYERS:
      last_loss_index = len(model)
      target = model(content_img).detach()
      content_loss = ContentLoss(target)
      model.add_module("content_loss_{}".format(i), content_loss)
      content_losses.append(content_loss)
    if name in STYLE_LAYERS:
      last_loss_index = len(model)
      target = model(style_img).detach()
      style_loss = StyleLoss(target)
      model.add_module("style_loss_{}".format(i), style_loss)
      style_losses.append(style_loss)
  model = model[:last_loss_index + 1]
  return model, content_losses, style_losses

if __name__ == "__main__":
  plt.ioff()
  content_img = load_image("style_transfer_data/dancing.jpg")
#  style_img = load_image("style_transfer_data/picasso.jpg")
  style_img = load_image("style_transfer_data/pintura.jpg")
  assert style_img.size() == content_img.size()
#  imshow(content_img, 0)
#  imshow(style_img, 1)
  print("Loaded images")
  cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()
  model, content_losses, style_losses = create_style_model_losses(cnn, content_img, style_img)
  input_img = content_img.clone().requires_grad_()
  optimizer = optim.LBFGS([input_img])
  print("Loaded model")
  imshow(input_img, 2)
  for iter in range(20):
    print("Iter:", iter + 1)
    def closure():
      input_img.data.clamp_(0, 1)
      optimizer.zero_grad()
      model(input_img)
      content_score = 0
      style_score = 0
      for cl in content_losses:
        content_score += cl.loss
      for sl in style_losses:
        style_score += sl.loss
      loss = CONTENT_WEIGHT * content_score + STYLE_WEIGHT * style_score
      loss.backward()
      return CONTENT_WEIGHT * content_score + STYLE_WEIGHT * style_score
    optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    imshow(input_img, 2)
  plt.show(block=True)

