from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters
  )

def readLines(filename):
  lines = open(filename, encoding='utf-8').read().strip().split('\n')
  return [unicodeToAscii(line) for line in lines]

def lineToTensor(line):
  tensor = torch.zeros(len(line), 1, n_letters)
  for li, letter in enumerate(line):
    index = all_letters.find(letter)
    tensor[li][0][index] = 1
  return tensor

def randomChoice(l):
  return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(data):
  category = randomChoice(data.categories)
  line = randomChoice(data.category_samples[category])
  category_tensor = torch.tensor([data.categories.index(category)], dtype=torch.long)
  line_tensor = lineToTensor(line)
  return category, line, category_tensor, line_tensor

def load_data():
  category_samples = {}
  categories = []
  for filename in glob.glob('names_data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    categories.append(category)
    lines = readLines(filename)
    category_samples[category] = lines
  data_type = namedtuple("data_type", ["categories", "category_samples"])
  return data_type(categories, category_samples)

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)
    self.init()

  def init(self):
    self.hidden = torch.zeros(1, self.hidden_size)
    self.zero_grad()

  def forward(self, input):
    combined = torch.cat((input, self.hidden), 1)
    self.hidden = self.i2h(combined)
    output = self.i2o(combined)
    output = self.softmax(output)
    return output

def train_net(model, data):
  criterion = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.005)
  current_loss = 0
  loss_trace = []
  for iter in range(100000):
    category, line, category_tensor, line_tensor = randomTrainingExample(data)
    model.init()
    for i in range(line_tensor.size()[0]):
      output = model(line_tensor[i])
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    current_loss += loss.item()
    if (iter + 1) % 1000 == 0:
      loss_trace.append(current_loss / 1000)
      print("Iter: %d, Loss: %f"%(iter + 1, current_loss / 1000))
      current_loss = 0.0
  plt.figure()
  plt.plot(loss_trace)
  plt.show()

def categoryFromOutput(data, output):
  _, top = output.topk(1)
  category = top[0].item()
  return data.categories[category], category

def model_output(model, line_tensor):
  with torch.no_grad():
    model.init()
    for i in range(line_tensor.size()[0]):
      output = model(line_tensor[i])
  return output

def evaluate(model, data):
  confusion = torch.zeros(len(data.categories), len(data.categories))
  for _ in range(10000):
    category, line, category_tensor, line_tensor = randomTrainingExample(data)
    gcat_n, gcat_i = categoryFromOutput(data, model_output(model, line_tensor))
    category_i = data.categories.index(category)
    confusion[category_i][gcat_i] += 1
  for i in range(len(data.categories)):
    confusion[i] = confusion[i] / confusion[i].sum()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(confusion.numpy())
  fig.colorbar(cax)
  ax.set_xticklabels([''] + data.categories, rotation=90)
  ax.set_yticklabels([''] + data.categories)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.show()
  return confusion

def predict(model, data, input_line, n_predictions=3):
  print('\n> %s' % input_line)
  with torch.no_grad():
    output = model_output(model, lineToTensor(input_line))
    topv, topi = output.topk(n_predictions, 1, True)
    predictions = []
    for i in range(n_predictions):
      value = topv[0][i].item()
      category_index = topi[0][i].item()
      print('(%.2f) %s' % (value, data.categories[category_index]))
      predictions.append([value, data.categories[category_index]])

if __name__ == '__main__':
  data = load_data()
  rnn = RNN(n_letters, 128, len(data.categories))
  train_net(rnn, data)
  evaluate(rnn, data)
  predict(rnn, data, 'Dovesky')
  predict(rnn, data, 'Jackson')
  predict(rnn, data, 'Satoshi')
  predict(rnn, data, 'Param')
