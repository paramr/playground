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

all_letters = string.ascii_letters
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

def categoryTensor(data, category):
  li = data.categories.index(category)
  tensor = torch.zeros(1, len(data.categories))
  tensor[0][li] = 1
  return tensor

def inputTensor(line):
  tensor = torch.zeros(len(line), 1, n_letters + 1)
  for li in range(len(line)):
    letter = line[li]
    tensor[li][0][all_letters.find(letter)] = 1
  return tensor

def targetTensor(line):
  letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
  letter_indexes.append(n_letters) # EOS
  tensor = torch.LongTensor(letter_indexes)
  tensor.unsqueeze_(-1)
  return tensor

def randomTrainingExample(data):
  category = randomChoice(data.categories)
  line = randomChoice(data.category_samples[category])
  category_tensor = categoryTensor(data, category)
  input_tensor = inputTensor(line)
  target_tensor = targetTensor(line)
  return category_tensor, input_tensor, target_tensor

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
  def __init__(self, category_size, input_size, hidden_size, output_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.i2h = nn.Linear(category_size + input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(category_size + input_size + hidden_size, output_size)
    self.o2o = nn.Linear(hidden_size + output_size, output_size)
    self.dropout = nn.Dropout(0.1)
    self.softmax = nn.LogSoftmax(dim=1)
    self.init()

  def init(self):
    self.hidden = torch.zeros(1, self.hidden_size)
    self.zero_grad()

  def forward(self, category, input):
    input_combined = torch.cat((category, input, self.hidden), 1)
    self.hidden = self.i2h(input_combined)
    output = self.i2o(input_combined)
    output_combined = torch.cat((self.hidden, output), 1)
    output = self.o2o(output_combined)
    output = self.dropout(output)
    output = self.softmax(output)
    return output

def train_net(model, data):
  criterion = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.005)
  current_loss = 0
  loss_trace = []
  for iter in range(100000):
    category_tensor, input_tensor, target_tensor = randomTrainingExample(data)
    model.init()
    loss = 0
    for i in range(input_tensor.size(0)):
      output = model(category_tensor, input_tensor[i])
      loss += criterion(output, target_tensor[i])
    loss.backward()
    optimizer.step()
    current_loss += loss.item() / input_tensor.size(0)
    if (iter + 1) % 1000 == 0:
      loss_trace.append(current_loss / 1000)
      print("Iter: %d, Loss: %f"%(iter + 1, current_loss / 1000))
      current_loss = 0.0
  plt.figure()
  plt.plot(loss_trace)
  plt.show()

def sample(model, data, category, start_letter):
  with torch.no_grad():
    category_tensor = categoryTensor(data, category)
    input = inputTensor(start_letter)
    output_name = start_letter
    for i in range(20):
      output = model(category_tensor, input[0])
      topv, topi = output.topk(1)
      topi = topi[0][0]
      if topi == n_letters:
        break
      else:
        letter = all_letters[topi]
        output_name += letter
      input = inputTensor(letter)
    return output_name

def samples(model, data, category, start_letters='ABC'):
  print("Samples for:", category)
  for start_letter in start_letters:
    print(sample(model, data, category, start_letter))


if __name__ == '__main__':
  data = load_data()
  rnn = RNN(len(data.categories), n_letters + 1, 64, n_letters + 1)
  train_net(rnn, data)
  samples(rnn, data, "Russian", "RUS")
  samples(rnn, data, "German", "GER")
  samples(rnn, data, "Spanish", "SPA")
  samples(rnn, data, "Chinese", "CHI")
