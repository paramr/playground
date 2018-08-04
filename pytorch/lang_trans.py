from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
  fig, ax = plt.subplots()
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)
  plt.show()

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10
NUM_ITER = 75000
SHOW_STEP = 100
HIDDEN_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS"}
    self.n_words = 2

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

  def tensorFromSentence(self, sentence):
    indexes = [self.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def load_data(lang_name1, lang_name2, reverse):
  data_type = namedtuple("data_type", ["src_lang", "tgt_lang", "pairs"])
  lines = open('lang_data/%s-%s-small.txt' % (lang_name1, lang_name2), encoding='utf-8') \
          .read().strip().split('\n')
  pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
  pairs = [pair for pair in pairs
      if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH]
  lang1 = Lang(lang_name1)
  lang2 = Lang(lang_name2)
  for pair in pairs:
    lang1.addSentence(pair[0])
    lang2.addSentence(pair[1])
  if reverse:
    return data_type(lang2, lang1, [(pair[1], pair[0]) for pair in pairs])
  else:
    return data_type(lang1, lang2, pairs)

class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)

  def init_hidden(self):
    self.hidden = torch.zeros(1, 1, self.hidden_size, device=DEVICE)

  def forward(self, input):
    embedded = self.embedding(input).view(1, 1, -1)
    output, self.hidden = self.gru(embedded, self.hidden)
    return output

class AttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, dropout_p=0.1):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout_p = dropout_p
    self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    self.attn = nn.Linear(self.hidden_size * 2, MAX_LENGTH)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_size)

  def init_hidden(self, hidden):
    self.hidden = hidden

  def forward(self, input, encoder_outputs):
    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)
    attn_weights = F.softmax(
        self.attn(torch.cat((embedded[0], self.hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(
        attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)
    output = F.relu(output)
    output, self.hidden = self.gru(output, self.hidden)
    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, attn_weights

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  encoder.init_hidden()
  input_length = input_tensor.size(0)
  target_length = target_tensor.size(0)
  encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=DEVICE)
  for ei in range(input_length):
    encoder_output = encoder(input_tensor[ei])
    encoder_outputs[ei] = encoder_output[0, 0]
  loss = 0
  decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
  decoder.init_hidden(encoder.hidden)
  use_teacher_forcing = True if random.random() < 0.5 else False
  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_attention = decoder(
          decoder_input, encoder_outputs)
      loss += criterion(decoder_output, target_tensor[di])
      decoder_input = target_tensor[di]  # Teacher forcing
  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      decoder_output, decoder_attention = decoder(
          decoder_input, encoder_outputs)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()  # detach from history as input
      loss += criterion(decoder_output, target_tensor[di])
      if decoder_input.item() == EOS_token:
        break
  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()
  return loss.item() / target_length

def train_net(encoder, decoder, data):
  losses = []
  loss_total = 0  # Reset every 100
  learning_rate = 0.01
  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()
  training_pairs = [
      (data.src_lang.tensorFromSentence(pair[0]), data.tgt_lang.tensorFromSentence(pair[1]))
      for pair in [random.choice(data.pairs) for i in range(NUM_ITER)]]
  print("Starting training...")
  for iter in range(NUM_ITER):
    training_pair = training_pairs[iter]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]
    loss = train_step(
        input_tensor, target_tensor, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion)
    loss_total += loss
    if iter % SHOW_STEP == SHOW_STEP - 1:
      loss_avg = loss_total / SHOW_STEP
      losses.append(loss_avg)
      loss_total = 0
      i = iter + 1
      print('Iter (%d %d%%) %.4f' % (i, i / NUM_ITER * 100, loss_avg))
  print("Done training...")
  showPlot(losses)

def evaluate(encoder, decoder, data, sentence):
  with torch.no_grad():
    input_tensor = data.src_lang.tensorFromSentence(sentence)
    input_length = input_tensor.size()[0]
    encoder.init_hidden()
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=DEVICE)
    for ei in range(input_length):
      encoder_output = encoder(input_tensor[ei])
      encoder_outputs[ei] += encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    decoder.init_hidden(encoder.hidden)
    decoded_words = []
    decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)
    for di in range(MAX_LENGTH):
      decoder_output, decoder_attention = decoder(decoder_input, encoder_outputs)
      decoder_attentions[di] = decoder_attention.data
      topv, topi = decoder_output.data.topk(1)
      if topi.item() == EOS_token:
        decoded_words.append('<EOS>')
        break
      else:
        decoded_words.append(data.tgt_lang.index2word[topi.item()])
      decoder_input = topi.squeeze().detach()
    return ' '.join(decoded_words), decoder_attentions[:di + 1]

def show_attention(input_sentence, output_sentence, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_sentence.split(' '))
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def evaluate_randomly(encoder, decoder, data, n=20):
  for i in range(n):
    pair = random.choice(data.pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_sentence, attentions = evaluate(encoder, decoder, data, pair[0])
    print('<', output_sentence)
    print('')
    show_attention(pair[0], output_sentence, attentions)

if __name__ == "__main__":
  print("Using device:", DEVICE)
  print("Loading data...")
  data = load_data("eng", "fra", True)
  print("Data loaded...")
  encoder = EncoderRNN(data.src_lang.n_words, HIDDEN_SIZE).to(DEVICE)
  decoder = AttnDecoderRNN(HIDDEN_SIZE, data.tgt_lang.n_words, dropout_p=0.1).to(DEVICE)
  train_net(encoder, decoder, data)
  evaluate_randomly(encoder, decoder, data)
