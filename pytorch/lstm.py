import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

def load_data():
  training_data = [
      ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
      ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
  ]
  word_to_index = {}
  for sent, tags in training_data:
    for word in sent:
      if word not in word_to_index:
        word_to_index[word] = len(word_to_index)
  tag_to_index = {"DET": 0, "NN": 1, "V": 2}
  data_type = namedtuple("data_type", ["training", "word_to_index", "tag_to_index"])
  training = []
  for seqs, tags in training_data:
    tup = (torch.tensor([word_to_index[w] for w in seqs], dtype=torch.long),
           torch.tensor([tag_to_index[w] for w in tags], dtype=torch.long))
    training.append(tup)
  return data_type(training, word_to_index, tag_to_index)

class LSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
    super(LSTMTagger, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    self.init_hidden()

  def init_hidden(self):
    self.hidden = (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

  def forward(self, sentence):
    embeds = self.word_embeddings(sentence)
    lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
    tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    tag_scores = F.log_softmax(tag_space, dim=1)
    return tag_scores

def train_net(model, data):
  loss_function = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)
  for epoch in range(300):
    for datum in data.training:
      model.zero_grad()
      model.init_hidden()
      tag_scores = model(datum[0])
      loss = loss_function(tag_scores, datum[1])
      loss.backward()
      optimizer.step()

if __name__ == "__main__":
  data = load_data()
  model = LSTMTagger(6, 6, len(data.word_to_index), len(data.tag_to_index))
  print(model.word_embeddings(torch.LongTensor([0, 1, 2])))
  train_net(model, data)
  print(model.word_embeddings(torch.LongTensor([0, 1, 2])))


