import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

EMBEDDING_DIM = 10

def load_data():
  test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
  vocab = set(test_sentence)
  word_to_index = {word: i for i, word in enumerate(vocab)}
  cbow_data = [
      ([test_sentence[i-2], test_sentence[i-1],
        test_sentence[i+1], test_sentence[i+2]],
        test_sentence[i])
          for i in range(2, len(test_sentence) - 2)]
  data_type = namedtuple("data_type", ["vocabulary", "word_to_index", "cbow_data"])
  return data_type(vocab, word_to_index, cbow_data)


class CBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size):
    super(CBOW, self).__init__()
    self.context_size = context_size
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear = nn.Linear(embedding_dim, vocab_size)

  def forward(self, inputs):
    embeds = self.embeddings(inputs)
    embeds = embeds.unsqueeze(0)
    out = torch.sum(embeds, dim=1) / self.context_size
    out = self.linear(out)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs

def make_context_vector(context, data):
    idxs = [data.word_to_index[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def train_net(model, data):
  loss_function = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)
  for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in data.cbow_data:
      context_idxs = torch.tensor([data.word_to_index[w] for w in context], dtype=torch.long)
      model.zero_grad()
      log_probs = model(context_idxs)
      loss = loss_function(log_probs, torch.tensor([data.word_to_index[target]], dtype=torch.long))
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print("Loss at epoch {}: {}".format(epoch, total_loss))

if __name__ == "__main__":
  data = load_data()
  model = CBOW(len(data.vocabulary), EMBEDDING_DIM, 4)
  print(model.embeddings(torch.LongTensor([0, 1, 2])))
  train_net(model, data)
  print(model.embeddings(torch.LongTensor([0, 1, 2])))
