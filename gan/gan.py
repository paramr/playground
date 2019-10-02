import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils import Logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def minst_data():
  compose = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
  out_dir = './dataset'
  return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def images_to_vectors(images):
  return images.view(images.size(0), 784)

def vectors_to_images(vectors):
  return vectors.view(vectors.size(0), 1, 28, 28)

GEN_N_FEATURES = 100

def noise(size):
  n = torch.randn(size, GEN_N_FEATURES).to(DEVICE)
  return n

def ones_target(size):
  n = torch.ones(size, 1).to(DEVICE)
  return n

def zeros_target(size):
  n = torch.zeros(size, 1).to(DEVICE)
  return n

class DiscriminatorNet(torch.nn.Module):
  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    n_features = 784
    n_out = 1
    self.hidden0 = nn.Sequential(
        nn.Linear(n_features, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
    self.hidden1 = nn.Sequential(
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
    self.hidden2 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
    self.out = nn.Sequential(
        nn.Linear(256, n_out),
        nn.Sigmoid())

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x

class GeneratorNet(torch.nn.Module):
  def __init__(self):
    super(GeneratorNet, self).__init__()
    n_out = 784
    self.hidden0 = nn.Sequential(
        nn.Linear(GEN_N_FEATURES, 256),
        nn.LeakyReLU(0.2))
    self.hidden1 = nn.Sequential(
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2))
    self.hidden2 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2))
    self.out = nn.Sequential(
        nn.Linear(1024, n_out),
        nn.Tanh())

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x

def train_discriminator(generator, discriminator, loss, optimizer, real_data):
  N = real_data.size(0)
  optimizer.zero_grad()
  # Training with real data
  prediction_real = discriminator(real_data)
  error_real = loss(prediction_real, ones_target(N))
  error_real.backward()
  # Train on Fake data
  fake_data = generator(noise(N)).detach()
  prediction_fake = discriminator(fake_data)
  error_fake = loss(prediction_fake, zeros_target(N))
  error_fake.backward()

  optimizer.step()
  return error_real + error_fake, prediction_real, prediction_fake

def train_generator(generator, discriminator, loss, optimizer, N):
  optimizer.zero_grad()
  fake_data = generator(noise(N))
  prediction = discriminator(fake_data)
  error = loss(prediction, ones_target(N))
  error.backward()
  optimizer.step()
  return error

if __name__ == '__main__':
  data = minst_data()
  data_loader = DataLoader(data, batch_size=100, shuffle=True)
  num_batches = len(data_loader)
  discriminator = DiscriminatorNet().to(DEVICE)
  generator = GeneratorNet().to(DEVICE)
  d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
  g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
  loss = nn.BCELoss()

  num_test_samples = 16
  test_noise = noise(num_test_samples)

  logger = Logger(model_name='VGAN', data_name='MNIST')
  num_epochs = 200

  for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
      N = real_batch.size(0)
      # Train Discriminator
      real_data = images_to_vectors(real_batch).to(DEVICE)
      d_error, d_pred_real, d_pred_fake = train_discriminator(
          generator, discriminator, loss, d_optimizer, real_data)
      g_error = train_generator(generator, discriminator, loss, g_optimizer, N)
      logger.log(d_error, g_error, epoch, n_batch, num_batches)
      if n_batch % 100 == 0:
        test_images = vectors_to_images(generator(test_noise)).cpu()
        logger.log_images(test_images.data, num_test_samples, epoch, n_batch, num_batches)
        logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)

