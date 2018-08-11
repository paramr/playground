import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_op = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
SCREEN_WIDTH = 600
VIEW_WIDTH = 320
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, *args):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

class DQN(nn.Module):
  def __init__(self):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)
    self.head = nn.Linear(448, 2)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    return self.head(x.view(x.size(0), -1))

def get_cart_location_slice(env):
  world_width = env.x_threshold * 2
  scale = SCREEN_WIDTH / world_width
  cart_location = int(env.state[0] * scale + SCREEN_WIDTH / 2.0)
  if cart_location < VIEW_WIDTH // 2:
    slice_range = slice(VIEW_WIDTH)
  elif cart_location > (SCREEN_WIDTH - VIEW_WIDTH) // 2:
    slice_range = slice(-VIEW_WIDTH, None)
  else:
    slice_range = slice(cart_location - VIEW_WIDTH // 2, cart_location + VIEW_WIDTH // 2)
  return cart_location, slice_range

def get_screen(env):
  screen = env.render(mode='rgb_array').transpose((2, 0, 1)) # to CHW
  screen = screen[:, 160:320] # Remove top and bottom
  cart_location, slice_range = get_cart_location_slice(env)
  screen = screen[:, :, slice_range]
  screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
  screen = torch.from_numpy(screen)
  return resize_op(screen).unsqueeze(0).to(DEVICE) # BCHW

steps_done = 0
def select_action(policy_net, state):
  global steps_done
  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
  steps_done += 1
  if sample > eps_threshold:
    with torch.no_grad():
      return policy_net(state).max(1)[1].view(1, 1)
  else:
    return torch.tensor([[random.randrange(2)]], device=DEVICE, dtype=torch.long)

def plot_durations(episode_durations):
  plt.figure(2)
  plt.clf()
  plt.xlabel('Episode')
  plt.ylabel('Duration')
  plt.plot(episode_durations)
  durations_t = torch.tensor(episode_durations, dtype=torch.float)
  if len(durations_t) >= 100:
    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy())
  plt.pause(0.001)

def optimize_model(memory, policy_net, target_net, optimizer):
  if len(memory) < BATCH_SIZE:
    return
  transitions = memory.sample(BATCH_SIZE)
  # http://stackoverflow.com/a/19343/3343043
  batch = Transition(*zip(*transitions))
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.uint8)
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  state_action_values = policy_net(state_batch).gather(1, action_batch)
  next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
  next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch
  loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # Huber loss
  optimizer.zero_grad()
  loss.backward()
  for param in policy_net.parameters():
    param.grad.clamp_(-1, 1)
  optimizer.step()

if __name__ == "__main__":
  policy_net = DQN().to(DEVICE)
  target_net = DQN().to(DEVICE)
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval() # set to evaluation mode :-/
  optimizer = optim.RMSprop(policy_net.parameters())
  memory = ReplayMemory(10000)
  episode_durations = []
  env = gym.make('CartPole-v0').unwrapped
  for i_e in range(5000):
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen
    for t in count():
      action = select_action(policy_net, state)
      _, reward, done, _ = env.step(action.item())
      reward = torch.tensor([reward], device=DEVICE)
      last_screen = current_screen
      current_screen = get_screen(env)
      if not done:
        next_state = current_screen - last_screen
      else:
        next_state = None
      memory.push(state, action, next_state, reward)
      state = next_state
      optimize_model(memory, policy_net, target_net, optimizer)
      if done:
        episode_durations.append(t + 1)
        plot_durations(episode_durations)
        break
    if i_e % TARGET_UPDATE == 0:
      target_net.load_state_dict(policy_net.state_dict())
  env.render()
  env.close()
  plt.show()
