import numpy as np
import nengo
import nengo_gui
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
import math
from nengo.utils.ensemble import tuning_curves

DURATION = 2 * math.pi

model = nengo.Network(label="single_neuron")
with model:
  cos = nengo.Node(lambda t: np.cos(2.0 * t))
  neuron = nengo.Ensemble(1, dimensions=1, encoders=[[1]])
  nengo.Connection(cos, neuron)
  cos_probe = nengo.Probe(cos)
  spikes = nengo.Probe(neuron.neurons)
  voltage = nengo.Probe(neuron.neurons,'voltage')
  filtered = nengo.Probe(neuron, synapse=0.01)

if __name__ == "__main__":
  with nengo.Simulator(model) as sim:
    sim.run(DURATION)
    eval_points, activities = tuning_curves(neuron, sim)

  plt.figure()
  plt.subplot(2, 2, 1)
  plt.plot(sim.trange(), sim.data[filtered])
  plt.plot(sim.trange(), sim.data[cos_probe])
  plt.xlim(0, DURATION)

  plt.subplot(2, 2, 2)
  rasterplot(sim.trange(), sim.data[spikes])
  plt.xlim(0, DURATION)

  plt.subplot(2, 2, 3)
  plt.plot(sim.trange(), sim.data[voltage][:, 0], 'r')
  plt.xlim(0, DURATION)

  plt.subplot(2, 2, 4)
  plt.plot(eval_points, activities)
  plt.xlim(0, DURATION)

  plt.show()

