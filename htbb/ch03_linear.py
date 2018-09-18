import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
import math
from nengo.utils.functions import piecewise

DURATION = 5.0

model = nengo.Network(label="linear")
with model:
  input = nengo.Node(lambda t: [.5,-.5])
  ensemble_2d = nengo.Ensemble(50, dimensions=2, max_rates = Uniform(100, 200))
  ensemble_3d = nengo.Ensemble(50, dimensions=3, max_rates = Uniform(100, 200))
  nengo.Connection(input, ensemble_2d)
  nengo.Connection(ensemble_2d, ensemble_3d, transform=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
  input_probe = nengo.Probe(input)
  ensemble_2d_probe = nengo.Probe(ensemble_2d, synapse=0.01)
  ensemble_3d_probe = nengo.Probe(ensemble_3d, synapse=0.01)

if __name__ == "__main__":
  with nengo.Simulator(model) as sim:
    sim.run(DURATION)

  plt.figure()
  plt.subplot(3, 1, 1)
  plt.plot(sim.trange(), sim.data[input_probe])
  plt.xlim(0, DURATION)
  plt.subplot(3, 1, 2)
  plt.plot(sim.trange(), sim.data[ensemble_2d_probe])
  plt.xlim(0, DURATION)
  plt.subplot(3, 1, 3)
  plt.plot(sim.trange(), sim.data[ensemble_3d_probe])
  plt.xlim(0, DURATION)

  plt.show()
