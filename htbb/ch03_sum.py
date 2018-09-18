import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
import math
from nengo.utils.functions import piecewise

DURATION = 5.0

model = nengo.Network(label="sum")
with model:
  input0 = nengo.Node(piecewise({0: -0.75, 1.25: 0.5, 2.5: 0.70, 3.75: 0}))
  input1 = nengo.Node(piecewise({0: 0.25, 1.25: -0.5, 2.5: 0.85, 3.75: 0}))
  ensemble0 = nengo.Ensemble(100, dimensions=1, max_rates = Uniform(100, 200))
  ensemble1 = nengo.Ensemble(100, dimensions=1, max_rates = Uniform(100, 200))
  ensemble_sum = nengo.Ensemble(100, dimensions=1, max_rates = Uniform(100, 200))
  nengo.Connection(input0, ensemble0)
  nengo.Connection(input1, ensemble1)
  nengo.Connection(ensemble0, ensemble_sum)
  nengo.Connection(ensemble1, ensemble_sum)
  input0_probe = nengo.Probe(input0)
  input1_probe = nengo.Probe(input1)
  ensemble0_probe = nengo.Probe(ensemble0, synapse=0.01)
  ensemble1_probe = nengo.Probe(ensemble1, synapse=0.01)
  ensemble_sum_probe = nengo.Probe(ensemble_sum, synapse=0.01)

if __name__ == "__main__":
  with nengo.Simulator(model) as sim:
    sim.run(DURATION)

  plt.figure()
  plt.subplot(3, 1, 1)
  plt.plot(sim.trange(), sim.data[input0_probe])
  plt.plot(sim.trange(), sim.data[input1_probe])
  plt.xlim(0, DURATION)
  plt.subplot(3, 1, 2)
  plt.plot(sim.trange(), sim.data[ensemble0_probe])
  plt.plot(sim.trange(), sim.data[ensemble1_probe])
  plt.xlim(0, DURATION)
  plt.subplot(3, 1, 3)
  plt.plot(sim.trange(), sim.data[ensemble_sum_probe])
  plt.xlim(0, DURATION)

  plt.show()
