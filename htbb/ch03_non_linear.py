import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
import math
from nengo.utils.functions import piecewise

DURATION = 5.0

def product(x):
  return x[0] * x[1]

def square(x):
  return x[0] * x[0]

model = nengo.Network(label="sum")
with model:
  input0 = nengo.Node(piecewise({0: -0.75, 1.25: 0.50, 2.5: -0.75, 3.75: 0}))
  input1 = nengo.Node(piecewise({0:  1.00, 1.25: 0.25, 2.5: -0.25, 3.75: 0}))
  ensemble0 = nengo.Ensemble(100, dimensions=1, radius=1)
  ensemble1 = nengo.Ensemble(100, dimensions=1, radius=1)
  ensemble_inp_2d = nengo.Ensemble(100, dimensions=2, radius=2)
  ensemble_prod = nengo.Ensemble(100, dimensions=1, radius=1)
  ensemble_sqr = nengo.Ensemble(100, dimensions=1, radius=1)
  nengo.Connection(input0, ensemble0)
  nengo.Connection(input1, ensemble1)
  nengo.Connection(ensemble0, ensemble_inp_2d[0])
  nengo.Connection(ensemble1, ensemble_inp_2d[1])
  nengo.Connection(ensemble_inp_2d, ensemble_prod, function=product)
  nengo.Connection(ensemble0, ensemble_sqr, function=square)
  input0_probe = nengo.Probe(input0)
  input1_probe = nengo.Probe(input1)
  ensemble0_probe = nengo.Probe(ensemble0, synapse=0.01)
  ensemble1_probe = nengo.Probe(ensemble1, synapse=0.01)
  ensemble_prod_probe = nengo.Probe(ensemble_prod, synapse=0.01)
  ensemble_sqr_probe = nengo.Probe(ensemble_sqr, synapse=0.01)

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
  plt.plot(sim.trange(), sim.data[ensemble_prod_probe], label="prod")
  plt.plot(sim.trange(), sim.data[ensemble_sqr_probe], label="sqr")
  plt.legend()
  plt.xlim(0, DURATION)

  plt.show()
