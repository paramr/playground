import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
import math

DURATION = 3 * math.pi

model = nengo.Network(label="vector")
with model:
  input0 = nengo.Node(output=np.sin)
  input1 = nengo.Node(output=np.cos)
  ensemble = nengo.Ensemble(25, dimensions=2, max_rates = Uniform(100, 200))
  nengo.Connection(input0, ensemble[0])
  nengo.Connection(input1, ensemble[1])
  input0_probe = nengo.Probe(input0)
  input1_probe = nengo.Probe(input1)
  spikes_probe = nengo.Probe(ensemble.neurons)
  ensemble_probe = nengo.Probe(ensemble, synapse=0.01)

if __name__ == "__main__":
  with nengo.Simulator(model) as sim:
    sim.run(DURATION)

  plt.figure()
  plt.plot(sim.trange(), sim.data[ensemble_probe])
  plt.plot(sim.trange(), sim.data[input0_probe])
  plt.plot(sim.trange(), sim.data[input1_probe])
  plt.xlim(0, DURATION)

  plt.figure()
  rasterplot(sim.trange(), sim.data[spikes_probe])
  plt.xlim(0, DURATION)

  plt.show()
