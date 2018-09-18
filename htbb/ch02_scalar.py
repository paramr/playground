import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
import math
from nengo.utils.ensemble import tuning_curves

DURATION = 2 * math.pi

model = nengo.Network(label="scalar")
with model:
  input = nengo.Node(output=np.sin)
  ensemble = nengo.Ensemble(25, dimensions=1, max_rates = Uniform(100, 200))
  nengo.Connection(input, ensemble)
  input_probe = nengo.Probe(input)
  spikes_probe = nengo.Probe(ensemble.neurons)
  ensemble_probe = nengo.Probe(ensemble, synapse=0.01)

if __name__ == "__main__":
  with nengo.Simulator(model) as sim:
    sim.run(DURATION)
    eval_points, activities = tuning_curves(ensemble, sim)

  plt.figure()
  plt.plot(sim.trange(), sim.data[ensemble_probe])
  plt.plot(sim.trange(), sim.data[input_probe])
  plt.xlim(0, DURATION)

  plt.figure()
  rasterplot(sim.trange(), sim.data[spikes_probe])
  plt.xlim(0, DURATION)

  plt.figure()
  plt.plot(eval_points, activities)
  plt.xlim(-1, 1)

  plt.show()
