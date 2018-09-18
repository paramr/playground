import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.spa import Vocabulary

DURATION = 1.0
DIM = 20
NUM_NEURONS = 300

vocab = Vocabulary(dimensions=DIM, rng=np.random.RandomState(0))
model = nengo.Network(label="struct")
with model:
  input0 = nengo.Node(output=vocab['A'].v)
  input1 = nengo.Node(output=vocab['B'].v)
  ensemble0 = nengo.Ensemble(NUM_NEURONS, dimensions=DIM)
  ensemble1 = nengo.Ensemble(NUM_NEURONS, dimensions=DIM)
  ensemble_result = nengo.Ensemble(NUM_NEURONS, dimensions=DIM)
  ensemble_sum = nengo.Ensemble(NUM_NEURONS, dimensions=DIM)
  bind = nengo.networks.CircularConvolution(70, dimensions=DIM)

  nengo.Connection(input0, ensemble0)
  nengo.Connection(input1, ensemble1)
  nengo.Connection(ensemble0, bind.A)
  nengo.Connection(ensemble1, bind.B)
  nengo.Connection(bind.output, ensemble_result)
  nengo.Connection(ensemble0, ensemble_sum)
  nengo.Connection(ensemble1, ensemble_sum)

  input0_probe = nengo.Probe(input0)
  input1_probe = nengo.Probe(input1)
  ensemble0_probe = nengo.Probe(ensemble0, synapse=0.01)
  ensemble1_probe = nengo.Probe(ensemble1, synapse=0.01)
  ensemble_result_probe = nengo.Probe(ensemble_result, synapse=0.01)
  ensemble_sum_probe = nengo.Probe(ensemble_sum, synapse=0.01)

if __name__ == "__main__":
  with nengo.Simulator(model) as sim:
    sim.run(DURATION)

  plt.figure()
  plt.subplot(4, 1, 1)
  plt.plot(sim.trange(), sim.data[input0_probe])
  plt.plot(sim.trange(), sim.data[input1_probe])
  plt.xlim(0, DURATION)
  plt.subplot(4, 1, 2)
  plt.plot(sim.trange(), sim.data[ensemble0_probe])
  plt.plot(sim.trange(), sim.data[ensemble1_probe])
  plt.xlim(0, DURATION)
  plt.subplot(4, 1, 3)
  plt.plot(sim.trange(), sim.data[ensemble_sum_probe])
  plt.xlim(0, DURATION)
  plt.subplot(4, 1, 4)
  plt.plot(sim.trange(), sim.data[ensemble_result_probe])
  plt.xlim(0, DURATION)

  plt.show()
