from .neuron import ActivityState, MaturationStage, Neuron
from ..relays.hippocampus_relay import HippocampusModule


class Hippocampus:

    def __init__(self, neuron_count, features=384, input_dim=384, neurogenesis_rate=0.01):
        self.neurons = [Neuron(neuron_id=i, specialization='place_cell', abilities={'memory': 0.95},
                              maturation=MaturationStage.DIFFERENTIATED,
                              activity=ActivityState.RESTING,
                              n_features=features,
                              n_outputs=1)
                        for i in range(neuron_count)]
        # Bound outputs and add mild regularization for stability
        for n in self.neurons:
            n.nlms_head.clamp = (0.0, 1.0)
            n.nlms_head.l2 = 1e-4
        self.neurogenesis_rate = neurogenesis_rate
        self.place_cells = self.neurons
        self.relay: HippocampusModule = HippocampusModule(input_dim=input_dim, memory_strength=0.85)

    async def init_population(self):
        """Initialize all neurons with proper attach calls"""
        import trio
        async with trio.open_nursery() as n:
            for neuron in self.neurons:
                n.start_soon(neuron.attach)

    def encode(self, x):
        return self.relay.encode(x)

    def encode_memory(self, input_pattern, time=0):
        memories = []
        for neuron in self.place_cells:
            neuron.update_activity(input_pattern, time)
            if neuron.spike_history:
                memories.append(neuron.spike_history[-1][1])
            else:
                memories.append(0)
        return memories

    def stimulate_neurogenesis(self):
        new_count = int(len(self.neurons) * self.neurogenesis_rate)
        new_neurons = []
        for i in range(new_count):
            new_neuron = Neuron(neuron_id=len(self.neurons), specialization="newborn", abilities={'memory':0.8},
                                maturation=MaturationStage.PROGENITOR,
                                activity=ActivityState.RESTING,
                                n_features=10, n_outputs=1)
            self.neurons.append(new_neuron)
            new_neurons.append(new_neuron)
        return new_neurons

    async def init_weights(self):
        """Initialize weights for all neurons in the hippocampus"""
        for neuron in self.neurons:
            if hasattr(neuron, 'init_weights'):
                await neuron.init_weights()
            elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                await neuron.nlms_head.init_weights()
