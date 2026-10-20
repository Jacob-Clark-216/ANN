import neurons

class Layer():
    def __init__(self, layer_neurons, num_inputs, neuron_type="ReLU"):
        self.layer_neurons = []
        if type(layer_neurons) == int:
            for i in range(layer_neurons):
                neuron = getattr(neurons, neuron_type)
                self.layer_neurons.append(neuron(num_inputs))
        elif type(layer_neurons) == list:
            for n in layer_neurons:
                neuron = getattr(neurons, n)
                self.layer_neurons.append(neuron(num_inputs))

    # <-------------------- Inspection Methods ----------------------->

    def __str__(self):
        output = []
        for n in self.layer_neurons:
            output.append(n.type)
        return ", ".join(output)

    def print_weights(self):
        for n in self.layer_neurons:
            print(n.weights)

    def get_weights(self):
        weights = []
        for n in self.layer_neurons:
            weights.append(n.weights)
        return weights

    def print_biases(self):
        for n in self.layer_neurons:
            print(n.bias)

    def get_biases(self):
        biases = []
        for n in self.layer_neurons:
            biases.append(n.bias)
        return biases

    # <--------------------- Forward Pass ------------------------>

    def forward(self):
        self.outputs = []
        for n in self.layer_neurons:
            self.output.append(n.forward())

    # <-------------------- Backpropagation ---------------------->

    def backward(self, layer_index, target, learning_rate):
        for n in self.layer_neurons:
            if layer_index == -1:
                n.output_get_error(target)
            else:
                n.hidden_get_error(target, self.layer_neurons.index(n))
            n.calc_error_gradient()
            n.calc_perameter_gradients()
            n.update_params(learning_rate)