import neurons

class Layer():
    def __init__(self, layer_neurons, num_inputs: int, neuron_type="ReLU"):
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

    def __len__(self):
        return len(self.layer_neurons)

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

    # <--------------------- Update Methods --------------------->

    def update_num_inputs(self, new_num_inputs):
        for n in self.layer_neurons:
            n.update_num_input(new_num_inputs)

    # <--------------------- Forward Pass ------------------------>

    def forward(self, inputs):
        self.outputs = []
        for n in self.layer_neurons:
            self.output.append(n.forward(inputs))
        return self.outputs

    def input_forward(self, inputs):
        self.outputs = []
        for i, n in zip(inputs, self.layer_neurons):
            self.output.append(n.forward(i))
        return self.outputs

    # <-------------------- Backpropagation ---------------------->

    def backward(self, layer_index:int, targets:list, learning_rate):
        for n in self.layer_neurons:
            if layer_index == -1:
                n.output_get_error(targets[self.layer_neurons.index(n)])
            else:
                n.hidden_get_error(targets, self.layer_neurons.index(n))
            n.calc_error_gradient()
            n.calc_perameter_gradients()
            n.update_params(learning_rate)

# <------------------------------------------------------------- Network Class ------------------------------------------------------------->

class Network():
    def __init__(self, layers:list):
        self.layers = []
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(Layer(layers[i], 0))
            else:
                self.layers.append(Layer(layers[i], len(self.layers[-1])))

    # <------------------ Inspection Methods ------------------>

    def __str__(self):
        output = ""
        for layer in self.layers:
            output += str(layer) + "\n"
        return output

    def weights(self):
        for layer in self.layers:
            for n in layer:
                print(n.weights())
            print()

    def bias(self):
        for layer in self.layers:
            for n in layer:
                print(n.bias())
            print()

    # <--------------------- Edit Layers --------------------->

    def add_output_layer(self, layer_neurons, neuron_type = "ReLU"):
        self.layers.append(Layer(layer_neurons, len(self.layers[-1]), neuron_type))

    def insert_layer(self, index, layer_neurons, neuron_type = "ReLU"):
        if index > len(self.layers):
            self.add_output_layer(index, layer_neurons, neuron_type)
        elif index == 0:
            self.layers.insert(0, 1, neuron_type)
            for layer in self.layers[1:]:
                layer.update_num_inputs(layer_neurons)
                layer_neurons = len(layer)
        else:
            self.layers.insert(index, Layer(layer_neurons, len(self.layers[index-1]), neuron_type))
            for layer in self.layers[index+1:]:
                layer.update_num_inputs(layer_neurons)
                layer_neurons = len(layer)

    # <----------------------- Forward ----------------------->

    def forward(self, inputs:list):
        self.inputs = inputs
        self.inputs = self.layers[0].input_forward(inputs)
        for layer in self.layers[1:]:
            self.inputs = layer.forward(inputs)
        self.output = self.inputs
        return self.output

    # <------------------- Backpropagation ------------------->

    def backward(self, target:list, learning_rate = 0.001):
        i = -1
        while i + len(self.layers) != -1:
            self.layers[i].backward(i+len(self.layers), target, learning_rate)
            target = self.layers[i]
            i -= 1

    def train(self, data:list, targets:list, learning_rate = 0.001):
        for i in range(len(data)):
            self.forward(data[i])
            self.backward(targets[i], learning_rate)
            print(f"Round {i} (of {len(data)}) complete!")