import neurons, random

class Layer():
    def __init__(self, layer_neurons, num_inputs: int, neuron_type="LeakyReLU"):
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
            self.outputs.append(n.forward(inputs))
        return self.outputs

    # <-------------------- Backpropagation ---------------------->

    def output_backward(self, targets:list):
        for n in self.layer_neurons:
            n.output_get_error(targets[self.layer_neurons.index(n)])
            n.calc_error_gradient(False)
            n.calc_parameter_gradients()
    
    def backward(self, targets:list):
        for n in self.layer_neurons:
            n.hidden_get_error(targets, self.layer_neurons.index(n))
            n.calc_error_gradient()
            n.calc_parameter_gradients()

    def update_params(self, learning_rate:list):
        for n in self.layer_neurons:
            n.update_params(learning_rate)

# <------------------------------------------------------------- Network Class ------------------------------------------------------------->

class Network():
    def __init__(self, num_inputs:int, layers:list):
        self.layers = []
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(Layer(layers[i], num_inputs))
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
            for n in layer.layer_neurons:
                print(n.weights)
            print()

    def bias(self):
        for layer in self.layers:
            for n in layer.layer_neurons:
                print(n.bias)
            print()

    # <--------------------- Edit Layers --------------------->

    def add_output_layer(self, layer_neurons, neuron_type = "LeakyReLU"):
        self.layers.append(Layer(layer_neurons, len(self.layers[-1]), neuron_type))

    def insert_layer(self, index, layer_neurons, neuron_type = "LeakyReLU"):
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
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.output = inputs
        return self.output

    # <------------------- Backpropagation ------------------->

    def backward(self, targets:list, learning_rate = 0.001):
        self.layers[-1].output_backward(targets)
        targets = self.layers[-1].layer_neurons
        i = -2
        while i + len(self.layers) >= 0:
            self.layers[i].backward(targets)
            targets = self.layers[i].layer_neurons
            i -= 1
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, data:list, targets:list, epochs = 50, learning_rate = 0.001):
        for e in range(epochs):
            combined = list(zip(data, targets))
            random.shuffle(combined)
            for inputs, target in combined:
                output = self.forward(inputs)
                self.backward(target, learning_rate)
                print(f"Round complete! ({output} | {target})")
            print(f"Epoch {e+1} Complete!")

    def test(self, data:list, expected_values:list):
        success = 0
        for i in range(len(data)):
            output = self.forward(data[i])
            expected = expected_values[i]
            if output == expected:
                success += 1
            print(f"{output} | {expected}")