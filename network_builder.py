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

    def update_bias(self, bias, index = None):
        if type(bias) == list:
            # Update all neurons with individual biases
            if len(bias) != len(self.layer_neurons):
                print(f"Number of bias inputs ({len(bias)}) does not match number of neurons in layer ({len(self.layer_neurons)}).")
                raise SystemExit
            for i, n in enumerate(self.layer_neurons):
                n.update_bias(bias[i])
        elif type(bias) == float and index == None:
            # Update all neurons with the same bias
            for n in self.layer_neurons:
                n.update_bias(bias)
        elif type(bias) == float and type(index) == int:
            try:
                # Update specified neuron at index with new bias
                self.layer_neurons[index].update_bias(bias)
            except(IndexError):
                print(f"Neuron at index {index} does not exist. Please enter an index between 0 and {len(self.layer_neurons)} or leave blank to change all neurons.")
                raise SystemExit
        else:
            print("Incorrect args for update bias method. Bias must be a float or list of floats and index, if used, must be an integer value.")
            raise SystemExit

    def update_weights(self, weights, index = None):
        if index == None:
            if len(weights) != len(self.layer_neurons):
                print(f"Number of weight values do not match number of neurons. Include 1 list of weights per neuron or 'None' for no change or specify the index of the neuron you wish to change.")
                raise SystemExit
            # Update all neurons with each weight list
            for i, n in enumerate(self.layer_neurons):
                if weights[i] == None:
                    pass
                else:
                    if type(weights[i]) == list:
                        n.update_weights(weights[i])
        elif type(index) == int and type(weights[0]) == float:
            # Update specific neuron at given index with new weights
            try:
                self.layer_neurons[index].update_weights(weights)
            except(IndexError):
                print(f"Neuron at index {index} does not exist. Please enter an index between 0 and {len(self.layer_neurons)}.")
                raise SystemExit
        else:
            print("Incorrect args for update weight method. Weights must be either a list of lists and no index specified or a list of floats with an integer index.")
            raise SystemExit
    
    def update_params(self, weights, index = None):
        if index == None:
            if len(weights) != len(self.layer_neurons):
                print(f"Number of weight values do not match number of neurons. Include 1 list of weights per neuron or 'None' for no change or specify the index of the neuron you wish to change.")
                raise SystemExit
            # Update all neurons with each parameter list
            for i, n in enumerate(self.layer_neurons):
                if weights[i] == None:
                    pass
                else:
                    if type(weights[i]) == list:
                        n.update_params(weights[i])
        elif type(index) == int and type(weights[0]) == float:
            # Update specific neuron at given index with new parameters
            try:
                self.layer_neurons[index].update_params(weights)
            except(IndexError):
                print(f"Neuron at index {index} does not exist. Please enter an index between 0 and {len(self.layer_neurons)}.")
                raise SystemExit
        else:
            print("Incorrect args for update params method. Weights must be either a list of lists and no index specified or a list of floats with an integer index.")
            raise SystemExit


    # <--------------------- Forward Pass ------------------------>

    def forward(self, inputs):
        self.outputs = []
        for n in self.layer_neurons:
            self.outputs.append(n.forward(inputs))
        return self.outputs

    # <-------------------- Backpropagation ---------------------->

    def output_backward(self, targets:list, method:str):
        for n in self.layer_neurons:
            n.output_get_error(targets[self.layer_neurons.index(n)])
            n.calc_error_gradient(method)
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

    def update_bias(self, bias, layer_index = None, neuron_index = None):
        if type(bias) == float and layer_index == None and neuron_index == None:
                # Update all neurons with the same new bias
                for layer in self.layers:
                    layer.update(bias)
        elif layer_index != None and neuron_index == None:
            # Update all neurons in layer with new index
            self.layers[layer_index].update_bias(bias)
        elif type(bias) == float and layer_index != None and neuron_index == None:
            # Update specific neuron bias
            self.layers[layer_index].update_bias(bias, neuron_index)
        else:
            print("Incorrect argument format for network update bias method. The bias must be a float or list of floats and indicies can be added as integers in the order layer, neuron.")
            raise SystemExit
    
    def update_weights(self, weights:list, layer_index = None, neuron_index = None):
        if layer_index != None and neuron_index == None:
            # Update all neurons in layer with new index
            self.layers[layer_index].update_weights(weights)
        elif  layer_index != None and neuron_index == None:
            # Update specific neuron weights
            self.layers[layer_index].update_weights(weights, neuron_index)
        else:
            print("Incorrect argument format for network update weights method. The weights must be a list or array (list of lists) of floats and indicies can be added as integers in the order layer, neuron.")
            raise SystemExit
    
    def update_parameters(self, parameters:list, layer_index = None, neuron_index = None):
        if layer_index != None and neuron_index == None:
            # Update all neurons in layer with new index
            self.layers[layer_index].update_parameters(parameters)
        elif  layer_index != None and neuron_index == None:
            # Update specific neuron parameters
            self.layers[layer_index].update_parameters(parameters, neuron_index)
        else:
            print("Incorrect argument format for network update parameters method. The parameters must be a list or array (list of lists) of floats and indicies can be added as integers in the order layer, neuron.")
            raise SystemExit

    # <----------------------- Forward ----------------------->

    def forward(self, inputs:list):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.output = inputs
        return self.output

    # <------------------- Backpropagation ------------------->

    def backward(self, targets:list, learning_rate = 0.001, method = "BCE"):
        self.layers[-1].output_backward(targets, method)
        targets = self.layers[-1].layer_neurons
        i = -2
        while i + len(self.layers) >= 0:
            self.layers[i].backward(targets)
            targets = self.layers[i].layer_neurons
            i -= 1
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, data:list, targets:list, epochs = 50, learning_rate = 0.001, method = "BCE"):
        for e in range(epochs):
            combined = list(zip(data, targets))
            random.shuffle(combined)
            for inputs, target in combined:
                output = self.forward(inputs)
                self.backward(target, learning_rate, method)
            status = f"Epoch {e+1} of {epochs} Complete!"
            print(status, end="")
            print("\b"*len(status), end="")
        print(f"Epoch {epochs} of {epochs} Complete!")

    def test(self, data:list, expected_values:list):
        success = 0
        for i in range(len(data)):
            output = self.forward(data[i])
            expected = expected_values[i]
            if output == expected:
                success += 1
            print(f"{output} | {expected}")