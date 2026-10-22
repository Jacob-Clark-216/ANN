import math
import random

class Neuron():
    def __init__(self, num_inputs:int):
        self.weights = [random.uniform(-1, 1) for i in range(num_inputs)]
        self.bias = 0
        self.error = 0
        self.weight_gradients = []
        self.bias_gradient = 0

    def update_num_input(self, num_inputs:int):
        while len(self.weights) > num_inputs:
            self.weights.pop()
        while len(self.weights) < num_inputs:
            self.weights.append(random.random())

    def __str__(self):
        output = ""
        output += f"{self.type}\n"
        output += f"{self.bias}\n"
        output += f"{self.weights}"
        return output

    # <----------------------- Forward Pass ------------------------->

    def update_inputs(self, inputs:list):
        self.inputs = inputs

    def weighted_sum(self):
        return sum(w*x for w, x in zip(self.weights, self.inputs))

    def activation_func():
        # Abstract Method
        print("No activation_func method declared")
        raise SystemExit

    def forward(self, inputs:list):
        self.update_inputs(inputs)
        self.activation_func()
        return self.output

    # <---------------------- Backpropagation ---------------------->

    def output_get_error(self, target):
        self.error = self.output - target

    def hidden_get_error(self, next_layer_neurons:list, neuron_index:int):
        try:
            self.error = sum(next_neuron.error_gradient * next_neuron.weights[neuron_index] for next_neuron in next_layer_neurons)
        except(IndexError):
            print(neuron_index)
            print([next_neuron.weights for next_neuron in next_layer_neurons])
            raise SystemExit

    def calc_error_gradient(self):
        # Abstract Method
        print("No calc_error_gradient method declared")
        raise SystemExit

    def calc_parameter_gradients(self):
        self.bias_gradient = self.error_gradient
        self.weight_gradients = [self.error_gradient * i for i in self.inputs]

    def update_params(self, learning_rate):
        # Update weights and bias
        self.bias -= learning_rate * self.bias_gradient
        if len(self.weight_gradients) == 0:
            print("Error: weight_gradients not present. Cannot update weights.")
            raise SystemExit
        self.weights = [weight - learning_rate * weight_gradient for weight, weight_gradient in zip(self.weights, self.weight_gradients)]

# <---------------------------------------------------- Concrete Neurons ---------------------------------------------------------->

class Identity(Neuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)
        self.type = "Identity"

    def activation_func(self):
        self.output = self.weighted_sum()

    def calc_error_gradient(self, is_hidden = True):
        self.error_gradient = self.error
    
class Binary(Neuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)
        self.type = "Binary"

    def activation_func(self):
        if self.weighted_sum() >= 0:
            self.output = 1
        else:
            self.output = 0

    def calc_error_gradient(self, is_hidden = True):
        self.error_gradient = self.error


class Sigmoid(Neuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)
        self.type = "Sigmoid"

    def activation_func(self):
        self.output = 1/(1+math.e**-self.weighted_sum())

    def calc_error_gradient(self, is_hidden = True):
        if is_hidden:
            self.error_gradient = -1.0 * self.error * (self.output * (1.0 - self.output))
        else:
            self.error_gradient = self.error * (self.output * (1.0 - self.output))

class ReLU(Neuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)
        self.type = "ReLU"

    def activation_func(self):
        weighted_sum = self.weighted_sum()
        if weighted_sum > 0:
            self.output = weighted_sum
        else:
            self.output = 0

    def calc_error_gradient(self, is_hidden = True):
        if self.output > 0:
            self.error_gradient = self.error
        else:
            self.error_gradient = 0

class LeakyReLU(Neuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)
        self.type = "Leaky ReLU"

    def activation_func(self):
        weighted_sum = self.weighted_sum()
        if weighted_sum > 0:
            self.output = weighted_sum
        else:
            self.output = 0.01 * weighted_sum

    def calc_error_gradient(self, is_hidden = True):
        if self.output > 0:
            self.error_gradient = self.error
        else:
            self.error_gradient = 0.01