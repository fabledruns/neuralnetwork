class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias
    
    def relu(self, x):
        return x if x > 0 else 0.01 * x
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        self.output = self.relu(self.z)
        
        return self.output
    
    def backward(self, grad_output):
        grad_relu = 1 if self.z > 0 else 0.01

        grad_z = grad_output * grad_relu
        self.grad_weights = [grad_z * input for input in self.inputs]
        self.grad_bias = grad_z
        grad_inputs = [grad_z * weight for weight in self.weights]

        return grad_inputs
    
class Layer:
    def __init__(self, neurons) -> None:
        self.neurons = neurons
    
    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]
    
    def backward(self, grad_outputs):
        grad_inputs = [0] * len(self.neurons[0].inputs)

        for neuron, grad in zip(self.neurons, grad_outputs):
            returned = neuron.backward(grad)

            for i in range(len(grad_inputs)):
                grad_inputs[i] += returned[i]

        return grad_inputs
    
class Network:
    def __init__(self, layers) -> None:
        self.layers = layers
    
    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_outputs):
        grads = grad_outputs

        for layer in reversed(self.layers):
            grads = layer.backward(grads)
            
        return grads