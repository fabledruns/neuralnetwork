from nn.network import Network, Layer, Neuron
from nn.loss import MSE

def loss(network, inputs, target):
    mse = MSE()
    prediction = network.forward(inputs)
    return mse.forward(prediction, target)

"""
def numerical_gradient(neuron, weight_index, epsilon = 0.001):
    og_weight = neuron.weights[weight_index]

    neuron.weights[weight_index] = og_weight + epsilon
    loss_plus  = loss(network, [8, 90], [10, 20, 30])

    neuron.weights[weight_index] = og_weight - epsilon
    loss_minus = loss(network, [8, 90], [10, 20, 30])

    gradient = (loss_plus - loss_minus) / (2 * epsilon)
    neuron.weights[weight_index] = og_weight

    return gradient
"""

def update_weights(network, learning_rate):
    for layer in network.layers:
        for neuron in layer.neurons:
            for i in range(len(neuron.weights)):
                neuron.weights[i] -= (learning_rate * neuron.grad_weights[i])

            neuron.bias -= (learning_rate * neuron.grad_bias)

network = Network([
    Layer([
        Neuron([0.02, -0.05], 0),
        Neuron([-0.08, 0.04], 0),
        Neuron([0.01, 0.09], 0)
    ]),
    Layer([
        Neuron([0.03, -0.07, 0.05], 0),
        Neuron([-0.02, 0.06, -0.01], 0),
        Neuron([0.04, -0.03, 0.02], 0)
    ])
])

test = [8, 90]

prediction = network.forward(test)

target = [10, 20, 30]

current_loss = loss(network, [8,90], [10,20,30])

print(f"Prediction: {prediction}")

mse = MSE()

for epoch in range(100):
    prediction = network.forward(test)

    loss32 = mse.forward(prediction, target)

    grad_outputs = mse.backward()

    network.backward(grad_outputs)
    for i, neuron in enumerate(network.layers[0].neurons):
        print(i, neuron.grad_weights)

    update_weights(network, 0.0001)

    print(f"Epoch {epoch}: Loss = {loss32}")

print(network.forward(test))

print("Hidden outputs:")
for neuron in network.layers[0].neurons:
    print(neuron.output)