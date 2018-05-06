import numpy as np

batch_size = 64
input_dimensions = 1000
hidden_layers = 100
output_dimensions = 10

random_input = np.random.randn(batch_size,input_dimensions)
random_output = np.random.randn(batch_size,output_dimensions)

w1 = np.random.randn(input_dimensions, hidden_layers)
w2 = np.random.randn(hidden_layers, output_dimensions)

learning_rate = 1e-6

for t in range(500):
    # Forward pass
    hidden_layers = random_input.dot(w1)
    hidden_layers_relu = np.maximum(hidden_layers, 0)
    output_prediction = hidden_layers_relu.dot(w2)
    
    # Compute Loss
    loss = np.square(output_prediction - random_output).sum()
    print(t, loss)
    
    # Backprop to compute gradients of w1 and w2 with respect to loss
    output_prediction_gradient = 2.0 * (output_prediction - random_output)
    w2_gradient = hidden_layers_relu.T.dot(output_prediction_gradient)
    hidden_layers_relu_gradient = output_prediction_gradient.dot(w2.T)
    hidden_layers_gradient = hidden_layers_relu_gradient.copy()
    hidden_layers_gradient[hidden_layers < 0] = 0
    w1_gradient = random_input.T.dot(hidden_layers_gradient)
    
    #Update weights
    
    w1 -= learning_rate * w1_gradient
    w2 -= learning_rate * w2_gradient
    