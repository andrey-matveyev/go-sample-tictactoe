package main

import (
	"math"
	"math/rand"
)

// --- Neural network structures (with Clone method) ---

// NeuralNetwork represents a complete neural network.
type NeuralNetwork struct {
	Layers []*NeuralNetworkLayer
}

// NewNeuralNetwork creates a new neural network with combined layers.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	item := &NeuralNetwork{}

	currentInputSize := inputSize
	for _, hs := range hiddenSizes {
		// Each hidden layer is a single NeuralNetworkLayer
		item.Layers = append(item.Layers, NewNeuralNetworkLayer(currentInputSize, hs, activation))
		currentInputSize = hs
	}

	// Output layer with activation
	item.Layers = append(item.Layers, NewNeuralNetworkLayer(currentInputSize, outputSize, activation))

	return item
}

// Predict performs the forward pass to get network predictions.
func (item *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range item.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Train performs one training step for the network.
// input: input data
// targetOutput: target output data (Q-values for training)
// learningRate: learning rate
func (item *NeuralNetwork) Train(input []float64, targetOutput []float64, learningRate float64) {
	// Forward pass (saving intermediate values)
	predictedOutput := item.Predict(input)

	// Calculate output gradient (MSE loss derivative)
	// dLoss/dOutput = 2 * (predicted - target)
	outputGradient := make([]float64, len(predictedOutput))
	for i := range predictedOutput {
		outputGradient[i] = 2 * (predictedOutput[i] - targetOutput[i])
	}

	// Backward pass
	currentGradient := outputGradient
	for i := len(item.Layers) - 1; i >= 0; i-- {
		currentGradient = item.Layers[i].Backward(currentGradient)
	}

	// Update weights
	for _, layer := range item.Layers {
		layer.Update(learningRate)
	}
}

// Clone creates a deep copy of the neural network. This is important for the DQN target network.
func (nn *NeuralNetwork) Clone() *NeuralNetwork {
	clone := &NeuralNetwork{
		Layers: make([]*NeuralNetworkLayer, len(nn.Layers)),
	}
	for i, layer := range nn.Layers {
		newLayer := &NeuralNetworkLayer{
			InputSize:      layer.InputSize,
			OutputSize:     layer.OutputSize,
			Weights:        make([][]float64, len(layer.Weights)),
			Biases:         make([]float64, len(layer.Biases)),
			ActivationFunc: layer.ActivationFunc,
			DerivativeFunc: layer.DerivativeFunc,
		}
		for r := range layer.Weights {
			newLayer.Weights[r] = make([]float64, len(layer.Weights[r]))
			copy(newLayer.Weights[r], layer.Weights[r])
		}
		copy(newLayer.Biases, layer.Biases)
		clone.Layers[i] = newLayer
	}
	return clone
}

// --- Combined neural network layer structure ---

// NeuralNetworkLayer represents one fully connected layer with an activation function.
type NeuralNetworkLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64

	// Temporary values for backpropagation
	InputVector  []float64 // Input values to the layer (from the previous layer)
	WeightedSums []float64 // Values after linear transformation (before activation)
	OutputVector []float64 // Output values after activation

	// Gradients for updating weights and biases
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Gradient passed to the previous layer
}

// NewNeuralNetworkLayer creates a new fully connected layer with an activation function.
func NewNeuralNetworkLayer(inputSize, outputSize int, activationName string) *NeuralNetworkLayer {
	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		// Initializing weights with random values
		for j := range weights[i] { 
			weights[i][j] = rand.NormFloat64() * math.Sqrt(1.0/float64(inputSize))
		}
		biases[i] = 0.0
	}

	layer := &NeuralNetworkLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    weights,
		Biases:     biases,
	}

	switch activationName {
	case "tanh": // Tanh activation added
		layer.ActivationFunc = Tanh
		layer.DerivativeFunc = TanhDerivative
	case "none": // For output layer without activation
		layer.ActivationFunc = func(x float64) float64 { return x }
		layer.DerivativeFunc = func(x float64) float64 { return 1.0 }
	default:
		panic("Unknown activation function: " + activationName)
	}

	return layer
}

// Forward performs the forward pass through the layer (linear part + activation).
func (item *NeuralNetworkLayer) Forward(input []float64) []float64 {
	item.InputVector = input
	// 1. Linear transformation
	item.WeightedSums = MultiplyMatrixVector(item.Weights, input)
	item.WeightedSums = AddVectors(item.WeightedSums, item.Biases)

	// 2. Activation
	item.OutputVector = make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		item.OutputVector[i] = item.ActivationFunc(item.WeightedSums[i])
	}
	return item.OutputVector
}

// Backward performs the backward pass through the layer.
func (item *NeuralNetworkLayer) Backward(outputGradient []float64) []float64 {
	// 1. Gradient through the activation function (apply activation derivative to WeightedSums)
	activationGradient := make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		activationGradient[i] = item.DerivativeFunc(item.WeightedSums[i])
	}
	// Combine the gradient from the next layer with the activation gradient (element-wise multiplication)
	gradientAfterActivation := MultiplyVectors(outputGradient, activationGradient)

	// 2. Gradient for biases is equal to the gradient after activation
	item.BiasGradients = gradientAfterActivation

	// 3. Gradient for weights = outer product (Input X gradientAfterActivation)
	item.WeightGradients = OuterProduct(gradientAfterActivation, item.InputVector)

	// 4. Gradient for input = TransposedWeights * gradientAfterActivation
	transposedWeights := TransposeMatrix(item.Weights)
	item.InputGradient = MultiplyMatrixVector(transposedWeights, gradientAfterActivation)

	return item.InputGradient
}

// Update - Updates the layer's weights and biases.
func (item *NeuralNetworkLayer) Update(learningRate float64) {
	// Update weights
	for i := range item.Weights {
		for j := range item.Weights[i] {
			item.Weights[i][j] -= learningRate * item.WeightGradients[i][j]
		}
	}
	// Update biases
	for i := range item.Biases {
		item.Biases[i] -= learningRate * item.BiasGradients[i]
	}
}
