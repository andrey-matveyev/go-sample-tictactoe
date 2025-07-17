package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
)

// --- Neural network structures (with Clone method) ---

// NeuralNetwork represents a complete neural network.
type NeuralNetwork struct {
	InputSize   int    `json:"input_size"`
	HiddenSizes []int  `json:"hidden_sizes"`
	OutputSize  int    `json:"output_size"`
	Activation  string `json:"activation"`

	Layers []*NeuralNetworkLayer `json:"-"`
}

// NewNeuralNetwork creates a new neural network with combined layers.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	item := &NeuralNetwork{
		InputSize:   inputSize,
		HiddenSizes: hiddenSizes,
		OutputSize:  outputSize,
		Activation:  activation,
		Layers:      make([]*NeuralNetworkLayer, 0),
	}

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
func (item *NeuralNetwork) Clone() *NeuralNetwork {
	clone := &NeuralNetwork{
		InputSize:   item.InputSize,
		HiddenSizes: item.HiddenSizes,
		OutputSize:  item.OutputSize,
		Activation:  item.Activation,
		Layers:      make([]*NeuralNetworkLayer, len(item.Layers)),
	}
	for i, layer := range item.Layers {
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

	ActivationName string                `json:"activation_name"`
	ActivationFunc func(float64) float64 `json:"-"`
	DerivativeFunc func(float64) float64 `json:"-"`

	// Temporary values for backpropagation
	InputVector  []float64 `json:"-"` // Input values to the layer (from the previous layer)
	WeightedSums []float64 `json:"-"` // Values after linear transformation (before activation)
	OutputVector []float64 `json:"-"` // Output values after activation

	// Gradients for updating weights and biases
	WeightGradients [][]float64 `json:"-"`
	BiasGradients   []float64   `json:"-"`
	InputGradient   []float64   `json:"-"` // Gradient passed to the previous layer
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
		InputSize:      inputSize,
		OutputSize:     outputSize,
		Weights:        weights,
		Biases:         biases,
		ActivationName: activationName,
	}

	setActivationFuncs(layer, activationName)

	return layer
}

// setActivationFuncs sets the activation functions and their derivatives for the layer
func setActivationFuncs(layer *NeuralNetworkLayer, activationName string) {
	switch activationName {
	case "tanh":
		layer.ActivationFunc = Tanh
		layer.DerivativeFunc = TanhDerivative
	case "none":
		layer.ActivationFunc = func(x float64) float64 { return x }
		layer.DerivativeFunc = func(x float64) float64 { return 1.0 }
	default:
		panic("Unknown activation function: " + activationName)
	}
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

// NeuralNetworkSaveData represents the serializable state of the NeuralNetwork.
// Used to save and load weights.
type NeuralNetworkSaveData struct {
	InputSize   int    `json:"input_size"`
	HiddenSizes []int  `json:"hidden_sizes"`
	OutputSize  int    `json:"output_size"`
	Activation  string `json:"activation"` // General activation for hidden layers

	LayersData []LayerSaveData `json:"layers"`
}

// LayerSaveData represents the serializable state of a single layer.
type LayerSaveData struct {
	Weights        [][]float64 `json:"weights"`
	Biases         []float64   `json:"biases"`
	ActivationName string      `json:"activation_name"` // Activation name for each layer
}

// SaveWeights сохраняет веса и архитектуру нейронной сети в JSON-файл.
func SaveNeuralNetwork(filePath string, neural *NeuralNetwork) error {
	saveData := NeuralNetworkSaveData{
		InputSize:   neural.InputSize,
		HiddenSizes: neural.HiddenSizes,
		OutputSize:  neural.OutputSize,
		Activation:  neural.Activation,
		LayersData:  make([]LayerSaveData, len(neural.Layers)),
	}

	for i, layer := range neural.Layers {
		saveData.LayersData[i] = LayerSaveData{
			Weights:        layer.Weights,
			Biases:         layer.Biases,
			ActivationName: layer.ActivationName,
		}
	}

	// Маршалинг данных в JSON с отступами для читаемости
	jsonData, err := json.MarshalIndent(saveData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal neural network data: %w", err)
	}

	// Запись JSON в файл
	err = ioutil.WriteFile(filePath, jsonData, 0644) // 0644 - права доступа (чтение/запись для владельца, только чтение для остальных)
	if err != nil {
		return fmt.Errorf("failed to write neural network data to file %s: %w", filePath, err)
	}

	return nil
}

// LoadWeights загружает веса и архитектуру нейронной сети из JSON-файла.
func LoadNeuralNetwork(filePath string) (*NeuralNetwork, error) {
	// Чтение данных из файла
	jsonData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read neural network data from file %s: %w", filePath, err)
	}

	var saveData NeuralNetworkSaveData
	err = json.Unmarshal(jsonData, &saveData)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal neural network data: %w", err)
	}

	// Воссоздание архитектуры нейронной сети
	neural := NewNeuralNetwork(saveData.InputSize, saveData.HiddenSizes, saveData.OutputSize, saveData.Activation)

	// Загрузка весов и смещений в слои
	if len(neural.Layers) != len(saveData.LayersData) {
		return nil, fmt.Errorf("mismatch in number of layers: expected %d, got %d from file", len(neural.Layers), len(saveData.LayersData))
	}

	for i, layerData := range saveData.LayersData {
		layer := neural.Layers[i]

		if len(layer.Weights) != len(layerData.Weights) || len(layer.Weights[0]) != len(layerData.Weights[0]) {
			return nil, fmt.Errorf("mismatch in weights dimensions for layer %d", i)
		}
		if len(layer.Biases) != len(layerData.Biases) {
			return nil, fmt.Errorf("mismatch in biases dimensions for layer %d", i)
		}

		// Копирование загруженных весов и смещений
		for r := range layerData.Weights {
			copy(layer.Weights[r], layerData.Weights[r])
		}
		copy(layer.Biases, layerData.Biases)

		// Убедимся, что функции активации корректно установлены после загрузки
		// NewNeuralNetworkLayer уже вызывает setActivationFuncs, но для надежности
		// можно переустановить, если ActivationName отличается (что не должно быть)
		setActivationFuncs(layer, layerData.ActivationName)
	}

	return neural, nil
}
