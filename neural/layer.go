// https://github.com/gorgonia/gorgonia/blob/master/README.md

package neural

import (
	"math"
	"math/rand"
)

// Layer интерфейс для слоев нейронной сети.
type Layer interface {
	Forward(input []float64) []float64
	Backward(outputGradient []float64) []float64
	Update(learningRate float64)
}

// DenseLayer представляет полносвязный слой.
type DenseLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	// Временные значения для обратного распространения
	Input        []float64
	Output       []float64
	WeightedSums []float64 // z-values before activation
	// Градиенты для обновления весов и смещений
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Градиент, передаваемый на предыдущий слой
}

// NewDenseLayer создает новый полносвязный слой.
func NewDenseLayer(inputSize, outputSize int) *DenseLayer {
	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Инициализация весов случайными значениями
			weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize)) // He initialization
		}
		biases[i] = 0.0
	}

	return &DenseLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    weights,
		Biases:     biases,
	}
}

// Forward выполняет прямой проход через полносвязный слой.
func (l *DenseLayer) Forward(input []float64) []float64 {
	l.Input = input
	l.WeightedSums = MultiplyMatrixVector(l.Weights, input)
	l.Output = AddVectors(l.WeightedSums, l.Biases) // Output of dense layer (before activation)
	return l.Output
}

// Backward выполняет обратный проход через полносвязный слой.
func (l *DenseLayer) Backward(outputGradient []float64) []float64 {
	// Градиент по смещениям равен градиенту по выходам
	l.BiasGradients = outputGradient

	// Градиент по весам = внешнее произведение (Input X OutputGradient)
	l.WeightGradients = OuterProduct(outputGradient, l.Input)

	// Градиент по входу = ТранспонированныеВеса * OutputGradient
	transposedWeights := TransposeMatrix(l.Weights)
	l.InputGradient = MultiplyMatrixVector(transposedWeights, outputGradient)

	return l.InputGradient
}

// Update обновляет веса и смещения слоя с использованием градиентов.
func (l *DenseLayer) Update(learningRate float64) {
	// Обновление весов
	for i := range l.Weights {
		for j := range l.Weights[i] {
			l.Weights[i][j] -= learningRate * l.WeightGradients[i][j]
		}
	}
	// Обновление смещений
	for i := range l.Biases {
		l.Biases[i] -= learningRate * l.BiasGradients[i]
	}
}

// ActivationLayer представляет слой активации (например, ReLU или Sigmoid).
type ActivationLayer struct {
	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64
	Input          []float64 // Входные значения перед активацией
	Output         []float64 // Выходные значения после активации
	InputGradient  []float64 // Градиент, передаваемый на предыдущий слой
}

// NewActivationLayer создает новый слой активации.
func NewActivationLayer(activationName string) *ActivationLayer {
	switch activationName {
	case "sigmoid":
		return &ActivationLayer{ActivationFunc: Sigmoid, DerivativeFunc: SigmoidDerivative}
	case "relu":
		return &ActivationLayer{ActivationFunc: ReLU, DerivativeFunc: ReLUDerivative}
	default:
		panic("Неизвестная функция активации: " + activationName)
	}
}

// Forward выполняет прямой проход через слой активации.
func (l *ActivationLayer) Forward(input []float64) []float64 {
	l.Input = input
	l.Output = make([]float64, len(input))
	for i := range input {
		l.Output[i] = l.ActivationFunc(input[i])
	}
	return l.Output
}

// Backward выполняет обратный проход через слой активации.
func (l *ActivationLayer) Backward(outputGradient []float64) []float64 {
	l.InputGradient = make([]float64, len(outputGradient))
	for i := range outputGradient {
		l.InputGradient[i] = outputGradient[i] * l.DerivativeFunc(l.Input[i])
	}
	return l.InputGradient
}

// Update не делает ничего для слоя активации, так как у него нет обучаемых параметров.
func (l *ActivationLayer) Update(learningRate float64) {}

// NeuralNetwork представляет полную нейронную сеть.
type NeuralNetwork struct {
	Layers []Layer
}

// NewNeuralNetwork создает новую нейронную сеть.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	nn := &NeuralNetwork{}

	currentInputSize := inputSize
	for _, hs := range hiddenSizes {
		nn.Layers = append(nn.Layers, NewDenseLayer(currentInputSize, hs))
		nn.Layers = append(nn.Layers, NewActivationLayer(activation))
		currentInputSize = hs
	}

	// Выходной слой без активации для DQN (для предсказания Q-значений)
	nn.Layers = append(nn.Layers, NewDenseLayer(currentInputSize, outputSize))

	return nn
}

// Predict выполняет прямой проход для получения предсказаний сети.
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Train выполняет один шаг обучения сети.
// input: входные данные
// targetOutput: целевые выходные данные (Q-значения для обучения)
// learningRate: скорость обучения
func (nn *NeuralNetwork) Train(input []float64, targetOutput []float64, learningRate float64) {
	// Прямой проход (сохранение промежуточных значений)
	predictedOutput := nn.Predict(input)

	// Вычисление градиента по выходу (MSE loss derivative)
	// dLoss/dOutput = 2 * (predicted - target)
	outputGradient := make([]float64, len(predictedOutput))
	for i := range predictedOutput {
		outputGradient[i] = 2 * (predictedOutput[i] - targetOutput[i])
	}

	// Обратный проход
	currentGradient := outputGradient
	for i := len(nn.Layers) - 1; i >= 0; i-- {
		currentGradient = nn.Layers[i].Backward(currentGradient)
	}

	// Обновление весов
	for _, layer := range nn.Layers {
		layer.Update(learningRate)
	}
}

// Clone создает глубокую копию нейронной сети.
func (nn *NeuralNetwork) Clone() *NeuralNetwork {
	clone := &NeuralNetwork{
		Layers: make([]Layer, len(nn.Layers)),
	}
	for i, layer := range nn.Layers {
		// В зависимости от типа слоя нужно скопировать его специфичные поля
		switch l := layer.(type) {
		case *DenseLayer:
			newDense := &DenseLayer{
				InputSize:  l.InputSize,
				OutputSize: l.OutputSize,
				Weights:    make([][]float64, len(l.Weights)),
				Biases:     make([]float64, len(l.Biases)),
			}
			for r := range l.Weights {
				newDense.Weights[r] = make([]float64, len(l.Weights[r]))
				copy(newDense.Weights[r], l.Weights[r])
			}
			copy(newDense.Biases, l.Biases)
			clone.Layers[i] = newDense
		case *ActivationLayer:
			clone.Layers[i] = &ActivationLayer{
				ActivationFunc: l.ActivationFunc,
				DerivativeFunc: l.DerivativeFunc,
			}
		}
	}
	return clone
}
