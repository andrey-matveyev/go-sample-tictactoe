package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- Вспомогательные функции для матричных операций (из tools.go) ---

// AddVectors поэлементно складывает два вектора.
func AddVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// SubtractVectors поэлементно вычитает один вектор из другого.
func SubtractVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// MultiplyVectors поэлементно умножает два вектора.
func MultiplyVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// MultiplyScalarVector умножает вектор на скаляр.
func MultiplyScalarVector(scalar float64, vector []float64) []float64 {
	result := make([]float64, len(vector))
	for i := range vector {
		result[i] = scalar * vector[i]
	}
	return result
}

// DotProduct выполняет скалярное произведение двух векторов.
func DotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// OuterProduct вычисляет внешнее произведение двух векторов.
func OuterProduct(a, b []float64) [][]float64 {
	rows := len(a)
	cols := len(b)
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
		for j := range result[i] {
			result[i][j] = a[i] * b[j]
		}
	}
	return result
}

// MultiplyMatrixVector умножает матрицу на вектор.
func MultiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		result[i] = DotProduct(matrix[i], vector)
	}
	return result
}

// TransposeMatrix транспонирует матрицу.
func TransposeMatrix(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	if rows == 0 {
		return [][]float64{}
	}
	cols := len(matrix[0])
	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			transposed[j][i] = matrix[i][j]
		}
	}
	return transposed
}

// --- Функции активации (из tools.go) ---

// Sigmoid функция активации.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative производная функции Sigmoid.
func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// ReLU функция активации.
func ReLU(x float64) float64 {
	return math.Max(0, x)
}

// ReLUDerivative производная функции ReLU.
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

// Tanh функция активации (гиперболический тангенс).
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// TanhDerivative производная функции Tanh.
func TanhDerivative(x float64) float64 {
	t := math.Tanh(x)
	return 1 - t*t
}

// --- Объединенная структура слоя нейронной сети (из layer.go) ---

// NeuralNetworkLayer представляет один полносвязный слой с функцией активации.
type NeuralNetworkLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64

	// Временные значения для обратного распространения
	InputVector  []float64 // Входные значения в слой (из предыдущего слоя)
	WeightedSums []float64 // Значения после линейного преобразования (перед активацией)
	OutputVector []float64 // Выходные значения после активации

	// Градиенты для обновления весов и смещений
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Градиент, передаваемый предыдущему слою
}

// NewNeuralNetworkLayer создает новый полносвязный слой с функцией активации.
// activationName может быть "relu", "sigmoid" или "none" для линейного слоя.
func NewNeuralNetworkLayer(inputSize, outputSize int, activationName string) *NeuralNetworkLayer {
	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Инициализация весов случайными значениями (He инициализация для ReLU, общая для других)
			if activationName == "relu" {
				weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize))
			} else {
				weights[i][j] = rand.NormFloat64() * math.Sqrt(1.0/float64(inputSize))
			}
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
	case "sigmoid":
		layer.ActivationFunc = Sigmoid
		layer.DerivativeFunc = SigmoidDerivative
	case "relu":
		layer.ActivationFunc = ReLU
		layer.DerivativeFunc = ReLUDerivative
	case "tanh": // Добавлена активация tanh
		layer.ActivationFunc = Tanh
		layer.DerivativeFunc = TanhDerivative
	case "none": // Для выходного слоя без активации
		layer.ActivationFunc = func(x float64) float64 { return x }
		layer.DerivativeFunc = func(x float64) float64 { return 1.0 }
	default:
		panic("Неизвестная функция активации: " + activationName)
	}

	return layer
}

// Forward выполняет прямой проход через слой (линейная часть + активация).
func (item *NeuralNetworkLayer) Forward(input []float64) []float64 {
	item.InputVector = input
	// 1. Линейное преобразование
	item.WeightedSums = MultiplyMatrixVector(item.Weights, input)
	item.WeightedSums = AddVectors(item.WeightedSums, item.Biases)

	// 2. Активация
	item.OutputVector = make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		item.OutputVector[i] = item.ActivationFunc(item.WeightedSums[i])
	}
	return item.OutputVector
}

// Backward выполняет обратный проход через слой.
func (item *NeuralNetworkLayer) Backward(outputGradient []float64) []float64 {
	// 1. Градиент через функцию активации (применение производной активации к WeightedSums)
	activationGradient := make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		activationGradient[i] = item.DerivativeFunc(item.WeightedSums[i])
	}
	// Комбинирование градиента из следующего слоя с градиентом активации (поэлементное умножение)
	gradientAfterActivation := MultiplyVectors(outputGradient, activationGradient)

	// 2. Градиент по смещениям равен градиенту после активации
	item.BiasGradients = gradientAfterActivation

	// 3. Градиент по весам = внешнее произведение (Input X gradientAfterActivation)
	item.WeightGradients = OuterProduct(gradientAfterActivation, item.InputVector)

	// 4. Градиент после входа = TransposedWeights * gradientAfterActivation
	transposedWeights := TransposeMatrix(item.Weights)
	item.InputGradient = MultiplyMatrixVector(transposedWeights, gradientAfterActivation)

	return item.InputGradient
}

// Update - Обновляет веса и смещения слоя.
func (item *NeuralNetworkLayer) Update(learningRate float64) {
	// Добавляем порог отсечения градиента. Общее значение, можно настроить.
	const gradientClipValue = 1.0

	// Обновление весов
	for i := range item.Weights {
		for j := range item.Weights[i] {
			// Отсечение градиента веса
			clippedWeightGradient := item.WeightGradients[i][j]
			if clippedWeightGradient > gradientClipValue {
				clippedWeightGradient = gradientClipValue
			} else if clippedWeightGradient < -gradientClipValue {
				clippedWeightGradient = -gradientClipValue
			}
			item.Weights[i][j] -= learningRate * clippedWeightGradient
		}
	}
	// Обновление смещений
	for i := range item.Biases {
		// Отсечение градиента смещения
		clippedBiasGradient := item.BiasGradients[i]
		if clippedBiasGradient > gradientClipValue {
			clippedBiasGradient = gradientClipValue
		} else if clippedBiasGradient < -gradientClipValue {
			clippedBiasGradient = -gradientClipValue
		}
		item.Biases[i] -= learningRate * clippedBiasGradient
	}
}

// --- Структуры нейронной сети (из network.go, с методом Clone) ---

// NeuralNetwork представляет полную нейронную сеть.
type NeuralNetwork struct {
	Layers []*NeuralNetworkLayer
}

// NewNeuralNetwork создает новую нейронную сеть с объединенными слоями.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	item := &NeuralNetwork{}

	currentInputSize := inputSize
	for _, hs := range hiddenSizes {
		// Каждый скрытый слой является одним NeuralNetworkLayer
		item.Layers = append(item.Layers, NewNeuralNetworkLayer(currentInputSize, hs, activation))
		currentInputSize = hs
	}

	// Выходной слой без активации (или с активацией "none")
	item.Layers = append(item.Layers, NewNeuralNetworkLayer(currentInputSize, outputSize, "tanh"))

	return item
}

// Predict выполняет прямой проход для получения предсказаний сети.
func (item *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range item.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Train выполняет один шаг обучения сети.
// input: входные данные
// targetOutput: целевые выходные данные (Q-значения для обучения)
// learningRate: скорость обучения
func (item *NeuralNetwork) Train(input []float64, targetOutput []float64, learningRate float64) {
	// Прямой проход (сохранение промежуточных значений)
	predictedOutput := item.Predict(input)

	// Вычисление градиента на выходе (производная функции потерь MSE)
	// dLoss/dOutput = 2 * (predicted - target)
	outputGradient := make([]float64, len(predictedOutput))
	for i := range predictedOutput {
		outputGradient[i] = 2 * (predictedOutput[i] - targetOutput[i])
	}

	// Обратный проход
	currentGradient := outputGradient
	for i := len(item.Layers) - 1; i >= 0; i-- {
		currentGradient = item.Layers[i].Backward(currentGradient)
	}

	// Обновление весов
	for _, layer := range item.Layers {
		layer.Update(learningRate)
	}
}

// Clone создает глубокую копию нейронной сети. Это важно для целевой сети DQN.
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

// --- Игровая логика Крестиков-ноликов ---

const (
	Empty   = 0
	PlayerX = 1
	PlayerO = -1
)

// Board представляет игровое поле Крестиков-ноликов.
type Board struct {
	Cells         [9]int // 0: пусто, 1: X, -1: O
	CurrentPlayer int    // 1 для X, -1 для O
}

// NewBoard создает новое пустое поле.
func NewBoard() *Board {
	return &Board{
		Cells:         [9]int{Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty},
		CurrentPlayer: PlayerX, // X всегда начинает
	}
}

// MakeMove пытается сделать ход в указанной позиции.
// Возвращает true, если ход был успешным, false в противном случае.
func (b *Board) MakeMove(pos int) bool {
	if pos < 0 || pos >= 9 || b.Cells[pos] != Empty {
		return false
	}
	b.Cells[pos] = b.CurrentPlayer
	return true
}

// CheckWin проверяет, выиграл ли данный игрок.
// Эту функцию можно вызывать для любого игрока для проверки условия победы.
func (b *Board) CheckWin(player int) bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Горизонтальные
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Вертикальные
		{0, 4, 8}, {2, 4, 6}, // Диагонали
	}

	for _, cond := range winConditions {
		if b.Cells[cond[0]] == player &&
			b.Cells[cond[1]] == player &&
			b.Cells[cond[2]] == player {
			return true
		}
	}
	return false
}

// IsBoardFull проверяет, заполнено ли поле.
func (b *Board) IsBoardFull() bool {
	for _, cell := range b.Cells {
		if cell == Empty {
			return false
		}
	}
	return true
}

// GetGameOutcome проверяет, завершена ли игра, и возвращает победителя, если таковой имеется.
// Возвращает (true, символПобедителя) если игра завершена (победа или ничья), (false, 0) в противном случае.
// СимволПобедителя - это PlayerX, PlayerO или Empty (для ничьей/игры в процессе).
func (b *Board) GetGameOutcome() (bool, int) {
	if b.CheckWin(PlayerX) {
		return true, PlayerX
	}
	if b.CheckWin(PlayerO) {
		return true, PlayerO
	}
	if b.IsBoardFull() {
		return true, Empty // Это ничья, нет конкретного победителя
	}
	return false, Empty // Игра еще не завершена
}

// GetReward возвращает награду для агента на основе исхода игры.
// Эта функция вызывается ПОСЛЕ того, как был сделан ход и состояние игры потенциально изменилось.
func (b *Board) GetReward(agentPlayer int) float64 {
	isOver, winner := b.GetGameOutcome()

	if isOver {
		if winner == agentPlayer {
			return 1.0 // Агент выиграл
		} else if winner == Empty {
			return 0.1 // Ничья
		} else { // winner == -agentPlayer (оппонент)
			return -1.0 // Агент проиграл (оппонент выиграл)
		}
	}
	return 0.0 // Нет отрицательной награды за ход для крестиков-ноликов
}

// GetStateVector преобразует состояние доски в вектор для нейронной сети.
// Представляет поле 3x3 в виде плоского 9-элементного вектора.
// 1.0 для клетки агента, -1.0 для клетки оппонента, 0.0 для пустой.
func (b *Board) GetStateVector(agentPlayer int) []float64 {
	state := make([]float64, 9)
	for i, cell := range b.Cells {
		if cell == agentPlayer {
			state[i] = 1.0
		} else if cell == -agentPlayer { // Оппонент
			state[i] = -1.0
		} else { // Пусто
			state[i] = 0.0
		}
	}
	return state
}

// GetEmptyCells возвращает список индексов пустых клеток.
func (b *Board) GetEmptyCells() []int {
	var emptyCells []int
	for i, cell := range b.Cells {
		if cell == Empty {
			emptyCells = append(emptyCells, i)
		}
	}
	return emptyCells // Возвращает ВСЕ пустые клетки
}

// SwitchPlayer переключает текущего игрока.
func (b *Board) SwitchPlayer() {
	b.CurrentPlayer = -b.CurrentPlayer
}

// PrintBoard выводит поле в консоль.
func (b *Board) PrintBoard() {
	fmt.Println("-------------")
	for i := 0; i < 3; i++ {
		fmt.Print("| ")
		for j := 0; j < 3; j++ {
			val := b.Cells[i*3+j]
			switch val {
			case PlayerX:
				fmt.Print("X")
			case PlayerO:
				fmt.Print("O")
			case Empty:
				fmt.Print(" ")
			}
			fmt.Print(" | ")
		}
		fmt.Println("\n-------------")
	}
}

// --- Буфер опыта для DQN ---

// Experience представляет один игровой опыт.
type Experience struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer хранит игровой опыт.
type ReplayBuffer struct {
	Experiences []Experience
	Capacity    int

	Index int
	Size  int
}

// NewReplayBuffer создает новый буфер опыта.
func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		Experiences: make([]Experience, capacity),
		Capacity:    capacity,
	}
}

// Add добавляет новый опыт в буфер.
func (rb *ReplayBuffer) Add(exp Experience) {
	rb.Experiences[rb.Index] = exp
	rb.Index = (rb.Index + 1) % rb.Capacity
	if rb.Size < rb.Capacity {
		rb.Size++
	}
}

// Sample выбирает случайный батч опытов из буфера.
func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
	if rb.Size < batchSize {
		return nil // Недостаточно опыта для выборки батча
	}

	samples := make([]Experience, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(rb.Size)
		samples[i] = rb.Experiences[idx]
	}
	return samples
}

// --- Агент DQN ---

// DQNAgent представляет агента Deep Q-Learning.
type DQNAgent struct {
	QNetwork      *NeuralNetwork
	TargetNetwork *NeuralNetwork
	ReplayBuffer  *ReplayBuffer
	Gamma         float64 // Коэффициент дисконтирования
	Epsilon       float64 // Для эпсилон-жадной стратегии
	MinEpsilon    float64 // Минимальное значение эпсилон
	EpsilonDecay  float64 // Скорость затухания эпсилон за эпизод
	LearningRate  float64
	UpdateTarget  int // Интервал обновления целевой сети (шаги)
	PlayerSymbol  int // Символ, которым играет этот агент (PlayerX или PlayerO)
}

// NewDQNAgent создает нового агента DQN.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	qNet := NewNeuralNetwork(inputSize, []int{180}, outputSize, "tanh") // Пример архитектуры
	targetNet := qNet.Clone()                                          // Клонирование для целевой сети

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         0.75,     // Коэффициент дисконтирования 0.99
		Epsilon:       1.0,      // Начать с исследования
		MinEpsilon:    0.001,    // Минимальное значение эпсилон 0.001
		EpsilonDecay:  0.999996, // Скорость затухания эпсилон за шаг (очень медленная) 0.999997
		LearningRate:  0.0002, // 0.0002
		UpdateTarget:  30000, // Обновлять целевую сеть каждые 10000 шагов (реже)
		PlayerSymbol:  playerSymbol,
	}
}

// ChooseAction выбирает действие, используя эпсилон-жадную стратегию.
// board: текущее состояние доски.
func (agent *DQNAgent) ChooseAction(board *Board) int {
	emptyCells := board.GetEmptyCells()
	if len(emptyCells) == 0 {
		return -1 // Нет доступных ходов
	}

	// Эпсилон-жадная стратегия: случайный ход или лучший ход согласно Q-сети
	if rand.Float64() < agent.Epsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Случайный ход
	}

	// Выбираем лучший ход согласно Q-сети
	stateVec := board.GetStateVector(agent.PlayerSymbol)
	qValues := agent.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64 // Инициализируем очень маленьким числом

	for _, action := range emptyCells { // Итерируем ТОЛЬКО по пустым клеткам
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestAction = action
		}
	}
	// Если все доступные ходы имеют одинаково плохие Q-значения (например, -Inf),
	// мы все равно должны выбрать случайную доступную пустую клетку.
	if bestAction == -1 {
		return emptyCells[rand.Intn(len(emptyCells))]
	}
	return bestAction
}

// Train выполняет один шаг обучения для агента.
// batchSize: размер батча для обучения.
// step: текущий шаг (для обновления целевой сети).
func (agent *DQNAgent) Train(batchSize, step int) {
	batch := agent.ReplayBuffer.Sample(batchSize)
	if batch == nil {
		return // Недостаточно опыта
	}

	for _, exp := range batch {
		// Предсказанные Q-значения для текущего состояния от Q-сети
		currentQValues := agent.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Копируем для изменения только одного значения

		// Вычисляем целевое Q-значение
		var targetQ float64
		if exp.Done {
			targetQ = exp.Reward // Если игра завершена, целевое значение - это немедленная награда
		} else {
			// --- Модификация Double DQN ---
			// 1. Получаем Q-значения для следующего состояния от Q-сети (для выбора лучшего действия)
			qValuesNextStateFromQNetwork := agent.QNetwork.Predict(exp.NextState)

			// Находим действие, которое было бы выбрано Q-сетью в следующем состоянии.
			bestActionFromQNetwork := -1
			maxQValFromQNetwork := -math.MaxFloat64

			// Находим индекс лучшего действия из предсказаний Q-сети.
			// Здесь предполагается, что сеть научится присваивать низкие Q-значения недопустимым ходам.
			for i, qVal := range qValuesNextStateFromQNetwork {
				if qVal > maxQValFromQNetwork {
					maxQValFromQNetwork = qVal
					bestActionFromQNetwork = i
				}
			}

			// Fallback для `bestActionFromQNetwork` на случай, если все предсказанные Q-значения
			// равны `maxQValFromQNetwork` (например, все -math.MaxFloat64 на очень ранних этапах).
			// В хорошо обученной сети `bestActionFromQNetwork` всегда должно быть допустимым действием.
			// Если он остается -1, это указывает на проблему с предсказаниями сети.
			if bestActionFromQNetwork == -1 {
				bestActionFromQNetwork = rand.Intn(9) // Запасной вариант: случайное действие (маловероятно)
			}

			// 2. Оцениваем Q-значение выбранного действия, используя Целевую Сеть
			qValueFromTargetNetwork := agent.TargetNetwork.Predict(exp.NextState)[bestActionFromQNetwork]

			targetQ = exp.Reward + agent.Gamma*qValueFromTargetNetwork // Уравнение Беллмана (DDQN)
			// --- Конец модификации Double DQN ---
		}

		// Обновляем целевое Q-значение для действия, выполненного в этом опыте
		targetQValues[exp.Action] = targetQ

		// Обучаем Q-сеть с обновленными целевыми Q-значениями
		agent.QNetwork.Train(exp.State, targetQValues, agent.LearningRate)
	}

	// Уменьшаем эпсилон (применяется за шаг обучения, а не за эпизод)
	if agent.Epsilon > agent.MinEpsilon {
		agent.Epsilon *= agent.EpsilonDecay
	}

	// Обновляем целевую сеть
	if step%agent.UpdateTarget == 0 {
		agent.TargetNetwork = agent.QNetwork.Clone()
		fmt.Printf("--- Целевая сеть обновлена на шаге %d (Epsilon: %.4f) ---\n", step, agent.Epsilon)
	}
}

// --- Основной цикл обучения ---

func main() {
	rand.Seed(time.Now().UnixNano()) // Инициализация генератора случайных чисел

	// Параметры обучения
	episodes := 400000       // Количество игровых эпизодов для обучения
	maxStepsPerEpisode := 10 // Максимальное количество шагов за эпизод
	batchSize := 10          // Размер батча для обучения DQN 32
	bufferCapacity := 5000   // Емкость буфера опыта 50000
	trainStartSize := 1000   // Начать обучение после накопления достаточного опыта

	// Создаем одного агента DQN (играет за X)
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)

	totalSteps := 0
	winsX := 0
	winsO := 0 // Победы O - это поражения X
	draws := 0

	fmt.Println("Начинается обучение агента DQN (X) против случайного оппонента (O) для Крестиков-ноликов...")

	for episode := 0; episode < episodes; episode++ {
		board := NewBoard() // Board.CurrentPlayer по умолчанию PlayerX
		isDone := false
		var gameWinner int // Для хранения победителя эпизода
		currentStepInEpisode := 0

		for !isDone && currentStepInEpisode < maxStepsPerEpisode {
			// Сохраняем текущее состояние до хода (с точки зрения агента X)
			stateBeforeMove := board.GetStateVector(dqnAgentX.PlayerSymbol)

			var chosenAction int
			// Определяем, чей ход
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				// Ход агента X
				chosenAction = dqnAgentX.ChooseAction(board)

				if chosenAction == -1 { // Нет доступных ходов, подразумевает ничью
					isDone, gameWinner = board.GetGameOutcome() // Должна быть ничья
					break
				}

				// Делаем ход
				board.MakeMove(chosenAction)

				// Проверяем, завершена ли игра СРАЗУ после хода
				isDone, gameWinner = board.GetGameOutcome()

				// Состояние доски после хода агента (с точки зрения агента X)
				nextState := board.GetStateVector(dqnAgentX.PlayerSymbol)
				// Награда для агента X на основе текущего исхода
				reward := board.GetReward(dqnAgentX.PlayerSymbol)

				// Добавляем опыт в буфер агента X
				dqnAgentX.ReplayBuffer.Add(Experience{
					State:     stateBeforeMove, // Состояние до действия
					Action:    chosenAction,
					Reward:    reward,
					NextState: nextState,
					Done:      isDone,
				})

				totalSteps++
				// Обучаем агента X
				if dqnAgentX.ReplayBuffer.Size >= trainStartSize {
					dqnAgentX.Train(batchSize, totalSteps)
				}

				if isDone { // Прерываем цикл, если игра завершена
					break
				}
			} else { // Ход оппонента (случайный игрок)
				emptyCells := board.GetEmptyCells()
				var opponentAction int
				if len(emptyCells) > 0 {
					opponentAction = emptyCells[rand.Intn(len(emptyCells))] // Случайный ход
				} else {
					opponentAction = -1 // Нет доступных ходов, подразумевает ничью
				}

				if opponentAction != -1 {
					board.MakeMove(opponentAction)
				} else {
					isDone, gameWinner = board.GetGameOutcome() // Должна быть ничья
					break
				}

				// Проверяем, завершена ли игра СРАЗУ после хода
				isDone, gameWinner = board.GetGameOutcome()

				if isDone { // Прерываем цикл, если игра завершена
					break
				}
			}

			// Переключаем игрока ТОЛЬКО если игра еще НЕ завершена после хода
			board.SwitchPlayer()
			currentStepInEpisode++
		}

		// Сводка по завершении эпизода - используем переменную 'gameWinner' из GetGameOutcome
		if gameWinner == PlayerX {
			winsX++
		} else if gameWinner == PlayerO {
			winsO++
		} else { // gameWinner == Empty (ничья)
			draws++
		}

		if (episode+1)%1000 == 0 {
			// Создаем временную пустую доску для оценки Q-значения первого хода
			emptyBoardForQEval := NewBoard()
			emptyBoardStateVec := emptyBoardForQEval.GetStateVector(dqnAgentX.PlayerSymbol)
			qValuesForEmptyBoard := dqnAgentX.QNetwork.Predict(emptyBoardStateVec)

			// Индекс 4 соответствует центральной клетке (0-8)
			//qValueCenterCell := qValuesForEmptyBoard[4]

			fmt.Printf("Эпизод: %d, Побед X: %d, Поражений X: %d, Ничьих: %d, Эпсилон X: %.4f, Q(start): %.4f|%.4f|%.4f  %.4f[%.4f]%.4f  %.4f|%.4f|%.4f\n",
				episode+1, winsX, winsO, draws, dqnAgentX.Epsilon,
				qValuesForEmptyBoard[0], qValuesForEmptyBoard[1], qValuesForEmptyBoard[2],
				qValuesForEmptyBoard[3], qValuesForEmptyBoard[4], qValuesForEmptyBoard[5],
				qValuesForEmptyBoard[6], qValuesForEmptyBoard[7], qValuesForEmptyBoard[8])

			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nОбучение завершено.")
	fmt.Println("Тестирование агента (X против случайного O)...")

	// --- Тестируем обученного агента против случайного оппонента ---
	testGames := 1000
	testWinsX := 0
	testDraws := 0
	testLossesX := 0

	// Устанавливаем эпсилон на минимум для тестирования
	dqnAgentX.Epsilon = 0.0

	for i := 0; i < testGames; i++ {
		board := NewBoard()
		isDone := false
		var gameWinner int // Для хранения победителя тестовой игры

		for !isDone {
			var action int
			// Ход текущего игрока
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				action = dqnAgentX.ChooseAction(board)
				if action == -1 {
					isDone, gameWinner = board.GetGameOutcome() // Должна быть ничья
					break
				}
				board.MakeMove(action)
			} else {
				// Ход случайного оппонента (O)
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) == 0 {
					isDone, gameWinner = board.GetGameOutcome() // Должна быть ничья
					break
				}
				randomAction := emptyCells[rand.Intn(len(emptyCells))]
				board.MakeMove(randomAction)
			}

			// Проверяем, завершена ли игра СРАЗУ после хода
			isDone, gameWinner = board.GetGameOutcome()
			if isDone { // Прерываем цикл, если игра завершена
				break
			}

			// Переключаем игрока ТОЛЬКО если игра еще НЕ завершена после хода
			board.SwitchPlayer()
		}

		// Подсчет результатов тестовой игры - используем переменную 'gameWinner' из GetGameOutcome
		if gameWinner == PlayerX {
			testWinsX++
		} else if gameWinner == PlayerO {
			testLossesX++
		} else { // gameWinner == Empty (ничья)
			testDraws++
		}
	}

	fmt.Printf("\nРезультаты теста (%d игр, Агент X против случайного O):\n", testGames)
	fmt.Printf("Побед Агента X: %d\n", testWinsX)
	fmt.Printf("Поражений Агента X (Побед случайного O): %d\n", testLossesX)
	fmt.Printf("Ничьих: %d\n", testDraws)

	// Пример игры после обучения
	fmt.Println("\nПример игры после обучения (X против случайного O):")
	board := NewBoard()
	dqnAgentX.Epsilon = 0.0 // Гарантируем, что агент играет оптимально

	for { // Бесконечный цикл, пока игра не завершится или не превысит макс. шагов
		// Проверяем исход игры в начале итерации цикла (после потенциальной победы предыдущего игрока)
		isOver, winner := board.GetGameOutcome()
		if isOver {
			board.PrintBoard() // Выводим финальное состояние доски
			if winner != Empty {
				fmt.Printf("Игра окончена! Игрок %s победил!\n", map[int]string{PlayerX: "X", PlayerO: "O"}[winner])
			} else {
				fmt.Println("Игра окончена! Ничья!")
			}
			break // Прерываем цикл, если игра завершена
		}

		// Выводим доску ПЕРЕД ходом текущего игрока
		board.PrintBoard()

		var currentPlayerName string
		if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
			currentPlayerName = "X"
		} else {
			currentPlayerName = "O"
		}

		fmt.Printf("%s's Ход:\n", currentPlayerName)
		var action int
		var moveSuccessful bool

		if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
			action = dqnAgentX.ChooseAction(board)
			if action == -1 {
				isOver, winner = board.GetGameOutcome()
				break
			}
			moveSuccessful = board.MakeMove(action)
		} else { // Ход случайного оппонента
			emptyCells := board.GetEmptyCells()
			if len(emptyCells) == 0 {
				isOver, winner = board.GetGameOutcome()
				break
			}
			action = emptyCells[rand.Intn(len(emptyCells))]
			moveSuccessful = board.MakeMove(action)
		}

		if !moveSuccessful {
			fmt.Println("Ошибка: Ход не удался неожиданно.")
			break
		}

		board.SwitchPlayer() // Всегда переключаем игрока после успешного хода, перед следующей итерацией
	}
}
