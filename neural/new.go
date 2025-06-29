package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- Утилиты для матричных операций (оставляем без изменений) ---

// DotProduct выполняет скалярное произведение двух векторов.
func DotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
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

// AddVectors складывает два вектора поэлементно.
func AddVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// SubtractVectors вычитает один вектор из другого поэлементно.
func SubtractVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
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

// ElementWiseMultiply поэлементно умножает два вектора.
func ElementWiseMultiply(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// --- Функции активации (оставляем без изменений) ---

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

// --- Объединенная структура слоя нейронной сети ---

// NeuralNetworkLayer представляет один полносвязный слой с функцией активации.
type NeuralNetworkLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64

	// Временные значения для обратного распространения
	Input        []float64 // Входные значения в слой (от предыдущего слоя)
	WeightedSums []float64 // Значения после линейной трансформации (до активации)
	Output       []float64 // Выходные значения после активации

	// Градиенты для обновления весов и смещений
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Градиент, передаваемый на предыдущий слой
}

// NewNeuralNetworkLayer создает новый полносвязный слой с функцией активации.
// activationName может быть "relu", "sigmoid" или "none" для линейного слоя.
func NewNeuralNetworkLayer(inputSize, outputSize int, activationName string) *NeuralNetworkLayer {
	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Инициализация весов
			weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize))
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
	case "none": // Для выходного слоя без активации
		layer.ActivationFunc = func(x float64) float64 { return x }
		layer.DerivativeFunc = func(x float64) float64 { return 1.0 }
	default:
		panic("Неизвестная функция активации: " + activationName)
	}

	return layer
}

// Forward выполняет прямой проход через слой (линейная часть + активация).
func (l *NeuralNetworkLayer) Forward(input []float64) []float64 {
	l.Input = input
	// 1. Линейная трансформация
	l.WeightedSums = MultiplyMatrixVector(l.Weights, input)
	l.WeightedSums = AddVectors(l.WeightedSums, l.Biases)

	// 2. Активация
	l.Output = make([]float64, len(l.WeightedSums))
	for i := range l.WeightedSums {
		l.Output[i] = l.ActivationFunc(l.WeightedSums[i])
	}
	return l.Output
}

// Backward выполняет обратный проход через слой.
func (l *NeuralNetworkLayer) Backward(outputGradient []float64) []float64 {
	// 1. Градиент через функцию активации (применяем производную активации к WeightedSums)
	activationGradient := make([]float64, len(l.WeightedSums))
	for i := range l.WeightedSums {
		activationGradient[i] = l.DerivativeFunc(l.WeightedSums[i])
	}
	// Совмещаем градиент от следующего слоя с градиентом активации
	gradientAfterActivation := ElementWiseMultiply(outputGradient, activationGradient)

	// 2. Градиент по смещениям равен градиенту после активации
	l.BiasGradients = gradientAfterActivation

	// 3. Градиент по весам = внешнее произведение (Input X gradientAfterActivation)
	l.WeightGradients = OuterProduct(gradientAfterActivation, l.Input)

	// 4. Градиент по входу = ТранспонированныеВеса * gradientAfterActivation
	transposedWeights := TransposeMatrix(l.Weights)
	l.InputGradient = MultiplyMatrixVector(transposedWeights, gradientAfterActivation)

	return l.InputGradient
}

// Update обновляет веса и смещения слоя.
func (l *NeuralNetworkLayer) Update(learningRate float64) {
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

// --- Структуры нейронной сети (обновленная) ---

// NeuralNetwork представляет полную нейронную сеть.
type NeuralNetwork struct {
	Layers []*NeuralNetworkLayer // Теперь содержит объединенные слои
}

// NewNeuralNetwork создает новую нейронную сеть с объединенными слоями.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	nn := &NeuralNetwork{}

	currentInputSize := inputSize
	for _, hs := range hiddenSizes {
		// Каждый скрытый слой теперь - это один NeuralNetworkLayer
		nn.Layers = append(nn.Layers, NewNeuralNetworkLayer(currentInputSize, hs, activation))
		currentInputSize = hs
	}

	// Выходной слой без активации (или с "none" активацией)
	nn.Layers = append(nn.Layers, NewNeuralNetworkLayer(currentInputSize, outputSize, "none"))

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

// --- Игровая логика крестиков-ноликов (оставляем без изменений) ---

const (
	Empty   = 0
	PlayerX = 1
	PlayerO = -1
)

// Board представляет игровое поле крестиков-ноликов.
type Board struct {
	Cells         [9]int // 0: пусто, 1: X, -1: O
	CurrentPlayer int    // 1 для X, -1 для O
}

// NewBoard создает новую пустую доску.
func NewBoard() *Board {
	return &Board{
		Cells:         [9]int{Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty},
		CurrentPlayer: PlayerX, // X всегда начинает
	}
}

// MakeMove пытается сделать ход на указанной позиции.
// Возвращает true, если ход был успешным, false в противном случае.
func (b *Board) MakeMove(pos int) bool {
	if pos < 0 || pos >= 9 || b.Cells[pos] != Empty {
		return false
	}
	b.Cells[pos] = b.CurrentPlayer
	return true
}

// CheckWin проверяет, выиграл ли текущий игрок.
func (b *Board) CheckWin() bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Горизонтали
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Вертикали
		{0, 4, 8}, {2, 4, 6}, // Диагонали
	}

	for _, cond := range winConditions {
		if b.Cells[cond[0]] == b.CurrentPlayer &&
			b.Cells[cond[1]] == b.CurrentPlayer &&
			b.Cells[cond[2]] == b.CurrentPlayer {
			return true
		}
	}
	return false
}

// IsBoardFull проверяет, заполнена ли доска.
func (b *Board) IsBoardFull() bool {
	for _, cell := range b.Cells {
		if cell == Empty {
			return false
		}
	}
	return true
}

// IsGameOver проверяет, завершена ли игра (выигрыш или ничья).
func (b *Board) IsGameOver() bool {
	return b.CheckWin() || b.IsBoardFull()
}

// GetReward возвращает вознаграждение для агента, который только что сделал ход.
// player: игрок, который только что сделал ход.
func (b *Board) GetReward(agentPlayer int) float64 {
	if b.CheckWin() {
		if b.CurrentPlayer == agentPlayer {
			return 1.0 // Выигрыш
		} else {
			return -1.0 // Проигрыш
		}
	}
	if b.IsBoardFull() {
		return 0.0 // Ничья
	}
	return -0.01 // Небольшое отрицательное вознаграждение за каждый шаг (для стимулирования быстрой игры)
}

// GetStateVector преобразует состояние доски в вектор для нейронной сети.
// Представляем доску 3x3 как плоский вектор из 9 элементов.
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
	return emptyCells
}

// SwitchPlayer переключает текущего игрока.
func (b *Board) SwitchPlayer() {
	b.CurrentPlayer = -b.CurrentPlayer
}

// PrintBoard выводит доску в консоль.
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

// --- Буфер опыта для DQN (оставляем без изменений) ---

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
	Index       int
	Size        int
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

// Sample выбирает случайный батч опыта из буфера.
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

// --- Агент DQN (обновленный для использования NeuralNetworkLayer) ---

// DQNAgent представляет агента глубокого Q-обучения.
type DQNAgent struct {
	QNetwork      *NeuralNetwork
	TargetNetwork *NeuralNetwork
	ReplayBuffer  *ReplayBuffer
	Gamma         float64 // Коэффициент дисконтирования
	Epsilon       float64 // Для эпсилон-жадной стратегии
	MinEpsilon    float64 // Минимальное значение эпсилон
	EpsilonDecay  float64 // Скорость уменьшения эпсилон
	LearningRate  float64
	UpdateTarget  int // Интервал обновления целевой сети
	PlayerSymbol  int // Символ, которым играет этот агент (PlayerX или PlayerO)
}

// NewDQNAgent создает нового агента DQN.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	// Теперь используем NewNeuralNetworkLayer внутри NewNeuralNetwork
	qNet := NewNeuralNetwork(inputSize, []int{64, 64}, outputSize, "relu") // Пример архитектуры
	targetNet := qNet.Clone()                                              // Клонируем для целевой сети

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         0.99,  // Дисконтный фактор
		Epsilon:       1.0,   // Начинаем с исследования
		MinEpsilon:    0.01,  // Минимальное значение эпсилон
		EpsilonDecay:  0.995, // Уменьшение эпсилон за эпизод
		LearningRate:  0.001,
		UpdateTarget:  1000, // Обновлять целевую сеть каждые 1000 шагов
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

	// Эпсилон-жадная стратегия: случайный ход или лучший ход по Q-сети
	if rand.Float64() < agent.Epsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Случайный ход
	}

	// Выбираем лучший ход по Q-сети
	stateVec := board.GetStateVector(agent.PlayerSymbol)
	qValues := agent.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64

	for _, action := range emptyCells {
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestAction = action
		}
	}
	return bestAction
}

// Train выполняет один шаг обучения агента.
// batchSize: размер батча для обучения.
// step: текущий шаг (для обновления целевой сети).
func (agent *DQNAgent) Train(batchSize, step int) {
	batch := agent.ReplayBuffer.Sample(batchSize)
	if batch == nil {
		return // Недостаточно опыта
	}

	for _, exp := range batch {
		// Предсказанные Q-значения для текущего состояния
		currentQValues := agent.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Копируем, чтобы изменить только одно значение

		// Вычисляем целевое Q-значение
		var nextMaxQ float64
		if !exp.Done {
			// Предсказания целевой сети для следующего состояния
			nextQValues := agent.TargetNetwork.Predict(exp.NextState)
			// Находим максимальное Q-значение для следующего состояния (среди возможных ходов)
			nextMaxQ = -math.MaxFloat64
			for _, qVal := range nextQValues {
				if qVal > nextMaxQ {
					nextMaxQ = qVal
				}
			}
		}

		// Обновляем целевое Q-значение для выбранного действия
		if exp.Done {
			targetQValues[exp.Action] = exp.Reward // Если игра закончена, целевое значение = награда
		} else {
			targetQValues[exp.Action] = exp.Reward + agent.Gamma*nextMaxQ // Уравнение Беллмана
		}

		// Обучаем Q-сеть
		agent.QNetwork.Train(exp.State, targetQValues, agent.LearningRate)
	}

	// Уменьшаем эпсилон
	if agent.Epsilon > agent.MinEpsilon {
		agent.Epsilon *= agent.EpsilonDecay
	}

	// Обновляем целевую сеть
	if step%agent.UpdateTarget == 0 {
		agent.TargetNetwork = agent.QNetwork.Clone()
		fmt.Printf("--- Целевая сеть обновлена на шаге %d ---\n", step)
	}
}

// --- Основной цикл обучения (оставляем без изменений) ---

func main() {
	rand.Seed(time.Now().UnixNano()) // Инициализация генератора случайных чисел

	// Параметры обучения
	episodes := 10000         // Количество игровых эпизодов для обучения
	maxStepsPerEpisode := 100 // Максимальное количество шагов в эпизоде
	batchSize := 32           // Размер батча для обучения DQN
	bufferCapacity := 50000   // Емкость буфера опыта
	trainStartSize := 1000    // Начинать обучение после накопления достаточного опыта

	// Создаем агента DQN (играет за X)
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)
	dqnAgentO := NewDQNAgent(9, 9, bufferCapacity, PlayerO) // Агент для O (может быть случайным или другим DQN)

	totalSteps := 0
	winsX := 0
	winsO := 0
	draws := 0

	fmt.Println("Начало обучения DQN агента для крестиков-ноликов...")

	for episode := 0; episode < episodes; episode++ {
		board := NewBoard()
		isDone := false
		currentStepInEpisode := 0

		for !isDone && currentStepInEpisode < maxStepsPerEpisode {
			var chosenAction int
			var agentPlaying *DQNAgent

			// Выбор агента в зависимости от текущего игрока
			if board.CurrentPlayer == PlayerX {
				agentPlaying = dqnAgentX
			} else {
				// Если играет оппонент (PlayerO), он может быть как DQN, так и случайным
				// Для простоты, здесь PlayerO играет случайно, но вы можете сделать его другим DQNAgent
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) > 0 {
					chosenAction = emptyCells[rand.Intn(len(emptyCells))]
				} else {
					chosenAction = -1 // Нет доступных ходов, игра окончена ничьей
				}
			}

			// Агент X делает ход
			if board.CurrentPlayer == PlayerX {
				// Состояние доски до хода агента
				state := board.GetStateVector(dqnAgentX.PlayerSymbol)
				chosenAction = dqnAgentX.ChooseAction(board)

				if chosenAction == -1 { // Нет доступных ходов, игра окончена
					isDone = true
					break
				}

				// Делаем ход
				board.MakeMove(chosenAction)
				// Состояние доски после хода агента
				nextState := board.GetStateVector(dqnAgentX.PlayerSymbol)
				reward := board.GetReward(dqnAgentX.PlayerSymbol)

				isDone = board.IsGameOver()

				// Добавляем опыт в буфер агента X
				dqnAgentX.ReplayBuffer.Add(Experience{
					State:     state,
					Action:    chosenAction,
					Reward:    reward,
					NextState: nextState,
					Done:      isDone,
				})

				totalSteps++
				// Обучение агента X
				if dqnAgentX.ReplayBuffer.Size >= trainStartSize {
					dqnAgentX.Train(batchSize, totalSteps)
				}

				if isDone {
					break
				}
			} else { // Оппонент (случайный игрок) делает ход
				if chosenAction != -1 {
					board.MakeMove(chosenAction)
				} else {
					isDone = true
					break
				}
				isDone = board.IsGameOver()
				if isDone {
					// Если оппонент завершил игру, его ход тоже приводит к вознаграждению для агента X.
					// Это вознаграждение будет обработано при следующем обучении агента X.
				}
			}

			if !isDone {
				board.SwitchPlayer() // Переключаем игрока только если игра не закончилась
			}
			currentStepInEpisode++
		}

		// Завершение эпизода
		if board.CheckWin() {
			if board.CurrentPlayer == PlayerX {
				winsX++
			} else {
				winsO++
			}
		} else if board.IsBoardFull() {
			draws++
		}

		if (episode+1)%100 == 0 {
			fmt.Printf("Эпизод: %d, Победы X: %d, Победы O: %d, Ничьи: %d, Epsilon: %.4f\n",
				episode+1, winsX, winsO, draws, dqnAgentX.Epsilon)
			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nОбучение завершено.")
	fmt.Println("Тестирование агента...")

	// --- Тестирование обученного агента против случайного оппонента ---
	testGames := 100
	testWinsX := 0
	testDraws := 0
	testLossesX := 0

	// Устанавливаем epsilon на минимальное значение для тестирования
	dqnAgentX.Epsilon = 0.0

	for i := 0; i < testGames; i++ {
		board := NewBoard()
		isDone := false

		for !isDone {
			// Ход агента X
			if board.CurrentPlayer == PlayerX {
				action := dqnAgentX.ChooseAction(board)
				if action == -1 {
					isDone = true
					break // Доска заполнена, ничья
				}
				board.MakeMove(action)
			} else {
				// Ход случайного оппонента (O)
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) == 0 {
					isDone = true
					break // Доска заполнена, ничья
				}
				randomAction := emptyCells[rand.Intn(len(emptyCells))]
				board.MakeMove(randomAction)
			}

			isDone = board.IsGameOver()
			if !isDone {
				board.SwitchPlayer()
			}
		}

		// Подсчет результатов тестовой игры
		if board.CheckWin() {
			if board.CurrentPlayer == PlayerX {
				testWinsX++
			} else {
				testLossesX++
			}
		} else if board.IsBoardFull() {
			testDraws++
		}
	}

	fmt.Printf("\nРезультаты тестирования (%d игр против случайного оппонента):\n", testGames)
	fmt.Printf("Победы агента X: %d\n", testWinsX)
	fmt.Printf("Поражения агента X: %d\n", testLossesX)
	fmt.Printf("Ничьи: %d\n", testDraws)

	// Пример игры после обучения
	fmt.Println("\nПример игры после обучения:")
	board := NewBoard()
	dqnAgentX.Epsilon = 0.0 // Убедиться, что агент играет оптимально

	for !board.IsGameOver() {
		board.PrintBoard()

		var action int
		if board.CurrentPlayer == PlayerX {
			fmt.Println("Ход X (Агент DQN):")
			action = dqnAgentX.ChooseAction(board)
		} else {
			fmt.Println("Ход O (Случайный игрок):")
			emptyCells := board.GetEmptyCells()
			action = emptyCells[rand.Intn(len(emptyCells))]
		}

		if action == -1 {
			fmt.Println("Нет доступных ходов. Ничья.")
			break
		}

		board.MakeMove(action)
		board.SwitchPlayer()
	}
	board.PrintBoard()
	if board.CheckWin() {
		fmt.Printf("Игра окончена! Игрок %s выиграл!\n", map[int]string{PlayerX: "X", PlayerO: "O"}[-board.CurrentPlayer])
	} else {
		fmt.Println("Игра окончена! Ничья!")
	}
}
