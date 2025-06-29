package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// --- Utility functions for matrix operations (from tools.go) ---

// AddVectors adds two vectors element-wise.
func AddVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// SubtractVectors subtracts one vector from another element-wise.
func SubtractVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// MultiplyVectors multiplies two vectors element-wise.
func MultiplyVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// MultiplyScalarVector multiplies a vector by a scalar.
func MultiplyScalarVector(scalar float64, vector []float64) []float64 {
	result := make([]float64, len(vector))
	for i := range vector {
		result[i] = scalar * vector[i]
	}
	return result
}

// DotProduct performs the scalar product of two vectors.
func DotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// OuterProduct calculates the outer product of two vectors.
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

// MultiplyMatrixVector multiplies a matrix by a vector.
func MultiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		result[i] = DotProduct(matrix[i], vector)
	}
	return result
}

// TransposeMatrix transposes the matrix.
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

// --- Activation functions (from tools.go) ---

// Sigmoid activation function.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative derivative of a function Sigmoid.
func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// ReLU activation function.
func ReLU(x float64) float64 {
	return math.Max(0, x)
}

// ReLUDerivative derivative of a function ReLU.
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}

// --- Combined structure of neural network layer (from layer.go) ---

// NeuralNetworkLayer represents one fully connected layer with an activation function.
type NeuralNetworkLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64

	// Temporal values ​​for backpropagation
	InputVector  []float64 // Input values ​​to layer (from previous layer)
	WeightedSums []float64 // Values ​​after linear transformation (before activation)
	OutputVector []float64 // Output values ​​after activation

	// Gradients for updating weights and biases
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Gradient passed to the previous layer
}

// NewNeuralNetworkLayer creates a new fully connected layer with activation function.
// activationName can be "relu", "sigmoid" or "none" for a linear layer.
func NewNeuralNetworkLayer(inputSize, outputSize int, activationName string) *NeuralNetworkLayer {
	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Initialize weights with random values (He initialization for ReLU, general for others)
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
	case "none": // For output layer without activation
		layer.ActivationFunc = func(x float64) float64 { return x }
		layer.DerivativeFunc = func(x float64) float64 { return 1.0 }
	default:
		panic("Unknown activation function: " + activationName)
	}

	return layer
}

// Forward performs a forward pass through the layer (linear part + activation).
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

// Backward performs a reverse pass through the layer.
func (item *NeuralNetworkLayer) Backward(outputGradient []float64) []float64 {
	// 1. Gradient via activation function (apply the derivative of the activation to WeightedSums)
	activationGradient := make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		activationGradient[i] = item.DerivativeFunc(item.WeightedSums[i])
	}
	// Combine the gradient from the next layer with the activation gradient (element-wise multiplication)
	gradientAfterActivation := MultiplyVectors(outputGradient, activationGradient)

	// 2. The gradient at offsets is equal to the gradient after activation
	item.BiasGradients = gradientAfterActivation

	// 3. Gradient by weights = outer product (Input X gradientAfterActivation)
	item.WeightGradients = OuterProduct(gradientAfterActivation, item.InputVector)

	// 4. Gradient after input = TransposedWeights * gradientAfterActivation
	transposedWeights := TransposeMatrix(item.Weights)
	item.InputGradient = MultiplyMatrixVector(transposedWeights, gradientAfterActivation)

	return item.InputGradient
}

// Update - Updates the layer's weights and biases.
func (item *NeuralNetworkLayer) Update(learningRate float64) {
	// Add a gradient clipping threshold. A common value, can be tuned.
	const gradientClipValue = 1.0

	// Update weights
	for i := range item.Weights {
		for j := range item.Weights[i] {
			// Clip weight gradient
			clippedWeightGradient := item.WeightGradients[i][j]
			if clippedWeightGradient > gradientClipValue {
				clippedWeightGradient = gradientClipValue
			} else if clippedWeightGradient < -gradientClipValue {
				clippedWeightGradient = -gradientClipValue
			}
			item.Weights[i][j] -= learningRate * clippedWeightGradient
		}
	}
	// Updating biases
	for i := range item.Biases {
		// Clip bias gradient
		clippedBiasGradient := item.BiasGradients[i]
		if clippedBiasGradient > gradientClipValue {
			clippedBiasGradient = gradientClipValue
		} else if clippedBiasGradient < -gradientClipValue {
			clippedBiasGradient = -gradientClipValue
		}
		item.Biases[i] -= learningRate * clippedBiasGradient
	}
}

// --- Neural network structures (from network.go, with Clone method) ---

// NeuralNetwork represents a complete neural network.
type NeuralNetwork struct {
	Layers []*NeuralNetworkLayer
}

// NewNeuralNetwork creates a new neural network with merged layers.
func NewNeuralNetwork(inputSize int, hiddenSizes []int, outputSize int, activation string) *NeuralNetwork {
	item := &NeuralNetwork{}

	currentInputSize := inputSize
	for _, hs := range hiddenSizes {
		// Each hidden layer is one NeuralNetworkLayer
		item.Layers = append(item.Layers, NewNeuralNetworkLayer(currentInputSize, hs, activation))
		currentInputSize = hs
	}

	// Output layer without activation (or with "none" activation)
	item.Layers = append(item.Layers, NewNeuralNetworkLayer(currentInputSize, outputSize, "none"))

	return item
}

// Predict performs a forward pass to obtain network predictions.
func (item *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range item.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Train performs one step of training the network.
// input: input data
// targetOutput: target output data (Q-values ​​for training)
// learningRate: learning rate
func (item *NeuralNetwork) Train(input []float64, targetOutput []float64, learningRate float64) {
	// Forward pass (saving intermediate values)
	predictedOutput := item.Predict(input)

	// Calculate the gradient of the output (MSE loss derivative)
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

	// Updating weights
	for _, layer := range item.Layers {
		layer.Update(learningRate)
	}
}

// Clone creates a deep copy of the neural network. This is crucial for DQN's Target Network.
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

// --- Tic-Tac-Toe Game Logic ---

const (
	Empty   = 0
	PlayerX = 1
	PlayerO = -1
)

// Board represents the Tic-Tac-Toe game board.
type Board struct {
	Cells         [9]int // 0: empty, 1: X, -1: O
	CurrentPlayer int    // 1 for X, -1 for O
}

// NewBoard creates a new empty board.
func NewBoard() *Board {
	return &Board{
		Cells:         [9]int{Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty},
		CurrentPlayer: PlayerX, // X always starts
	}
}

// MakeMove attempts to make a move at the specified position.
// Returns true if the move was successful, false otherwise.
func (b *Board) MakeMove(pos int) bool {
	if pos < 0 || pos >= 9 || b.Cells[pos] != Empty {
		return false
	}
	b.Cells[pos] = b.CurrentPlayer
	return true
}

// CheckWin checks if the given player has won.
// This function can be called for any player to check win condition.
func (b *Board) CheckWin(player int) bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Horizontals
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Verticals
		{0, 4, 8}, {2, 4, 6}, // Diagonals
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

// IsBoardFull checks if the board is full.
func (b *Board) IsBoardFull() bool {
	for _, cell := range b.Cells {
		if cell == Empty {
			return false
		}
	}
	return true
}

// GetGameOutcome checks if the game is over and returns the winner if any.
// Returns (true, winnerSymbol) if game is over (win or draw), (false, 0) otherwise.
// The winnerSymbol is PlayerX, PlayerO, or Empty (for draw/game in progress).
func (b *Board) GetGameOutcome() (bool, int) {
	if b.CheckWin(PlayerX) {
		return true, PlayerX
	}
	if b.CheckWin(PlayerO) {
		return true, PlayerO
	}
	if b.IsBoardFull() {
		return true, Empty // It's a draw, no specific winner
	}
	return false, Empty // Game is not over yet
}

// GetReward returns the reward for the agent based on the game's outcome.
// This is called AFTER a move has been made and the game state potentially changed.
func (b *Board) GetReward(agentPlayer int) float64 {
	isOver, winner := b.GetGameOutcome()

	if isOver {
		if winner == agentPlayer {
			return 1.0 // Agent won
		} else if winner == Empty {
			return 0.0 // Draw
		} else { // winner == -agentPlayer (opponent)
			return -1.0 // Agent lost (opponent won)
		}
	}
	return 0.0 // No negative reward per step for tic-tac-toe
}

// GetStateVector converts the board state into a vector for the neural network.
// Represents the 3x3 board as a flat 9-element vector.
// 1.0 for agent's cell, -1.0 for opponent's cell, 0.0 for empty.
func (b *Board) GetStateVector(agentPlayer int) []float64 {
	state := make([]float64, 9)
	for i, cell := range b.Cells {
		if cell == agentPlayer {
			state[i] = 1.0
		} else if cell == -agentPlayer { // Opponent
			state[i] = -1.0
		} else { // Empty
			state[i] = 0.0
		}
	}
	return state
}

// GetEmptyCells returns a list of indices of empty cells.
func (b *Board) GetEmptyCells() []int {
	var emptyCells []int
	for i, cell := range b.Cells {
		if cell == Empty {
			emptyCells = append(emptyCells, i)
		}
	}
	return emptyCells // Return ALL empty cells
}

// SwitchPlayer switches the current player.
func (b *Board) SwitchPlayer() {
	b.CurrentPlayer = -b.CurrentPlayer
}

// PrintBoard prints the board to the console.
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

// --- Experience Buffer for DQN ---

// Experience represents a single game experience.
type Experience struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores game experiences.
type ReplayBuffer struct {
	Experiences []Experience
	Capacity    int
	Index       int
	Size        int
}

// NewReplayBuffer creates a new experience buffer.
func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		Experiences: make([]Experience, capacity),
		Capacity:    capacity,
	}
}

// Add adds a new experience to the buffer.
func (rb *ReplayBuffer) Add(exp Experience) {
	rb.Experiences[rb.Index] = exp
	rb.Index = (rb.Index + 1) % rb.Capacity
	if rb.Size < rb.Capacity {
		rb.Size++
	}
}

// Sample selects a random batch of experiences from the buffer.
func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
	if rb.Size < batchSize {
		return nil // Not enough experience to sample a batch
	}

	samples := make([]Experience, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := rand.Intn(rb.Size)
		samples[i] = rb.Experiences[idx]
	}
	return samples
}

// --- DQN Agent ---

// DQNAgent represents a Deep Q-Learning agent.
type DQNAgent struct {
	QNetwork      *NeuralNetwork
	TargetNetwork *NeuralNetwork
	ReplayBuffer  *ReplayBuffer
	Gamma         float64 // Discount factor
	Epsilon       float64 // For epsilon-greedy strategy
	MinEpsilon    float64 // Minimum epsilon value
	EpsilonDecay  float64 // Epsilon decay rate per episode
	LearningRate  float64
	UpdateTarget  int // Target network update interval (steps)
	PlayerSymbol  int // Symbol this agent plays as (PlayerX or PlayerO)
}

// NewDQNAgent creates a new DQN agent.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	qNet := NewNeuralNetwork(inputSize, []int{64, 64}, outputSize, "sigmoid") // Example architecture
	targetNet := qNet.Clone()                                              // Clone for the target network

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         0.99,     // Discount factor
		Epsilon:       1.0,      // Start with exploration
		MinEpsilon:    0.01,     // Minimum epsilon value
		EpsilonDecay:  0.999985, // Epsilon decay per step (slower decay)
		LearningRate:  0.001,
		UpdateTarget:  10000, // Update target network every 10000 steps (less frequent)
		PlayerSymbol:  playerSymbol,
	}
}

// ChooseAction selects an action using an epsilon-greedy strategy.
// board: current board state.
func (agent *DQNAgent) ChooseAction(board *Board) int {
	emptyCells := board.GetEmptyCells()
	if len(emptyCells) == 0 {
		return -1 // No available moves
	}

	// Epsilon-greedy strategy: random move or best move according to Q-network
	if rand.Float64() < agent.Epsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Random move
	}

	// Choose the best move according to the Q-network
	stateVec := board.GetStateVector(agent.PlayerSymbol)
	qValues := agent.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64 // Initialize with a very small number

	for _, action := range emptyCells { // Iterate ONLY over empty cells
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestAction = action
		}
	}
	// If all valid moves have equally bad Q-values (e.g., -Inf),
	// we should still pick a valid random empty cell.
	if bestAction == -1 {
		return emptyCells[rand.Intn(len(emptyCells))]
	}
	return bestAction
}

// Train performs one training step for the agent.
// batchSize: batch size for training.
// step: current step (for target network update).
func (agent *DQNAgent) Train(batchSize, step int) {
	batch := agent.ReplayBuffer.Sample(batchSize)
	if batch == nil {
		return // Not enough experience
	}

	for _, exp := range batch {
		// Predicted Q-values for the current state from the Q-Network
		currentQValues := agent.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Copy to modify only one value

		// Calculate the target Q-value
		var targetQ float64
		if exp.Done {
			targetQ = exp.Reward // If game is over, target value is the immediate reward
		} else {
			// Find max Q-value for the next state from the Target Network
			nextQValues := agent.TargetNetwork.Predict(exp.NextState)
			nextMaxQ := -math.MaxFloat64
			// Iterate over all possible actions (0-8) to find the max Q-value from target network.
			// The agent's Q-network predicts for all 9 cells, so we need to consider all.
			// However, only *valid* moves should be considered for maxQ for the next state.
			// For Tic-Tac-Toe, Q-values for occupied cells should ideally be very low anyway.
			// For simplicity and alignment with current architecture, we find max over all 9 outputs.
			// In a more robust implementation, one might filter for `nextQValues[action]` where `action` is empty in `exp.NextState`.
			for _, qVal := range nextQValues {
				if qVal > nextMaxQ {
					nextMaxQ = qVal
				}
			}
			targetQ = exp.Reward + agent.Gamma*nextMaxQ // Bellman equation
		}

		// Update the target Q-value for the action taken in this experience
		targetQValues[exp.Action] = targetQ

		// Train the Q-network with the updated target Q-values
		agent.QNetwork.Train(exp.State, targetQValues, agent.LearningRate)
	}

	// Decrease epsilon (applied per training step, not per episode)
	if agent.Epsilon > agent.MinEpsilon {
		agent.Epsilon *= agent.EpsilonDecay
	}

	// Update the target network
	if step%agent.UpdateTarget == 0 {
		agent.TargetNetwork = agent.QNetwork.Clone()
		fmt.Printf("--- Target network updated at step %d (Epsilon: %.4f) ---\n", step, agent.Epsilon)
	}
}

// --- Main training loop ---

func main() {
	rand.Seed(time.Now().UnixNano()) // Initialize random number generator

	// Training parameters
	episodes := 100000         // Number of game episodes for training
	maxStepsPerEpisode := 100 // Maximum number of steps per episode
	batchSize := 64           // Batch size for DQN training
	bufferCapacity := 50000   // Experience buffer capacity
	trainStartSize := 1000    // Start training after accumulating enough experience

	// Create two DQN agents for self-play
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)
	dqnAgentO := NewDQNAgent(9, 9, bufferCapacity, PlayerO)

	totalSteps := 0
	winsX := 0
	winsO := 0
	draws := 0

	fmt.Println("Starting DQN self-play agent training for Tic-Tac-Toe...")

	for episode := 0; episode < episodes; episode++ {
		board := NewBoard() // Board.CurrentPlayer is PlayerX by default
		isDone := false
		var gameWinner int // To store the winner of the episode
		currentStepInEpisode := 0

		for !isDone && currentStepInEpisode < maxStepsPerEpisode {
			var currentAgent *DQNAgent
			//var currentPlayerSymbol int

			if board.CurrentPlayer == PlayerX {
				currentAgent = dqnAgentX
				//currentPlayerSymbol = PlayerX
			} else {
				currentAgent = dqnAgentO
				//currentPlayerSymbol = PlayerO
			}

			// Store current state before the move (from the perspective of the current agent)
			stateBeforeMove := board.GetStateVector(currentAgent.PlayerSymbol)

			chosenAction := currentAgent.ChooseAction(board)

			if chosenAction == -1 { // No available moves, implies draw
				isDone, gameWinner = board.GetGameOutcome() // Should be a draw
				break
			}

			// Make the move
			board.MakeMove(chosenAction)

			// Check for game over *immediately* after the move
			isDone, gameWinner = board.GetGameOutcome()

			// Board state after agent's move (from the perspective of the current agent)
			nextState := board.GetStateVector(currentAgent.PlayerSymbol)
			// Reward is for the current agent based on current outcome
			reward := board.GetReward(currentAgent.PlayerSymbol)

			// Add experience to the current agent's buffer
			currentAgent.ReplayBuffer.Add(Experience{
				State:     stateBeforeMove, // State before action
				Action:    chosenAction,
				Reward:    reward,
				NextState: nextState,
				Done:      isDone,
			})

			totalSteps++
			// Train current agent
			if currentAgent.ReplayBuffer.Size >= trainStartSize {
				currentAgent.Train(batchSize, totalSteps)
			}

			if isDone { // Break loop if game is over
				break
			}

			// Only switch player if the game is NOT yet over after the move
			board.SwitchPlayer()
			currentStepInEpisode++
		}

		// End of episode summary - Using the 'gameWinner' variable from GetGameOutcome
		if gameWinner == PlayerX {
			winsX++
		} else if gameWinner == PlayerO {
			winsO++
		} else { // gameWinner == Empty (draw)
			draws++
		}

		if (episode+1)%1000 == 0 {
			fmt.Printf("Episode: %d, Wins X: %d, Wins O: %d, Draws: %d, Epsilon X: %.4f, Epsilon O: %.4f\n",
				episode+1, winsX, winsO, draws, dqnAgentX.Epsilon, dqnAgentO.Epsilon)
			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nTraining finished.")
	fmt.Println("Testing the agents (X vs O self-play)...")

	// --- Test the trained agents against each other ---
	testGames := 1000
	testWinsX := 0
	testDraws := 0
	testLossesX := 0 // Losses for X means Wins for O

	// Set epsilon to minimum for testing for both agents
	dqnAgentX.Epsilon = 0.0
	dqnAgentO.Epsilon = 0.0

	for i := 0; i < testGames; i++ {
		board := NewBoard()
		isDone := false
		var gameWinner int // To store the winner of the test game

		for !isDone {
			var currentAgent *DQNAgent
			if board.CurrentPlayer == PlayerX {
				currentAgent = dqnAgentX
			} else {
				currentAgent = dqnAgentO
			}

			action := currentAgent.ChooseAction(board)
			if action == -1 {
				isDone, gameWinner = board.GetGameOutcome() // Should be a draw
				break
			}
			//currentBoardStateBeforeMove := board.Cells // Save state for debug if needed
			board.MakeMove(action)

			isDone, gameWinner = board.GetGameOutcome()
			if isDone { // Break loop if game is over
				break
			}

			board.SwitchPlayer()
		}

		// Count test game results - Using the 'gameWinner' variable from GetGameOutcome
		if gameWinner == PlayerX {
			testWinsX++
		} else if gameWinner == PlayerO {
			testLossesX++
		} else { // gameWinner == Empty (draw)
			testDraws++
		}
	}

	fmt.Printf("\nTest Results (%d games, Agent X vs Agent O):\n", testGames)
	fmt.Printf("Agent X Wins: %d\n", testWinsX)
	fmt.Printf("Agent O Wins: %d\n", testLossesX)
	fmt.Printf("Draws: %d\n", testDraws)

	// Example game after training
	fmt.Println("\nExample game after training (X vs O):")
	board := NewBoard()
	dqnAgentX.Epsilon = 0.0 // Ensure agent plays optimally
	dqnAgentO.Epsilon = 0.0 // Ensure agent plays optimally

	for { // Loop indefinitely until game is over or max steps
		// Check game outcome at the start of loop iteration (after previous player's potential win)
		isOver, winner := board.GetGameOutcome()
		if isOver {
			board.PrintBoard() // Print final board state
			if winner != Empty {
				fmt.Printf("Game Over! Player %s won!\n", map[int]string{PlayerX: "X", PlayerO: "O"}[winner])
			} else {
				fmt.Println("Game Over! Draw!")
			}
			break // Break the loop if game is over
		}

		// Print board BEFORE the current player's move
		board.PrintBoard()

		var currentAgent *DQNAgent
		var currentPlayerName string
		if board.CurrentPlayer == PlayerX {
			currentAgent = dqnAgentX
			currentPlayerName = "X"
		} else {
			currentAgent = dqnAgentO
			currentPlayerName = "O"
		}

		fmt.Printf("%s's Move (DQN Agent):\n", currentPlayerName)
		action := currentAgent.ChooseAction(board)

		if action == -1 {
			isOver, winner = board.GetGameOutcome() // Should be a draw
			if isOver && winner == Empty {
				fmt.Println("No available moves. Draw.")
			}
			break
		}

		moveSuccessful := board.MakeMove(action)
		if !moveSuccessful {
			fmt.Println("Error: Move failed unexpectedly.")
			break
		}

		board.SwitchPlayer() // Always switch player after a successful move, before the next iteration
	}
}
