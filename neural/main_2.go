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
	// Update weights
	for i := range item.Weights {
		for j := range item.Weights[i] {
			item.Weights[i][j] -= learningRate * item.WeightGradients[i][j]
		}
	}
	// Updating biases
	for i := range item.Biases {
		item.Biases[i] -= learningRate * item.BiasGradients[i]
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
			InputSize:       layer.InputSize,
			OutputSize:      layer.OutputSize,
			Weights:         make([][]float64, len(layer.Weights)),
			Biases:          make([]float64, len(layer.Biases)),
			ActivationFunc:  layer.ActivationFunc,
			DerivativeFunc:  layer.DerivativeFunc,
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

// CheckWin checks if the current player has won.
func (b *Board) CheckWin() bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Horizontals
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Verticals
		{0, 4, 8}, {2, 4, 6},           // Diagonals
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

// IsBoardFull checks if the board is full.
func (b *Board) IsBoardFull() bool {
	for _, cell := range b.Cells {
		if cell == Empty {
			return false
		}
	}
	return true
}

// IsGameOver checks if the game is over (win or draw).
func (b *Board) IsGameOver() bool {
	return b.CheckWin() || b.IsBoardFull()
}

// GetReward returns the reward for the agent who just made a move.
// agentPlayer: the player symbol for the agent.
func (b *Board) GetReward(agentPlayer int) float64 {
	if b.CheckWin() {
		if b.CurrentPlayer == agentPlayer {
			return 1.0 // Win
		} else {
			return -1.0 // Loss
		}
	}
	if b.IsBoardFull() {
		return 0.0 // Draw
	}
	return -0.01 // Small negative reward for each step (to encourage faster games)
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
	return emptyCells
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
	UpdateTarget  int // Target network update interval
	PlayerSymbol  int // Symbol this agent plays as (PlayerX or PlayerO)
}

// NewDQNAgent creates a new DQN agent.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	qNet := NewNeuralNetwork(inputSize, []int{64, 64}, outputSize, "relu") // Example architecture
	targetNet := qNet.Clone() // Clone for the target network

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         0.99,  // Discount factor
		Epsilon:       1.0,   // Start with exploration
		MinEpsilon:    0.01,  // Minimum epsilon value
		EpsilonDecay:  0.995, // Epsilon decay per episode
		LearningRate:  0.001,
		UpdateTarget:  1000, // Update target network every 1000 steps
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
		// Predicted Q-values for the current state
		currentQValues := agent.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Copy to modify only one value

		// Calculate the target Q-value
		var nextMaxQ float64
		if !exp.Done {
			// Target network predictions for the next state
			nextQValues := agent.TargetNetwork.Predict(exp.NextState)
			// Find the maximum Q-value for the next state (among possible moves)
			nextMaxQ = -math.MaxFloat64
			for _, qVal := range nextQValues {
				if qVal > nextMaxQ {
					nextMaxQ = qVal
				}
			}
		}

		// Update the target Q-value for the selected action
		if exp.Done {
			targetQValues[exp.Action] = exp.Reward // If game is over, target value = reward
		} else {
			targetQValues[exp.Action] = exp.Reward + agent.Gamma*nextMaxQ // Bellman equation
		}

		// Train the Q-network
		agent.QNetwork.Train(exp.State, targetQValues, agent.LearningRate)
	}

	// Decrease epsilon
	if agent.Epsilon > agent.MinEpsilon {
		agent.Epsilon *= agent.EpsilonDecay
	}

	// Update the target network
	if step%agent.UpdateTarget == 0 {
		agent.TargetNetwork = agent.QNetwork.Clone()
		fmt.Printf("--- Target network updated at step %d ---\n", step)
	}
}

// --- Main training loop ---

func main() {
	rand.Seed(time.Now().UnixNano()) // Initialize random number generator

	// Training parameters
	episodes := 50000          // Number of game episodes for training
	maxStepsPerEpisode := 100 // Maximum number of steps per episode
	batchSize := 32           // Batch size for DQN training
	bufferCapacity := 50000   // Experience buffer capacity
	trainStartSize := 1000    // Start training after accumulating enough experience

	// Create a DQN agent (plays as X)
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)
	// Opponent (simple random player, for initial training)
	// You could replace this with another DQNAgent for self-play later
	
	totalSteps := 0
	winsX := 0
	winsO := 0
	draws := 0

	fmt.Println("Starting DQN agent training for Tic-Tac-Toe...")

	for episode := 0; episode < episodes; episode++ {
		board := NewBoard() // Board.CurrentPlayer is PlayerX by default
		isDone := false
		currentStepInEpisode := 0

		for !isDone && currentStepInEpisode < maxStepsPerEpisode {
			// Determine whose turn it is
			var currentPlayerSymbol int
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				currentPlayerSymbol = dqnAgentX.PlayerSymbol
			} else {
				currentPlayerSymbol = -dqnAgentX.PlayerSymbol // Opponent's symbol
			}

			var chosenAction int
			// Agent X's turn
			if currentPlayerSymbol == dqnAgentX.PlayerSymbol {
				// Board state before agent's move
				state := board.GetStateVector(dqnAgentX.PlayerSymbol)
				chosenAction = dqnAgentX.ChooseAction(board)

				if chosenAction == -1 { // No available moves, game ends in a draw
					isDone = true
					break
				}
				
				// Make the move
				board.MakeMove(chosenAction)
				// Board state after agent's move
				nextState := board.GetStateVector(dqnAgentX.PlayerSymbol)
				reward := board.GetReward(dqnAgentX.PlayerSymbol) // Reward is for agentX if agentX won/lost
				
				isDone = board.IsGameOver()

				// Add experience to agent X's buffer
				dqnAgentX.ReplayBuffer.Add(Experience{
					State:     state,
					Action:    chosenAction,
					Reward:    reward,
					NextState: nextState,
					Done:      isDone,
				})

				totalSteps++
				// Train agent X
				if dqnAgentX.ReplayBuffer.Size >= trainStartSize {
					dqnAgentX.Train(batchSize, totalSteps)
				}
				
				if isDone {
					break
				}
			} else { // Opponent's turn (random player)
				emptyCells := board.GetEmptyCells()
				var opponentAction int
				if len(emptyCells) > 0 {
					opponentAction = emptyCells[rand.Intn(len(emptyCells))] // Random move
				} else {
					opponentAction = -1 // No available moves, game ends in a draw
				}

				if opponentAction != -1 {
					board.MakeMove(opponentAction)
				} else {
					isDone = true
					break
				}
				isDone = board.IsGameOver()
				// Note: The reward for opponent's move will affect agent X
				// via the Bellman equation in a subsequent training step,
				// when agent X's nextState and reward are considered.
				
				if isDone {
					break
				}
			}
			
			if !isDone {
				board.SwitchPlayer() // Switch player only if game is not over
			}
			currentStepInEpisode++
		}

		// End of episode summary - Corrected logic for winner identification
		if board.CheckWin() {
			// The player who made the last move is the winner.
			// Since CurrentPlayer was switched, the winner is the *previous* player.
			winner := -board.CurrentPlayer 
			if winner == PlayerX {
				winsX++
			} else { // winner == PlayerO
				winsO++
			}
		} else if board.IsBoardFull() {
			draws++
		}

		if (episode+1)%1000 == 0 {
			fmt.Printf("Episode: %d, Wins X: %d, Wins O: %d, Draws: %d, Epsilon: %.4f\n",
				episode+1, winsX, winsO, draws, dqnAgentX.Epsilon)
			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nTraining finished.")
	fmt.Println("Testing the agent...")

	// --- Test the trained agent against a random opponent ---
	testGames := 1000
	testWinsX := 0
	testDraws := 0
	testLossesX := 0

	// Set epsilon to minimum for testing
	dqnAgentX.Epsilon = 0.0

	for i := 0; i < testGames; i++ {
		board := NewBoard()
		isDone := false

		for !isDone {
			// Determine whose turn it is
			var currentPlayerSymbol int
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				currentPlayerSymbol = dqnAgentX.PlayerSymbol
			} else {
				currentPlayerSymbol = -dqnAgentX.PlayerSymbol // Opponent's symbol
			}

			var action int
			// Agent X's move
			if currentPlayerSymbol == dqnAgentX.PlayerSymbol {
				action = dqnAgentX.ChooseAction(board)
				if action == -1 {
					isDone = true
					break // Board full, draw
				}
				board.MakeMove(action)
			} else {
				// Random opponent's (O) move
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) == 0 {
					isDone = true
					break // Board full, draw
				}
				randomAction := emptyCells[rand.Intn(len(emptyCells))]
				board.MakeMove(randomAction)
			}

			isDone = board.IsGameOver()
			if !isDone {
				board.SwitchPlayer()
			}
		}

		// Count test game results - Corrected logic for winner identification
		if board.CheckWin() {
			// The winner is the player who made the last move (-board.CurrentPlayer)
			winner := -board.CurrentPlayer
			if winner == PlayerX {
				testWinsX++
			} else { // winner == PlayerO
				testLossesX++
			}
		} else if board.IsBoardFull() {
			testDraws++
		}
	}

	fmt.Printf("\nTest Results (%d games against random opponent):\n", testGames)
	fmt.Printf("Agent X Wins: %d\n", testWinsX)
	fmt.Printf("Agent X Losses: %d\n", testLossesX)
	fmt.Printf("Draws: %d\n", testDraws)
	
	// Example game after training - Corrected logic for winner identification
	fmt.Println("\nExample game after training:")
	board := NewBoard()
	dqnAgentX.Epsilon = 0.0 // Ensure agent plays optimally
	
	for !board.IsGameOver() {
		board.PrintBoard()
		
		var action int
		// Determine whose turn it is
		var currentPlayerSymbol int
		if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
			currentPlayerSymbol = dqnAgentX.PlayerSymbol
		} else {
			currentPlayerSymbol = -dqnAgentX.PlayerSymbol // Opponent's symbol
		}

		if currentPlayerSymbol == dqnAgentX.PlayerSymbol {
			fmt.Println("X's Move (DQN Agent):")
			action = dqnAgentX.ChooseAction(board)
		} else {
			fmt.Println("O's Move (Random Player):")
			emptyCells := board.GetEmptyCells()
			action = emptyCells[rand.Intn(len(emptyCells))]
		}
		
		if action == -1 {
			fmt.Println("No available moves. Draw.")
			break
		}

		board.MakeMove(action)
		board.SwitchPlayer()
	}
	board.PrintBoard()
	if board.CheckWin() {
		// The winner is the player who made the last move (-board.CurrentPlayer)
		fmt.Printf("Game Over! Player %s won!\n", map[int]string{PlayerX: "X", PlayerO: "O"}[-board.CurrentPlayer])
	} else {
		fmt.Println("Game Over! Draw!")
	}
}
