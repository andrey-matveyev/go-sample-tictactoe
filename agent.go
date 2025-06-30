package main

import (
	"fmt"
	"math"
	"math/rand"
)

// --- DQN Agent ---

// DQNAgent represents a Deep Q-Learning agent.
type DQNAgent struct {
	QNetwork      *NeuralNetwork
	TargetNetwork *NeuralNetwork
	ReplayBuffer  *ReplayBuffer
	Gamma         float64 // Discount factor
	MaxEpsilon    float64 // For epsilon-greedy strategy
	MinEpsilon    float64 // Minimum epsilon value
	EpsilonDecay  float64 // Epsilon decay rate per episode
	LearningRate  float64
	UpdateTarget  int // Target network update interval (steps)
	PlayerSymbol  int // Symbol this agent plays (PlayerX or PlayerO)
}

// NewDQNAgent creates a new DQN agent.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	qNet := NewNeuralNetwork(inputSize, []int{hiddenLayerSize}, outputSize, "tanh") // Example architecture
	targetNet := qNet.Clone()                                                       // Clone for the target network

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         gamma,
		MaxEpsilon:    maxEpsilon,
		MinEpsilon:    minEpsilon,
		EpsilonDecay:  epsilonDecay,
		LearningRate:  learningRate,
		UpdateTarget:  updateTarget,
		PlayerSymbol:  playerSymbol,
	}
}

// ChooseAction selects an action using an epsilon-greedy strategy.
// board: current board state.
func (item *DQNAgent) ChooseAction(board *Board) int {
	emptyCells := board.GetEmptyCells()

	// Uncomment this code if your agent moves first and you want it to make the first move to the center.
	/*
		if len(emptyCells) == 9 {
			return 4 // FIRST move to center
		}
	*/
	// OR
	// Uncomment this code if your agent moves first and you want it to make the first move randomly.
	/*
		if len(emptyCells) == 9 {
			return emptyCells[rand.Intn(len(emptyCells))] // Random FIRST move
		}
	*/

	// Epsilon-greedy strategy: random move or best move according to Q-network
	if rand.Float64() < item.MaxEpsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Random move (Research process)
	}

	// Choose the best move according to the Q-network
	stateVec := board.GetStateVector(item.PlayerSymbol)
	qValues := item.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64 // Initialize with a very small number
	for _, action := range emptyCells { // Iterate ONLY through empty cells
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestAction = action // Found a new maximum
		}
	}
	return bestAction
}

// Train performs one training step for the agent.
// batchSize: batch size for training.
// step: current step (for target network update).
func (item *DQNAgent) Train(batchSize, step int) {
	batch := item.ReplayBuffer.Sample(batchSize)
	if batch == nil {
		return // Not enough experience
	}

	for _, exp := range batch {
		// Predicted Q-values for the current state from the Q-network
		currentQValues := item.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Copy to modify only one value

		// Calculate the target Q-value
		var targetQ float64
		if exp.Done {
			targetQ = exp.Reward // If the game is over, the target value is the immediate reward
		} else {
			// 1. Get Q-values for the next state from the Q-network (to choose the best action)
			qValuesNextStateFromQNetwork := item.QNetwork.Predict(exp.NextState)
			// Find the action that would be chosen by the Q-network in the next state.
			bestActionFromQNetwork := -1
			maxQValFromQNetwork := -math.MaxFloat64
			// Find the index of the best action from the Q-network's predictions.
			for i, qVal := range qValuesNextStateFromQNetwork {
				if qVal > maxQValFromQNetwork {
					maxQValFromQNetwork = qVal
					bestActionFromQNetwork = i
				}
			}
			// 2. Evaluate the Q-value of the chosen action using the Target Network
			qValueFromTargetNetwork := item.TargetNetwork.Predict(exp.NextState)[bestActionFromQNetwork]
			targetQ = exp.Reward + item.Gamma*qValueFromTargetNetwork // Bellman Equation (DDQN) !!!
		}
		// Update the target Q-value for the action taken in this experience
		targetQValues[exp.Action] = targetQ
		// Train the Q-network with the updated target Q-values
		item.QNetwork.Train(exp.State, targetQValues, item.LearningRate)
	}
	// Decay epsilon (applied per training step, not per episode)
	if item.MaxEpsilon > item.MinEpsilon {
		item.MaxEpsilon *= item.EpsilonDecay
	}
	// Update the target network
	if step%item.UpdateTarget == 0 {
		item.TargetNetwork = item.QNetwork.Clone()
		fmt.Printf("--- Target network updated at step %d (Epsilon: %.4f) ---\n", step, item.MaxEpsilon)
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

	Index int
	Size  int
}

// NewReplayBuffer creates a new experience buffer.
func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		Experiences: make([]Experience, capacity),
		Capacity:    capacity,
	}
}

// Add adds a new experience to the buffer.
func (item *ReplayBuffer) Add(exp Experience) {
	item.Experiences[item.Index] = exp
	item.Index = (item.Index + 1) % item.Capacity
	if item.Size < item.Capacity {
		item.Size++
	}
}

// Sample selects a random batch of experiences from the buffer.
func (item *ReplayBuffer) Sample(batchSize int) []Experience {
	if item.Size < batchSize {
		return nil // Not enough experience to sample a batch
	}

	samples := make([]Experience, batchSize)
	for i := range batchSize {
		idx := rand.Intn(item.Size)
		samples[i] = item.Experiences[idx]
	}
	return samples
}
