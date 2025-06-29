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
	qNet := NewNeuralNetwork(inputSize, []int{27}, outputSize, "tanh") // Example architecture
	targetNet := qNet.Clone()                                          // Clone for the target network

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
func (agent *DQNAgent) ChooseAction(board *Board) int {
	emptyCells := board.GetEmptyCells()
	if len(emptyCells) == 0 {
		return -1 // No available moves
	}
	/*
		if len(emptyCells) == 9 {
			return emptyCells[rand.Intn(len(emptyCells))] // Random FIRST move
		}
	*/
	/*
		if len(emptyCells) == 9 {
			return 4 // FIRST move to center
		}
	*/
	// Epsilon-greedy strategy: random move or best move according to Q-network
	if rand.Float64() < agent.MaxEpsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Random move
	}

	// Choose the best move according to the Q-network
	stateVec := board.GetStateVector(agent.PlayerSymbol)
	qValues := agent.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64 // Initialize with a very small number

	var bestActions []int               // New slice to store all actions with the maximum Q-value
	for _, action := range emptyCells { // Iterate ONLY through empty cells
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestActions = []int{action} // Found a new maximum, clear the list
		} else if qValues[action] == maxQ { // If Q-value equals current maximum
			bestActions = append(bestActions, action) // Add to the list
			fmt.Println("*")
		}
	}

	// If multiple actions are found with the same maximum Q-value,
	// choose one of them randomly.
	if len(bestActions) > 0 {
		bestAction = bestActions[rand.Intn(len(bestActions))]
	} else {
		// This case should be very rare if emptyCells is not empty.
		// If it happens, choose a random available move as a fallback.
		if len(emptyCells) > 0 {
			bestAction = emptyCells[rand.Intn(len(emptyCells))]
		} else {
			return -1 // No available moves
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
		// Predicted Q-values for the current state from the Q-network
		currentQValues := agent.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Copy to modify only one value

		// Calculate the target Q-value
		var targetQ float64
		if exp.Done {
			targetQ = exp.Reward // If the game is over, the target value is the immediate reward
		} else {
			// --- Double DQN Modification ---
			// 1. Get Q-values for the next state from the Q-network (to choose the best action)
			qValuesNextStateFromQNetwork := agent.QNetwork.Predict(exp.NextState)

			// Find the action that would be chosen by the Q-network in the next state.
			// Here it's assumed the network will learn to assign low Q-values to invalid moves.
			bestActionFromQNetwork := -1
			maxQValFromQNetwork := -math.MaxFloat64

			// Find the index of the best action from the Q-network's predictions.
			for i, qVal := range qValuesNextStateFromQNetwork {
				if qVal > maxQValFromQNetwork {
					maxQValFromQNetwork = qVal
					bestActionFromQNetwork = i
				}
			}

			// Fallback for `bestActionFromQNetwork` in case all predicted Q-values
			// are equal to `maxQValFromQNetwork` (e.g., all -math.MaxFloat64 at very early stages).
			// In a well-trained network, `bestActionFromQNetwork` should always be a valid action.
			// If it remains -1, it indicates a problem with the network's predictions.
			if bestActionFromQNetwork == -1 {
				bestActionFromQNetwork = rand.Intn(9) // Fallback: random action (unlikely)
			}

			// 2. Evaluate the Q-value of the chosen action using the Target Network
			qValueFromTargetNetwork := agent.TargetNetwork.Predict(exp.NextState)[bestActionFromQNetwork]

			targetQ = exp.Reward + agent.Gamma*qValueFromTargetNetwork // Bellman Equation (DDQN)
			// --- End of Double DQN Modification ---
		}

		// Update the target Q-value for the action taken in this experience
		targetQValues[exp.Action] = targetQ

		// Train the Q-network with the updated target Q-values
		agent.QNetwork.Train(exp.State, targetQValues, agent.LearningRate)
	}

	// Decay epsilon (applied per training step, not per episode)
	if agent.MaxEpsilon > agent.MinEpsilon {
		agent.MaxEpsilon *= agent.EpsilonDecay
	}

	// Update the target network
	if step%agent.UpdateTarget == 0 {
		agent.TargetNetwork = agent.QNetwork.Clone()
		fmt.Printf("--- Target network updated at step %d (Epsilon: %.4f) ---\n", step, agent.MaxEpsilon)
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
