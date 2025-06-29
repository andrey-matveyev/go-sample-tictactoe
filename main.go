package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)




// --- Tic-Tac-Toe game logic ---

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
// This function can be called for any player to check the win condition.
func (b *Board) CheckWin(player int) bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Horizontal
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Vertical
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

// canPlayerWin checks if the given player can still potentially win on this board.
// It checks if there is any winning line that is not blocked by the opponent
// and still has empty cells where the player could place their marks.
func (b *Board) canPlayerWin(player int) bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Horizontal
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Vertical
		{0, 4, 8}, {2, 4, 6}, // Diagonals
	}

	for _, cond := range winConditions {
		isBlockedByOpponent := false
		emptyCellsInLine := 0

		for _, cellIdx := range cond {
			if b.Cells[cellIdx] == -player { // Opponent's mark in this line
				isBlockedByOpponent = true
				break // This line is blocked, cannot win here
			}
			if b.Cells[cellIdx] == Empty { // Empty cell in this line
				emptyCellsInLine++
			}
		}

		// If the line is not blocked by the opponent and has empty cells,
		// the player can still potentially win on this line.
		if !isBlockedByOpponent && emptyCellsInLine > 0 {
			return true
		}
	}
	return false // No winning lines left for this player
}

// GetGameOutcome checks if the game is over and returns the winner, if any.
// Returns (true, winnerSymbol) if the game is over (win or draw), (false, 0) otherwise.
// WinnerSymbol is PlayerX, PlayerO, or Empty (for a draw/game in progress).
func (b *Board) GetGameOutcome() (bool, int) {
	// First, check for a win
	if b.CheckWin(PlayerX) {
		return true, PlayerX
	}
	if b.CheckWin(PlayerO) {
		return true, PlayerO
	}

	// If no winner, check for a full board (draw)
	if b.IsBoardFull() {
		return true, Empty // Draw due to full board
	}

	// NEW: Check for an early draw if neither player can win anymore
	if !b.canPlayerWin(PlayerX) && !b.canPlayerWin(PlayerO) {
		return true, Empty // Early draw
	}

	return false, Empty // Game is still ongoing
}

/*
// GetGameOutcome checks if the game is over and returns the winner, if any.
// Returns (true, winnerSymbol) if the game is over (win or draw), (false, 0) otherwise.
// WinnerSymbol is PlayerX, PlayerO, or Empty (for a draw/game in progress).
func (b *Board) GetGameOutcome() (bool, int) {
	if b.CheckWin(PlayerX) {
		return true, PlayerX
	}
	if b.CheckWin(PlayerO) {
		return true, PlayerO
	}
	if b.IsBoardFull() {
		return true, Empty // This is a draw, no specific winner
	}
	return false, Empty // Game is not yet over
}
*/
// GetReward returns the reward for the agent based on the game outcome.
// This function is called AFTER a move has been made and the game state potentially changed.
func (b *Board) GetReward(agentPlayer int) float64 {
	isOver, winner := b.GetGameOutcome()

	if isOver {
		if winner == agentPlayer {
			return 0.999 // Agent wins
		} else if winner == Empty {
			return 0.001 // Draw
		} else { // winner == -agentPlayer (opponent)
			return -1.000 // Agent loses (opponent wins)
		}
	}
	return 0.0 // No negative reward for moves in Tic-Tac-Toe
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

// GetEmptyCells returns a list of empty cell indices.
func (b *Board) GetEmptyCells() []int {
	var emptyCells []int
	for i, cell := range b.Cells {
		if cell == Empty {
			emptyCells = append(emptyCells, i)
		}
	}
	return emptyCells // Returns ALL empty cells
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
		Gamma:         0.75,     // Discount factor 0.75
		Epsilon:       1.0,      // Start with exploration
		MinEpsilon:    0.0002,   // Minimum epsilon value 0.0002
		EpsilonDecay:  0.999996, // Epsilon decay rate per step (very slow) 0.999996
		LearningRate:  0.0002,   // 0.0002
		UpdateTarget:  50000,    // Update target network every 10000 steps (less frequently)
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
	if rand.Float64() < agent.Epsilon {
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
	episodes := 500000       // Number of game episodes for training
	maxStepsPerEpisode := 10 // Maximum number of steps per episode 10
	batchSize := 8           // Batch size for DQN training 10
	bufferCapacity := 50000  // Experience buffer capacity 5000
	trainStartSize := 1000   // Start training after accumulating enough experience

	// Create a DQN agent (plays as X)
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)

	totalSteps := 0
	winsX := 0
	winsO := 0 // Wins for O - these are losses for X
	draws := 0

	fmt.Println("Starting DQN agent training (X) against a random opponent (O) for Tic-Tac-Toe...")

	maxW := 0

	for episode := 0; episode < episodes; episode++ {
		board := NewBoard() // Board.CurrentPlayer defaults to PlayerX

		// --- Change for opponent to move first ---
		// Opponent (PlayerO) makes the first move
		// Q-values for an empty board state will not be relevant for agent X
		// as its first move will always be a response to O's move
		board.SwitchPlayer()
		board.MakeMove(rand.Intn(8))
		board.SwitchPlayer()
		// --- End of change ---

		isDone := false
		var gameWinner int // To store the winner of the episode
		currentStepInEpisode := 0

		for !isDone && currentStepInEpisode < maxStepsPerEpisode {
			// Save the current state before the move (from Agent X's perspective)
			stateBeforeMove := board.GetStateVector(dqnAgentX.PlayerSymbol)

			var chosenAction int
			// Determine whose turn it is
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				// Agent X's move
				chosenAction = dqnAgentX.ChooseAction(board)

				if chosenAction == -1 { // No available moves, implies a draw
					isDone, gameWinner = board.GetGameOutcome() // Should be a draw
					break
				}

				// Make the move
				board.MakeMove(chosenAction)

				// Check if the game is over IMMEDIATELY after the move
				isDone, gameWinner = board.GetGameOutcome()

				// Board state after agent's move (from Agent X's perspective)
				nextState := board.GetStateVector(dqnAgentX.PlayerSymbol)
				// Reward for Agent X based on the current outcome
				reward := board.GetReward(dqnAgentX.PlayerSymbol)

				// Add experience to Agent X's buffer
				dqnAgentX.ReplayBuffer.Add(Experience{
					State:     stateBeforeMove, // State before action
					Action:    chosenAction,
					Reward:    reward,
					NextState: nextState,
					Done:      isDone,
				})

				totalSteps++
				// Train Agent X
				if dqnAgentX.ReplayBuffer.Size >= trainStartSize {
					dqnAgentX.Train(batchSize, totalSteps)
				}

				if isDone { // Break loop if game is over
					break
				}
			} else { // Opponent's move (random player)
				emptyCells := board.GetEmptyCells()
				var opponentAction int
				if len(emptyCells) > 0 {
					opponentAction = emptyCells[rand.Intn(len(emptyCells))] // Random move
				} else {
					opponentAction = -1 // No available moves, implies a draw
				}

				if opponentAction != -1 {
					board.MakeMove(opponentAction)
				} else {
					isDone, gameWinner = board.GetGameOutcome() // Should be a draw
					break
				}

				// Check if the game is over IMMEDIATELY after the move
				isDone, gameWinner = board.GetGameOutcome()

				if isDone { // Break loop if game is over
					break
				}
			}

			// Switch player ONLY if game is NOT over after the move
			board.SwitchPlayer()
			currentStepInEpisode++
		}

		// Episode summary - use 'gameWinner' variable from GetGameOutcome
		if gameWinner == PlayerX {
			winsX++
		} else if gameWinner == PlayerO {
			winsO++
		} else { // gameWinner == Empty (draw)
			draws++
		}

		if (episode+1)%1000 == 0 {
			// Create a temporary empty board for Q-value evaluation of the first move
			emptyBoardForQEval := NewBoard()
			emptyBoardStateVec := emptyBoardForQEval.GetStateVector(dqnAgentX.PlayerSymbol)
			qValuesForEmptyBoard := dqnAgentX.QNetwork.Predict(emptyBoardStateVec)

			// Index 4 corresponds to the center cell (0-8)
			//qValueCenterCell := qValuesForEmptyBoard[4]
			if maxW < winsX {
				maxW = winsX
			}
			fmt.Printf("Episode: %d, Wins X: %d, Losses X: %d, Draws: %d, Epsilon X: %.4f, Q(start): %.4f|%.4f|%.4f  %.4f[%.4f]%.4f  %.4f|%.4f|%.4f  %d\n",
				episode+1, winsX, winsO, draws, dqnAgentX.Epsilon,
				qValuesForEmptyBoard[0], qValuesForEmptyBoard[1], qValuesForEmptyBoard[2],
				qValuesForEmptyBoard[3], qValuesForEmptyBoard[4], qValuesForEmptyBoard[5],
				qValuesForEmptyBoard[6], qValuesForEmptyBoard[7], qValuesForEmptyBoard[8],
				maxW)

			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nTraining complete.")
	fmt.Println("Testing the agent (X against random O)...")

	// --- Test the trained agent against a random opponent ---
	testGames := 1000
	testWinsX := 0
	testDraws := 0
	testLossesX := 0

	// Set epsilon to minimum for testing
	dqnAgentX.Epsilon = 0.0

	for i := 0; i < testGames; i++ {
		board := NewBoard()

		// --------------------
		// Opponent (PlayerO) makes the first move in the test
		board.SwitchPlayer()
		board.MakeMove(rand.Intn(8))
		board.SwitchPlayer()
		// --------------------

		isDone := false
		var gameWinner int // To store the winner of the test game

		for !isDone {
			var action int
			// Current player's turn
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				action = dqnAgentX.ChooseAction(board)
				if action == -1 {
					isDone, gameWinner = board.GetGameOutcome() // Should be a draw
					break
				}
				board.MakeMove(action)
			} else {
				// Opponent's move (O)
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) == 0 {
					isDone, gameWinner = board.GetGameOutcome() // Should be a draw
					break
				}
				randomAction := emptyCells[rand.Intn(len(emptyCells))]
				board.MakeMove(randomAction)
			}

			// Check if the game is over IMMEDIATELY after the move
			isDone, gameWinner = board.GetGameOutcome()
			if isDone { // Break loop if game is over
				break
			}

			// Switch player ONLY if game is NOT over after the move
			board.SwitchPlayer()
		}

		// Count test game results - use 'gameWinner' variable from GetGameOutcome
		if gameWinner == PlayerX {
			testWinsX++
		} else if gameWinner == PlayerO {
			testLossesX++
		} else { // gameWinner == Empty (draw)
			testDraws++
		}
	}

	fmt.Printf("\nTest Results (%d games, Agent X vs random O):\n", testGames)
	fmt.Printf("Agent X Wins: %d\n", testWinsX)
	fmt.Printf("Agent X Losses (Random O Wins): %d\n", testLossesX)
	fmt.Printf("Draws: %d\n", testDraws)

	// Example game after training
	fmt.Println("\nExample game after training (X vs random O):")
	board := NewBoard()
	dqnAgentX.Epsilon = 0.0 // Ensure agent plays optimally

	// --------------------
	// Opponent (PlayerO) makes the first move in the test
	board.SwitchPlayer()
	board.MakeMove(rand.Intn(8))
	board.SwitchPlayer()
	// --------------------

	for { // Infinite loop until game ends or exceeds max steps
		// Check game outcome at the beginning of the loop iteration (after potential win by previous player)
		isOver, winner := board.GetGameOutcome()
		if isOver {
			board.PrintBoard() // Print final board state
			if winner != Empty {
				fmt.Printf("Game Over! Player %s won!\n", map[int]string{PlayerX: "X", PlayerO: "O"}[winner])
			} else {
				fmt.Println("Game Over! It's a draw!")
			}
			break // Break loop if game is over
		}

		// Print board BEFORE current player's move
		board.PrintBoard()

		var currentPlayerName string
		if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
			currentPlayerName = "X"
		} else {
			currentPlayerName = "O"
		}

		fmt.Printf("%s's Turn:\n", currentPlayerName)
		var action int
		var moveSuccessful bool

		if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
			action = dqnAgentX.ChooseAction(board)
			if action == -1 {
				isOver, winner = board.GetGameOutcome()
				break
			}
			moveSuccessful = board.MakeMove(action)
		} else { // Random opponent's move
			emptyCells := board.GetEmptyCells()
			if len(emptyCells) == 0 {
				isOver, winner = board.GetGameOutcome()
				break
			}
			action = emptyCells[rand.Intn(len(emptyCells))]
			moveSuccessful = board.MakeMove(action)
		}

		if !moveSuccessful {
			fmt.Println("Error: Move failed unexpectedly.")
			break
		}

		board.SwitchPlayer() // Always switch player after a successful move, before next iteration
	}
}
