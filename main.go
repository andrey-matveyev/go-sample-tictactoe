package main

import (
	"fmt"
	"math/rand"
)

// --- Parameters ---

const (
	// Who makes the first move (first step)?
	agentsFirstStep bool = false // true = agent (PlayerX), false = opponent (PlayerO)
	// Training parameters
	episodes           int = 500000 // Number of game episodes for training
	batchSize          int = 10      // Batch size for DQN training
	bufferCapacity     int = 50000  // Experience buffer capacity
	trainStartSize     int = 1000   // Start training after accumulating enough experience
	// Learning parameters for DQNAgent
	gamma        float64 = 0.75     // Discount factor
	maxEpsilon   float64 = 1.0      // Start with exploration
	minEpsilon   float64 = 0.0002   // Minimum epsilon value
	epsilonDecay float64 = 0.999996 // Epsilon decay rate per step (very slow)
	learningRate float64 = 0.0002   //
	updateTarget int     = 50000    // Update target network every 10000 steps (less frequently)
	// Reward parameters
	winsReward  float64 = 0.999
	drawReward  float64 = 0.001
	losesReward float64 = -1.000
	// Hidden layer size
	hiddenLayerSize int = 27
)

// --- Main training loop ---

func main() {
	// Create a DQN agent (plays as X)
	dqnAgentX := NewDQNAgent(9, 9, bufferCapacity, PlayerX)

	totalSteps := 0
	winsX := 0
	winsO := 0 // Wins for O - these are losses for X
	draws := 0

	fmt.Println("Starting DQN agent training (X) against a random opponent (O) for Tic-Tac-Toe...")

	maxW := 0

	for episode := range episodes {
		board := NewBoard() // Board.CurrentPlayer defaults to PlayerX

		// --- If opponent to move first ---
		if !agentsFirstStep {
			board.SwitchPlayer()
			board.MakeMove(rand.Intn(8))
			board.SwitchPlayer()
		}

		isDone := false
		var gameWinner int // To store the winner of the episode

		for !isDone {
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
		}

		// Episode summary - use 'gameWinner' variable from GetGameOutcome
		switch gameWinner {
		case PlayerX:
			winsX++
		case PlayerO:
			winsO++
		default:
			draws++ // gameWinner == Empty (draw)
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
			fmt.Printf("Episode: %d, Wins X: %d (%d), Losses X: %d, Draws: %d, Epsilon X: %.4f, Q(start): %.4f|%.4f|%.4f  %.4f[%.4f]%.4f  %.4f|%.4f|%.4f\n",
				episode+1, winsX, maxW, winsO, draws, dqnAgentX.MaxEpsilon,
				qValuesForEmptyBoard[0], qValuesForEmptyBoard[1], qValuesForEmptyBoard[2],
				qValuesForEmptyBoard[3], qValuesForEmptyBoard[4], qValuesForEmptyBoard[5],
				qValuesForEmptyBoard[6], qValuesForEmptyBoard[7], qValuesForEmptyBoard[8])

			winsX = 0
			winsO = 0
			draws = 0
		}
	}

	fmt.Println("\nTraining complete.")
	fmt.Println("Testing the agent (X against random O)...")

	// Test the trained agent against a random opponent
	TestAgentAfterTraining(dqnAgentX)

	// Example game after training
	ExampleGameAfterTraining(dqnAgentX)
}

func TestAgentAfterTraining(dqnAgentX *DQNAgent) {
	testGames := 1000
	testWinsX := 0
	testDraws := 0
	testLossesX := 0

	// Set epsilon to minimum for testing
	dqnAgentX.MaxEpsilon = 0.0

	for range testGames {
		board := NewBoard()

		// --- If opponent (PlayerO) to move first ---
		if !agentsFirstStep {
			board.SwitchPlayer()
			board.MakeMove(rand.Intn(8))
			board.SwitchPlayer()
		}

		isDone := false
		var gameWinner int // To store the winner of the test game

		for !isDone {
			var action int
			// Current player's turn
			if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
				action = dqnAgentX.ChooseAction(board)
				if action == -1 {
					//isDone, gameWinner = board.GetGameOutcome() // Should be a draw
					break
				}
				board.MakeMove(action)
			} else {
				// Opponent's move (O)
				emptyCells := board.GetEmptyCells()
				if len(emptyCells) == 0 {
					//isDone, gameWinner = board.GetGameOutcome() // Should be a draw
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
		switch gameWinner {
		case PlayerX:
			testWinsX++
		case PlayerO:
			testLossesX++
		default: // gameWinner == Empty (draw)
			testDraws++
		}
	}

	fmt.Printf("\nTest Results (%d games, Agent X vs random O):\n", testGames)
	fmt.Printf("Agent X Wins: %d\n", testWinsX)
	fmt.Printf("Agent X Losses (Random O Wins): %d\n", testLossesX)
	fmt.Printf("Draws: %d\n", testDraws)
}

func ExampleGameAfterTraining(dqnAgentX *DQNAgent) {
	fmt.Println("\nExample game after training (X vs random O):")
	board := NewBoard()
	dqnAgentX.MaxEpsilon = 0.0 // Ensure agent plays optimally

	// --- If opponent (PlayerO) to move first ---
	if !agentsFirstStep {
		board.SwitchPlayer()
		board.MakeMove(rand.Intn(8))
		board.SwitchPlayer()
	}

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

		if board.CurrentPlayer == dqnAgentX.PlayerSymbol {
			action = dqnAgentX.ChooseAction(board)
			if action == -1 {
				//isOver, winner = board.GetGameOutcome()
				break
			}
			board.MakeMove(action)
		} else { // Random opponent's move
			emptyCells := board.GetEmptyCells()
			if len(emptyCells) == 0 {
				//isOver, winner = board.GetGameOutcome()
				break
			}
			action = emptyCells[rand.Intn(len(emptyCells))]
			board.MakeMove(action)
		}

		board.SwitchPlayer() // Always switch player after a successful move, before next iteration
	}
}
