package main

import (
	"fmt"
	"math/rand"
	"time"
)

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
