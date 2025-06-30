package main

import "fmt"

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

// SwitchPlayer switches the current player.
func (item *Board) SwitchPlayer() {
	item.CurrentPlayer = -item.CurrentPlayer
}

// GetReward returns the reward for the agent based on the game outcome.
// This function is called AFTER a move has been made and the game state potentially changed.
func (item *Board) GetReward(agentPlayer int) float64 {
	isOver, winner := item.GetGameOutcome()
	if isOver {
		switch winner {
		case agentPlayer:
			return winsReward // Agent wins
		case Empty:
			return drawReward // Draw
		default: // winner == -agentPlayer (opponent)
			return losesReward // Agent loses (opponent wins)
		}
	}
	return 0.0 // No negative reward for moves in Tic-Tac-Toe
}

// GetStateVector converts the board state into a vector for the neural network.
// Represents the 3x3 board as a flat 9-element vector.
// 1.0 for agent's cell, -1.0 for opponent's cell, 0.0 for empty.
func (item *Board) GetStateVector(agentPlayer int) []float64 {
	state := make([]float64, 9)
	for i, cell := range item.Cells {
		switch cell {
		case agentPlayer:
			state[i] = 1.0
		case -agentPlayer: // Opponent
			state[i] = -1.0
		default: // Empty
			state[i] = 0.0
		}
	}
	return state
}

// GetEmptyCells returns a list of empty cell indices.
func (item *Board) GetEmptyCells() []int {
	var emptyCells []int
	for i, cell := range item.Cells {
		if cell == Empty {
			emptyCells = append(emptyCells, i)
		}
	}
	return emptyCells // Returns ALL empty cells
}

// MakeMove attempts to make a move at the specified position.
// Returns true if the move was successful, false otherwise.
func (item *Board) MakeMove(pos int) {
	if pos < 0 || pos > 8 || item.Cells[pos] != Empty {
		// error in the algorithm
		panic("Move failed unexpectedly. Invalid board cell: " + fmt.Sprintf("%d", pos))
	}
	item.Cells[pos] = item.CurrentPlayer
}

// CheckBoardFull checks if the board is full.
func (item *Board) CheckBoardFull() bool {
	for _, cell := range item.Cells {
		if cell == Empty {
			return false
		}
	}
	return true
}

// CheckWin checks if the given player has won.
// This function can be called for any player to check the win condition.
func (item *Board) CheckWin(player int) bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Horizontal
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Vertical
		{0, 4, 8}, {2, 4, 6}, // Diagonals
	}

	for _, cond := range winConditions {
		if item.Cells[cond[0]] == player &&
			item.Cells[cond[1]] == player &&
			item.Cells[cond[2]] == player {
			return true
		}
	}
	return false
}

// CheckEarlyDraw checks if the given player can still potentially win on this board.
// It checks if there is any winning line that is not blocked by the opponent
// and still has empty cells where the player could place their marks.
func (item *Board) CheckEarlyDraw(player int) bool {
	winConditions := [][]int{
		{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Horizontal
		{0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Vertical
		{0, 4, 8}, {2, 4, 6}, // Diagonals
	}

	for _, cond := range winConditions {
		isBlockedByOpponent := false
		emptyCellsInLine := 0

		for _, cellIdx := range cond {
			if item.Cells[cellIdx] == -player { // Opponent's mark in this line
				isBlockedByOpponent = true
				break // This line is blocked, cannot win here
			}
			if item.Cells[cellIdx] == Empty { // Empty cell in this line
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
func (item *Board) GetGameOutcome() (bool, int) {
	// First, check for a win
	if item.CheckWin(PlayerX) {
		return true, PlayerX
	}
	if item.CheckWin(PlayerO) {
		return true, PlayerO
	}

	// If no winner, check for a full board (draw)
	if item.CheckBoardFull() {
		return true, Empty // Draw due to full board
	}

	// NEW: Check for an early draw if neither player can win anymore
	if !item.CheckEarlyDraw(PlayerX) && !item.CheckEarlyDraw(PlayerO) {
		return true, Empty // Early draw
	}

	return false, Empty // Game is still ongoing
}

// PrintBoard prints the board to the console.
func (item *Board) PrintBoard() {
	fmt.Println("-------------")
	for i := range 3 {
		fmt.Print("| ")
		for j := range 3 {
			val := item.Cells[i*3+j]
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
