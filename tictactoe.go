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
/*
	if isOver {
		if winner == agentPlayer {
			return winsReward // Agent wins
		} else if winner == Empty {
			return drawReward // Draw
		} else { // winner == -agentPlayer (opponent)
			return losesReward // Agent loses (opponent wins)
		}
	}
*/
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
