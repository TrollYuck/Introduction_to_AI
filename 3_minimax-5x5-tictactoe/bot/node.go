package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func getBestMove(board *[5][5]int, depth, player int) int {
	var is_maxminizing bool // 1 -> X maxing, 2 -> O mining
	var best_moves [][2]int
	var best_eval, curr_eval, ret int
	if player == 1 {
		is_maxminizing = true
		best_eval = -math.MaxInt16
	} else {
		is_maxminizing = false
		best_eval = math.MaxInt16
	}

	posMoves := possibleMoves(*board)

	if m, ok := immediateWin(board, posMoves, player); ok {
		fmt.Println("Immediate Win on: ", (m[0]+1)*10+m[1]+1)
		return (m[0]+1)*10 + m[1] + 1
	}

	opp := 3 - player
	cp_board := *board
	m, opp_win := immediateWin(board, posMoves, opp)
	cp_board[m[0]][m[1]] = player
	p_lose, _ := loseCheck(cp_board)
	if opp_win && !(p_lose) {
		fmt.Println("Immediate Win BLOCKED on: ", (m[0]+1)*10+m[1]+1)
		return (m[0]+1)*10 + m[1] + 1
	}

	for _, move := range posMoves {
		row, col := move[0], move[1]
		board[row][col] = player
		curr_eval = minimax(board, depth-1, -math.MaxInt16, math.MaxInt16, is_maxminizing)
		board[row][col] = 0

		if is_maxminizing { // Current player is maximizing
			if curr_eval > best_eval {
				best_eval = curr_eval
				best_moves = [][2]int{move}
			} else if curr_eval == best_eval {
				best_moves = append(best_moves, move)
			}
		} else { // Current player is minimizing
			if curr_eval < best_eval {
				best_eval = curr_eval
				best_moves = [][2]int{move}
			} else if curr_eval == best_eval {
				best_moves = append(best_moves, move)
			}
		}
	}

	// Pick a random move from best_moves
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	chosen := best_moves[rng.Intn(len(best_moves))]
	ret = (chosen[0]+1)*10 + chosen[1] + 1
	fmt.Println(best_eval, " <-score | move ->", ret)
	return ret
}

// minimax, X is maxing
func minimax(board *[5][5]int, depth, alpha, beta int, is_maxminizing bool) int {
	var max_eval, min_eval int
	game_over, end_eval := isGameOver(*board)

	if depth == 0 || game_over {
		if game_over {
			return end_eval
		}
		return evaluateNode(*board)
	}

	moves := possibleMoves(*board)

	if is_maxminizing {
		max_eval = -math.MaxInt16
		for _, move := range moves {
			row, col := move[0], move[1]
			// fmt.Println(row, " <r c> ", col, " max")

			board[row][col] = 1 // Apply move for player X
			eval := minimax(board, depth-1, alpha, beta, false)
			board[row][col] = 0 // Undo move

			max_eval = max(max_eval, eval)
			alpha = max(alpha, eval)
			if beta <= alpha {
				break
			}
		}
		return max_eval
	} else {
		min_eval = math.MaxInt16
		for _, move := range moves {
			row, col := move[0], move[1]

			// fmt.Println(row, " <r c> ", col, " min")

			board[row][col] = 2 // Apply move for player O
			eval := minimax(board, depth-1, alpha, beta, true)
			board[row][col] = 0 // Undo move

			min_eval = min(min_eval, eval)
			beta = min(beta, eval)
			if beta <= alpha {
				break
			}
		}
		return min_eval
	}
}

// returns list of possible moves for a player in certain state
func possibleMoves(board [5][5]int) [][2]int {
	var moves [][2]int
	for r := range 5 {
		for c := range 5 {
			if board[r][c] == 0 { // 0 represents an empty cell
				moves = append(moves, [2]int{r, c})
			}
		}
	}
	return moves
}

func evaluateNode(board [5][5]int) int {
	score := 0
	patternFound := false

	// Helper function to check line patterns
	checkLine := func(line [5]int) {
		// Two in a row
		for i := range 4 {
			if line[i] == 1 && line[i+1] == 1 {
				score += 5
				patternFound = true
			}
			if line[i] == 2 && line[i+1] == 2 {
				score -= 5
				patternFound = true
			}
		}
		// X 0 X pattern
		for i := range 3 {
			if line[i] == 1 && line[i+1] == 0 && line[i+2] == 1 {
				score += 3
				patternFound = true
			}
			if line[i] == 2 && line[i+1] == 0 && line[i+2] == 2 {
				score -= 3
				patternFound = true
			}
		}
		// X 0 0 X pattern
		for i := range 2 {
			if line[i] == 1 && line[i+1] == 0 && line[i+2] == 0 && line[i+3] == 1 {
				score += 2
				patternFound = true
			}
			if line[i] == 2 && line[i+1] == 0 && line[i+2] == 0 && line[i+3] == 2 {
				score -= 2
				patternFound = true
			}
		}
		// X X 0 X / X 0 X X pattern
		for i := range 2 {
			if line[i] == 1 && line[i+1] == 1 && line[i+2] == 0 && line[i+3] == 1 {
				score += 350
				patternFound = true
			}
			if line[i] == 2 && line[i+1] == 2 && line[i+2] == 0 && line[i+3] == 2 {
				score -= 350
				patternFound = true
			}
			if line[i] == 1 && line[i+1] == 0 && line[i+2] == 1 && line[i+3] == 1 {
				score += 350
				patternFound = true
			}
			if line[i] == 2 && line[i+1] == 0 && line[i+2] == 2 && line[i+3] == 2 {
				score -= 350
				patternFound = true
			}
		}

		for i := 0; i <= len(line)-4; i++ {
			// Player X (1) splits three of O's (2) pieces - good for X
			// O O X O pattern: line[i]=O, line[i+1]=O, line[i+2]=X, line[i+3]=O
			if line[i] == 2 && line[i+1] == 2 && line[i+2] == 1 && line[i+3] == 2 {
				score += 400 // X (player 1) splits three O's
				patternFound = true
			}
			// O X O O pattern: line[i]=O, line[i+1]=X, line[i+2]=O, line[i+3]=O
			if line[i] == 2 && line[i+1] == 1 && line[i+2] == 2 && line[i+3] == 2 {
				score += 400 // X (player 1) splits three O's
				patternFound = true
			}

			// Player O (2) splits three of X's (1) pieces - bad for X (good for O)
			// X X O X pattern: line[i]=X, line[i+1]=X, line[i+2]=O, line[i+3]=X
			if line[i] == 1 && line[i+1] == 1 && line[i+2] == 2 && line[i+3] == 1 {
				score -= 400 // O (player 2) splits three X's
				patternFound = true
			}
			// X O X X pattern: line[i]=X, line[i+1]=O, line[i+2]=X, line[i+3]=X
			if line[i] == 1 && line[i+1] == 2 && line[i+2] == 1 && line[i+3] == 1 {
				score -= 400 // O (player 2) splits three X's
				patternFound = true
			}
		}

		// Strongly unfavor X 0 X 0 X pattern
		if len(line) == 5 {
			if line[0] == 1 && line[1] == 0 && line[2] == 1 && line[3] == 0 && line[4] == 1 {
				score -= 300
				patternFound = true
			}
			if line[0] == 2 && line[1] == 0 && line[2] == 2 && line[3] == 0 && line[4] == 2 {
				score += 300
				patternFound = true
			}
		}
	}

	// Check rows
	for r := range 5 {
		var row [5]int
		for c := range 5 {
			row[c] = board[r][c]
		}
		checkLine(row)
	}

	// Check columns
	for c := range 5 {
		var col [5]int
		for r := range 5 {
			col[r] = board[r][c]
		}
		checkLine(col)
	}

	// Diagonals (top-left to bottom-right)
	for r := 0; r <= 1; r++ {
		for c := 0; c <= 1; c++ {
			var diag [5]int
			for i := 0; i < 5 && r+i < 5 && c+i < 5; i++ {
				diag[i] = board[r+i][c+i]
			}
			checkLine(diag)
		}
	}

	// Anti-diagonals (top-right to bottom-left)
	for r := 0; r <= 1; r++ {
		for c := 4; c >= 3; c-- {
			var adiag [5]int
			for i := 0; i < 5 && r+i < 5 && c-i >= 0; i++ {
				adiag[i] = board[r+i][c-i]
			}
			checkLine(adiag)
		}
	}

	if patternFound {
		return score
	} else {
		// No patterns found
		positionalScore := 0
		centerBonuses := [5][5]int{
			{0, 0, 1, 0, 0},
			{0, 1, 2, 1, 0},
			{1, 2, 3, 2, 1},
			{0, 1, 2, 1, 0},
			{0, 0, 1, 0, 0},
		}

		for r := range 5 {
			for c := range 5 {
				if board[r][c] == 1 { // Player X (maximizer)
					positionalScore += centerBonuses[r][c]
				} else if board[r][c] == 2 { // Player O (minimizer)
					positionalScore -= centerBonuses[r][c]
				}
			}
		}

		return positionalScore
	}
}

func isGameOver(board [5][5]int) (bool, int) {

	game_over, p := drawCheck(board)

	if game_over {
		return true, 0
	}

	game_over, p = winCheck(board)
	if game_over {
		if p == 1 {
			return true, 1000
		} else {
			return true, -1000
		}
	}

	game_over, p = loseCheck(board)
	if game_over {
		if p == 1 {
			return true, -1000
		} else {
			return true, 1000
		}
	}
	return false, 0
}

func immediateWin(board *[5][5]int, posMoves [][2]int, player int) (move [2]int, ok bool) {
	for _, m := range posMoves {
		r, c := m[0], m[1]
		board[r][c] = player
		win, _ := winCheck(*board)
		board[r][c] = 0
		if win {
			return m, true
		}
	}
	return [2]int{}, false
}
