package main

import (
	"fmt"
	"net"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) < 6 {
		fmt.Println("Usage: <numer ip> <numer portu> <gracz> <nick> <depth>")
		os.Exit(1)
	}

	player_no, err := strconv.Atoi(os.Args[3])
	if err != nil || (player_no != 1 && player_no != 2) {
		fmt.Println("Invalid player number. Must be 1 (X) or 2 (O).")
		os.Exit(1)
	}

	depth, err := strconv.Atoi(os.Args[5])
	if err != nil || depth <= 0 {
		fmt.Println("Invalid depth")
		os.Exit(1)
	}

	player_name := os.Args[4]
	server_address := os.Args[1] + ":" + os.Args[2]
	conn, err := net.Dial("tcp", server_address)
	if err != nil {
		fmt.Println("Unable to connect:", err)
		os.Exit(1)
	}
	defer conn.Close()
	fmt.Println("Connected to server at", server_address)

	// Receive initial server message
	buf := make([]byte, 16)
	n, err := conn.Read(buf)
	if err != nil {
		fmt.Println("Error receiving server message:", err)
		return
	}
	serverMsg := string(buf[:n])
	fmt.Println("Server message:", serverMsg)

	// Send player info
	playerMsg := fmt.Sprintf("%d %s", player_no, player_name)
	_, err = conn.Write([]byte(playerMsg))
	if err != nil {
		fmt.Println("Unable to send player info:", err)
		return
	}

	setBoard()
	// Main loop
	end_game := false
	for !end_game {
		buf := make([]byte, 16)
		n, err := conn.Read(buf)
		if err != nil {
			fmt.Println("Error while receiving server's message")
			break
		}
		msgStr := string(buf[:n])
		msg, err := strconv.Atoi(msgStr)
		if err != nil {
			fmt.Println("Error converting server message to int:", err)
			break
		}
		move := msg % 100
		msg = msg / 100

		// println(move, " and msg: ", msg)

		if move != 0 {
			setMove(move, 3-player_no)
			printBoard()
		}

		if msg == 0 || msg == 6 {
			// TODO: minimax logic
			moveToSend := getBestMove(&board, depth, player_no)

			setMove(moveToSend, player_no)
			printBoard()
			_, err = conn.Write(fmt.Appendf(nil, "%d", moveToSend))
			if err != nil {
				fmt.Println("Error sending move:", err)
				end_game = true
			}

		} else {
			end_game = true
			switch msg {
			case 1:
				fmt.Println("You won.")
			case 2:
				fmt.Println("You lost.")
			case 3:
				fmt.Println("Draw.")
			case 4:
				fmt.Println("You won. Opponent error.")
			case 5:
				fmt.Println("You lost. Your error.")
			}
		}
	}

	conn.Close()
}
