

def print_move(player, move):
	print("")
	print(player)
	print("(" +str(move.row) + "," + str(move.col) + ")" )


def print_board(board):
	for r in range(board.num_rows):
		print("")
		for c in range(board.num_cols):
			print(board.grid[r][c], end="")


