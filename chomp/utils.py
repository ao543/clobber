

def print_move(player, move):
	print("")
	print(player)
	move.print_mov()




def print_board(board):
	for r in range(board.num_rows):
		print("")
		for c in range(board.num_cols):
			print(board.grid[r][c], end='')


