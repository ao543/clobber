from chomp.chomp_types import Player

class Move:
	def __init__(self, from_row, from_col, to_row, to_col):
		self.from_row = from_row
		self.from_col = from_col
		self.to_row = to_row
		self.to_col = to_col

	def print_mov(self):
		print("(" + str(self.from_row) + "," + str(self.from_col) + ")->" +  "(" + str(self.to_row) + "," + str(self.to_col) + ")")


class Board():

	def __init__(self, num_rows = 2, num_cols = 2):
		#1 is board, 2 critical, 0 taken
		#1 is bob, .5 alice 0 empty
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.grid = [[1 for i in range(num_cols)] for j in range(num_rows)]

		for i in range(num_rows):
			for j in range(num_cols):
				if (i % 2 == 0):
					self.grid[i][j] = 1
				if (i % 2 == 1):
					self.grid[i][j] = .5




	def clone(self):
		num_rows = self.num_rows
		num_cols = self.num_cols
		new_Board = Board(num_rows, num_cols)
		#new_grid = [[1 for i in range(num_cols)] for j in range(num_rows)]
		for i in range(num_rows):
			for j in range(num_cols):
				new_Board.grid[i][j] = self.grid[i][j]

		return new_Board





	def make_move(self, move):
		self.grid[move.to_row][move.to_col] = self.grid[move.from_row][move.from_col]
		self.grid[move.from_row][move.from_col] = 0



class GameState():
	def __init__(self, board, next_player, previous_player):
		self.board = board
		self.next_player = next_player
		self.previous_player = previous_player


	def get_winner(self):
		return self.previous_player


	def apply_move(self, move):
		self.board.make_move(move)
		self.previous_player = self.previous_player.other
		self.next_player = self.next_player.other

	def clone(self):
		#not sure about correct enum handling here
		return GameState(self.board.clone(), self.next_player, self.previous_player)


	@classmethod
	def new_game(self, row_size = 2, col_size = 2):
		board = Board(row_size, col_size)
		return GameState(board, Player.alice, Player.bob)


	def is_over(self):
		return len(self.get_valid_moves()) == 0

	def is_valid_move(self, move):
		#Alice = 1, Bob = 2
		if(self.next_player == Player.alice):
			if(self.board.grid[move.from_row][move.from_col] != 1):
				return False
			if(self.board.grid[move.to_row][move.to_col] != .5):
				return False
			if( abs((move.to_row - move.from_row)) + abs((move.to_col - move.from_col)) != 1  ):
				return False
		if(self.next_player == Player.bob):
			if(self.board.grid[move.from_row][move.from_col] != .5):
				return False
			if(self.board.grid[move.to_row][move.to_col] != 1):
				return False
			if( abs((move.to_row - move.from_row)) + abs((move.to_col - move.from_col)) != 1  ):
				return False
		return True

	def get_valid_moves(self):
		return [Move(a, b, c, d) for a in range(self.board.num_rows) for b in range(self.board.num_cols)
				for c in range(self.board.num_rows) for d in range(self.board.num_cols) if self.is_valid_move(Move(a, b, c, d))]


