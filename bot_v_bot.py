
from chomp.agent import naive 
from chomp.chomp_board import GameState
from chomp.chomp_types import Player
from chomp.utils import print_board, print_move
import time 


def main():

	board_size = 5
	game = GameState.new_game(board_size, board_size)
	bots = {Player.alice: naive.RandomBot(), Player.bob: naive.RandomBot()}

	while not game.is_over():
		time.sleep(.3)
		print(chr(27) + "[2J")    
		print_board(game.board)
		bot_move = bots[game.next_player].select_move(game)
		print_move(game.next_player, bot_move)
		game.apply_move(bot_move)

	print_board(game.board)

if __name__ == '__main__':
	main()