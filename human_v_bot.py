from chomp.agent import naive
from chomp.clobber_board import GameState
from chomp.chomp_types import Player
from chomp.utils import print_board, print_move
from chomp.agent import MCTSAgent
import time

def main():

    board_size = 5
    game = GameState.new_game(board_size, board_size)