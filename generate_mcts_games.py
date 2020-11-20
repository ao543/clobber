import argparse
import numpy as np
from chomp.OnePlane import OnePlane
from chomp.chomp_board import GameState
from chomp.agent import MCTSAgent
from chomp.utils import print_board, print_move

def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []
    encoder = OnePlane(board_size)
    game = GameState.new_game(board_size, board_size)
    bot = MCTSAgent.MCTSAgent(rounds, temperature)
    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        boards.append(encoder.encode(game))
        move_one_hot = np.zeros(encoder.num_points())
        move_one_hot[encoder.encode_move(move)] = 1
        moves.append(move_one_hot)
        print_move(game.next_player, move)
        game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=5)
    parser.add_argument('--rounds', '-r', type=int, default=9)
    parser.add_argument('--temperature', '-t', type=float, default=.8)
    parser.add_argument('--max-move', '-m', type=int, default = 60, help='max moves per game')
    parser.add_argument('--num-games', '-n', type=int, default = 10)
    parser.add_argument('--board-out', default = 'features.npy')
    parser.add_argument('--move-out', default = 'labels.npy')
    args = parser.parse_args()

    xs = []
    ys = []

    for i in range(args.num_games):
        print('Generating game %d/%d ...' % (i + 1, args.num_games))
        x, y = generate_game(args.board_size, args.rounds, args.max_move, args.temperature)
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    np.save(args.board_out, x)
    np.save(args.move_out, y)


if __name__ == '__main__':
	main()







