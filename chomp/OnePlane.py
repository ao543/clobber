import numpy as np
from chomp.clobber_board import Move


class OnePlane:

    def __init__(self, BOARD_WIDTH, BOARD_HEIGHT):
        #self.board_width, self.board_height = board_size, board_size
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.num_planes = 1
        self.move_dict = [(0,0,0,0) for i in range(BOARD_WIDTH * BOARD_HEIGHT * BOARD_HEIGHT * BOARD_HEIGHT)]


        numb = 0

        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH):
                for k in range(BOARD_HEIGHT):
                    for l in range(BOARD_WIDTH):
                        self.move_dict[numb] = (i, j, k, l)
                        numb = numb + 1

    def name(self):
        return 'OnePlane'

    def encode(self, game_state):
        #Test
        #print("test-v")
        #print(self.shape())
        #print(self.board_height)
        #print(self.board_width)

        board_matrix = np.zeros(self.shape())
        for i in range(self.board_height):
            for j in range(self.board_width):
                board_matrix[0, i, j] = game_state.board.grid[i][j]

        return board_matrix

    def num_points(self):
        return self.board_width * self.board_height

    def encode_move(self, move):
        return self.move_dict.index((move.from_row,move.from_col, move.to_row, move.to_col))

    def shape(self):
        return 1, self.board_height, self.board_width

    def decode_move(self, one_hot):
        hot_index = np.argmax(one_hot)
        mov = self.move_dict[hot_index]
        decoded_move = Move(mov[0], mov[1], mov[2], mov[3])
        return decoded_move

    def decode_move_int(self, move_int):

        #Test
        #print(self.move_dict)

        hot_index = move_int
        mov = self.move_dict[hot_index]
        decoded_move = Move(mov[0], mov[1], mov[2], mov[3])
        return decoded_move



