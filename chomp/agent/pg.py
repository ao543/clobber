from chomp.OnePlane import OnePlane
import numpy as np


class PolicyAgent(Agent):

    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder


    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        self.model.save()

    def load_policy_agent(self, h5file):
        model = self.model.load()
        encoder_name = h5file['encoder'].attrs['name']
        board_width = h5file['encoder'].attrs['board_width']
        board_height = h5file['encoder'].attrs['board_height']
        encoder = OnePlane(board_width)
        return PolicyAgent(model, encoder )

    def clip_probs(self, original_probs):
        original_probs = original_probs.numpy()
        min_p = 1e-5
        max_p = 1 - min_p
        clipped_probs = np.clip(original_probs, min_p, max_p)
        clipped_probs = clipped_probs/np.sum(clipped_probs)
        return clipped_probs


    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        move_probs = (self.model.predict(board_tensor))[2]
        move_probs = self.clip_probs(move_probs)
        num_moves = self.encoder.board_height * self.encoder.board_width
        candidates = np.arrange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace = False, p = move_probs)

        for point_idx in ranked_moves:
            move = self.encoder.decode_move_int(point_idx)
            if game_state.is_valid_move(move):
                if self.collector is not None:
                    self.collector.record_decision(state = board_tensor, action = point_idx)
                return move
