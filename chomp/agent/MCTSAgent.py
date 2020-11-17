import math

from chomp.agent.base import Agent
from chomp.agent.MCTSNode import MCTSNode
from chomp.chomp_types import Player
from chomp.agent import naive


class MCTSAgent(Agent):

    def __init__(self, num_rounds, temperature):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):

        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root

            while (not node.can_add_children()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_children():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.game_state)
            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_pct = -1.0



        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)



            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def uct_score(self, parent_rollouts, child_rollouts, win_pct, temperature):
        exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
        return win_pct + temperature * exploration

    def select_child(self, node):
        total_rollouts = sum([child.num_rollouts for child in node.children])
        best_score = -1
        best_child = None
        for child in node.children:
            score = self.uct_score(total_rollouts, child.num_rollouts, child.winning_frac(node.game_state.next_player),
                                   self.temperature)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):

        bots = {Player.alice: naive.RandomBot(), Player.bob: naive.RandomBot()}

        #Test
        #print("test")
        #print(game)
        #print(game.is_over())

        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game.apply_move(bot_move)
            #game = game.apply_move(bot_move)

        #print("test2")
        return game.get_winner()
