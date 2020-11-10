import random 
from chomp.agent.base import Agent 

class RandomBot(Agent):
	def select_move(self, game_state):
		return random.choice(game_state.get_valid_moves() )
