import enum



class Player(enum.Enum):
	alice = 1
	bob = 2

	@property
	def other(self):
		return Player.alice if self == Player.bob else Player.bob


