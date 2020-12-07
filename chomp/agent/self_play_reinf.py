import h5py

from chomp.OnePlane import OnePlane
from chomp.chomp_board import GameState
from chomp.chomp_types import Player
from chomp.agent.pg import PolicyAgent
from chomp.rl.experience import ExperienceCollector, combine_experience, load_experience
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D
from chomp.agent.pg import PolicyAgent

def simulate_game(alice, bob):
    #black= 1st, white = second
    BOARD_SIZE = 2
    game = GameState.new_game(BOARD_SIZE, BOARD_SIZE)
    agents = {Player.alice: alice, Player.bob: bob}

    #Test
    #print("test")
    #print(game.board.grid)

    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        #game = game.apply_move(next_move)
        game.apply_move(next_move)

    return game.get_winner()

def base_model():
    size = 2
    input_shape = (size, size, 1)
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3,3),  activation='relu', padding = 'same',input_shape = input_shape ) )
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(48, kernel_size=(3,3), padding='same', activation='relu') )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(size * size, activation='softmax'))
    #model.summary()
    return model

def generate_experience(iteration, num_games = 1000):
    #Test filename
    # bots = {Player.alice: naive.RandomBot(), Player.bob: naive.RandomBot()}

    if iteration != 0:
        agent_filename = 'agent' + str(iteration - 1) + '.hdf5'
        f = h5py.File(agent_filename, 'a')
        agent1 = PolicyAgent.load_policy_agent(f)
        agent2 = PolicyAgent.load_policy_agent(f)
        #agent2 = naive.RandomBot()
    else:
        game_encoder = OnePlane(2)
        agent1 = PolicyAgent(game_encoder, base_model())
        agent2 = PolicyAgent(game_encoder, base_model())


    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        #Test
        #print("lion")
        #agent1.model.summary()

        game_record = simulate_game(agent1, agent2)

        if game_record == Player.alice:
            collector1.complete_episode(reward = 1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward = 1)
            collector1.complete_episode(reward=-1)

    #Test
    #print('hello')
    #print(collector1.states)
    #print(len(collector1.states))
    #print(len(collector1.states[1]))

    experience = combine_experience([collector1, collector2])
    experience_filename = 'experience' + str(iteration) + '.hdf5'
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)

def learn_from_experience(iteration):

    #Sets defaults here
    batch_size = 32
    learning_rate = .0001
    #look into below
    clipnorm = 1.0
    updated_agent_filename = 'agent' + str(iteration) + '.hdf5'
    exp_filename = '/Users/andrew/Desktop/chomp_proj/chomp/agent/experience' + str(iteration) +'.hdf5'

    if iteration == 0:
        BOARD_SIZE = 2
        learning_agent = PolicyAgent(OnePlane(BOARD_SIZE), base_model())
    else:
        agent_filename = 'agent' + str(iteration - 1) + '.hdf5'
        learning_agent = PolicyAgent.load_policy_agent(h5py.File(agent_filename))
    #for exp_filename in experience_files:
    expr_file = h5py.File(exp_filename)
    #Test
    #print("green")
    #print(expr_file)

    exp_buffer = load_experience(expr_file)

    #print("vegetables")
    #print(exp_buffer.states.shape)
    exp_buffer.states = np.squeeze(exp_buffer.states, axis = 1)
    #print(exp_buffer.states.shape)


    learning_agent.train(exp_buffer, lr=learning_rate, batch_size=batch_size)
    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)

def compute_self_play_stats(iteration, num_games = 10000):
    agent_filename = 'agent' + str(iteration) + '.hdf5'

    if iteration == 0:
        BOARD_SIZE = 2
        learning_agent = PolicyAgent(OnePlane(BOARD_SIZE), base_model())
    else:
        learning_agent = PolicyAgent.load_policy_agent(h5py.File(agent_filename))

    win_record = {Player.alice: 0, Player.bob: 0}

    for i in range(num_games):

        game_record = simulate_game(learning_agent, learning_agent)

        if game_record == Player.alice:
            win_record[Player.alice] += 1
        else:
            win_record[Player.bob] += 1

    return (win_record[Player.alice], win_record[Player.bob])
    #print("Alice wins: " + str(win_record[Player.alice]))
    #print("Bob wins: " + str(win_record[Player.bob]))

def learning_cycle(cycles = 100):
    results = {}

    for i in range(cycles):
        if i == 0:
            generate_experience(iteration = i)
            learn_from_experience(iteration = i)
            results[i] = compute_self_play_stats(iteration = i)
        else:
            generate_experience(iteration = i)
            learn_from_experience(iteration = i)
            results[i] = compute_self_play_stats(iteration = i)

    print(results)



if __name__ == '__main__':
	#generate_experience(10)
    #learn_from_experience()
    #compute_self_play_stats(is_default=True, learning_agent_filename=None, num_games=10000)
    learning_cycle(cycles=50)