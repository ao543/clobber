import os
from random import random

import h5py

from chomp.OnePlane import OnePlane
from chomp.clobber_board import GameState
from chomp.chomp_types import Player
from chomp.agent.pg import PolicyAgent
from chomp.rl import experience
from chomp.rl.experience import ExperienceCollector, combine_experience, load_experience, ExperienceBuffer
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D
from chomp.agent.pg import PolicyAgent
import random

def simulate_game(alice, bob, BOARD_WIDTH, BOARD_HEIGHT):
    #black= 1st, white = second
    #BOARD_SIZE = 2
    game = GameState.new_game(row_size = BOARD_HEIGHT, col_size = BOARD_WIDTH)
    agents = {Player.alice: alice, Player.bob: bob}


    while not game.is_over():


        next_move = agents[game.next_player].select_move(game)
        #game = game.apply_move(next_move)
        game.apply_move(next_move)



    return game.get_winner()

def base_model(BOARD_WIDTH, BOARD_HEIGHT):

    input_shape = (BOARD_HEIGHT, BOARD_WIDTH,  1)
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3,3),  activation='relu', padding = 'same',input_shape = input_shape ) )
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(48, kernel_size=(3,3), padding='same', activation='relu') )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(BOARD_WIDTH * BOARD_HEIGHT * BOARD_WIDTH * BOARD_HEIGHT, activation='softmax'))
    #model.summary()
    return model

def generate_experience(iteration, BOARD_WIDTH, BOARD_HEIGHT, num_games = 1000):
    #Test filename
    # bots = {Player.alice: naive.RandomBot(), Player.bob: naive.RandomBot()}

    if iteration != 0:
        alice_filename = 'alice' + str(iteration - 1) + '.hdf5'
        bob_filename = 'bob' + str(iteration - 1) + '.hdf5'
        f1 = h5py.File(alice_filename, 'a')
        f2 = h5py.File(bob_filename, 'a')
        agent1 = PolicyAgent.load_policy_agent(f1)
        agent2 = PolicyAgent.load_policy_agent(f2)
        #agent2 = naive.RandomBot()
    else:
        game_encoder = OnePlane(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)
        agent1 = PolicyAgent(game_encoder, base_model(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT))
        agent2 = PolicyAgent(game_encoder, base_model(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT))


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

        game_record = simulate_game(agent1, agent2, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)

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

    #experience = combine_experience([collector1, collector2])
    experience_filename1 = 'experience_alice' + str(iteration) + '.hdf5'
    experience_filename2 = 'experience_bob' + str(iteration) + '.hdf5'

    experience_buff1 = combine_experience([collector1])
    experience_buff2 = combine_experience([collector2])

    with h5py.File(experience_filename1, 'w') as experience_outf:
        experience_buff1.serialize(experience_outf)

    with h5py.File(experience_filename2, 'w') as experience_outf:
        experience_buff2.serialize(experience_outf)

def learn_from_experience(iteration, BOARD_WIDTH, BOARD_HEIGHT):

    #Sets defaults here
    batch_size = 32
    learning_rate = .0001
    #look into below
    clipnorm = 1.0
    #updated_agent_filename = 'agent' + str(iteration) + '.hdf5'
    exp_filename1 = '/Users/andrew/Desktop/clobber_proj/chomp/agent/experience_alice' + str(iteration) +'.hdf5'
    exp_filename2 = '/Users/andrew/Desktop/clobber_proj/chomp/agent/experience_bob' + str(iteration) + '.hdf5'

    if iteration == 0:
        learning_agent1 = PolicyAgent(OnePlane(BOARD_WIDTH, BOARD_HEIGHT), base_model(BOARD_WIDTH, BOARD_HEIGHT))
        learning_agent2 = PolicyAgent(OnePlane(BOARD_WIDTH, BOARD_HEIGHT), base_model(BOARD_WIDTH, BOARD_HEIGHT))
    else:
        #agent_filename = 'agent' + str(iteration - 1) + '.hdf5'
        agent_filename1 = 'alice' + str(iteration - 1) + '.hdf5'
        agent_filename2 = 'bob' + str(iteration - 1) + '.hdf5'
        learning_agent1 = PolicyAgent.load_policy_agent(h5py.File(agent_filename1))
        learning_agent2 = PolicyAgent.load_policy_agent(h5py.File(agent_filename2))


    expr_file1 = h5py.File(exp_filename1)
    expr_file2 = h5py.File(exp_filename2)

    exp_buffer1 = load_experience(expr_file1)
    exp_buffer2 = load_experience(expr_file2)


    exp_buffer1.states = np.squeeze(exp_buffer1.states, axis = 1)
    exp_buffer2.states = np.squeeze(exp_buffer2.states, axis=1)

    learning_agent1.train(exp_buffer1, lr=learning_rate, batch_size=batch_size)
    learning_agent2.train(exp_buffer2, lr=learning_rate, batch_size=batch_size)

    updated_agent_filename1 = 'alice' + str(iteration) + '.hdf5'
    updated_agent_filename2 = 'bob' + str(iteration) + '.hdf5'

    with h5py.File(updated_agent_filename1, 'w') as updated_agent_outf:
        learning_agent1.serialize(updated_agent_outf)

    with h5py.File(updated_agent_filename2, 'w') as updated_agent_outf:
        learning_agent2.serialize(updated_agent_outf)

def compute_self_play_stats(iteration, BOARD_WIDTH, BOARD_HEIGHT,  num_games = 10000):

    agent_filename1 = 'alice' + str(iteration) + '.hdf5'
    agent_filename2 = 'bob' + str(iteration) + '.hdf5'

    if iteration == 0:
        learning_agent1 = PolicyAgent(OnePlane(BOARD_WIDTH, BOARD_HEIGHT), base_model(BOARD_WIDTH, BOARD_HEIGHT))
        learning_agent2 = PolicyAgent(OnePlane(BOARD_WIDTH, BOARD_HEIGHT), base_model(BOARD_WIDTH, BOARD_HEIGHT))
    else:
        learning_agent1 = PolicyAgent.load_policy_agent(h5py.File(agent_filename1))
        learning_agent2 = PolicyAgent.load_policy_agent(h5py.File(agent_filename2))

    win_record = {Player.alice: 0, Player.bob: 0}

    for i in range(num_games):

        game_record = simulate_game(learning_agent1, learning_agent2, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)

        if game_record == Player.alice:
            win_record[Player.alice] += 1
        else:
            win_record[Player.bob] += 1

    return (win_record[Player.alice], win_record[Player.bob])
    #print("Alice wins: " + str(win_record[Player.alice]))
    #print("Bob wins: " + str(win_record[Player.bob]))

def learning_cycle(BOARD_WIDTH, BOARD_HEIGHT, cycles):
    results = {}

    for i in range(cycles):
        if i == 0:
            generate_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            learn_from_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            results[i] = compute_self_play_stats(iteration = i, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)
        else:
            generate_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            learn_from_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            results[i] = compute_self_play_stats(iteration = i, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)

    print(results)




def clean_directory():
    path = '/Users/andrew/Desktop/clobber_proj/chomp/agent'
    dir = os.listdir(path)
    i = 0
    for file in dir:
        if file.endswith('hdf5'):
            os.remove(file)




if __name__ == '__main__':
    random.seed(34)
    #Experiment for lemma 3.4-magnitude 10s
    #x = random.randrange(10, 100, 2)
    #print(x)
    #learning_cycle(BOARD_WIDTH = 8, BOARD_HEIGHT = 2, cycles= 100)
    #clean_directory()