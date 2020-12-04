from chomp import agent
from chomp.rl import experience



def main():
    agent_filenam = 'filename'
    agent1 = agent.load_policy_agent(h5py.File(agent_filename))
    agent2 = agent.load_policy_agent(h5py.File(agent_filename))
    collector1 = experience.ExperienceCollector()
    collector2 = experience.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    num_games = 10
    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()
        game_record = simulate_game(agent1, agent2)
        if game_record.winner == Player.black:
            collector1.complete_episode(reward = 1)
            collector2.complete_episode(reward = -1)
        else:
            collector1.complete_episode(reward = -1)
            collector2.complete_episode(reward = 1)

    experience.rl.combine_experience([collector1, collector2])


if __name__ == '__main__':
	main()