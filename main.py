# import imp
import layout
# from agents import PacmanAgent_QLearning, GhostAgent_Random
import agents
import json
import matplotlib.pyplot as plt

def train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name):
    pacman_total_reward_list = []
    ghost_1_total_reward_list = []
    ghost_2_total_reward_list = []
    step_each_episode = []

    for i in range(episodes):
        print('Episode {}'.format(i+1))

        # Reset env
        new_env = env.deep_copy_new()
        pacman.init_episode(new_env)
        ghost_1.init_episode(new_env)
        ghost_2.init_episode(new_env)

        step = 0
        step_max = 8000
        while True:
            step += 1
            if step % 10000 == 1:
                print('\t step {}'.format(step))
            if step == step_max:
                info = 'Too much steps, episode interrupted'
                break

            # Train pacman
            reward, done, info = pacman.step_q_learning()
            if done:
                break

            # # Train ghost 1
            # reward, done, info = ghost_1.step_random()
            # if done:
            #     break

            # # Train ghost 2
            # reward, done, info = ghost_2.step_random()
            # if done:
            #     break

        print('\t{}\n\ttotal reward = {}'.format(info, pacman.total_reward))

        # if step == step_max:
        #     pacman_total_reward_list.append(0)
        #     ghost_1_total_reward_list.append(0)
        #     ghost_2_total_reward_list.append(0)
        #     step_each_episode.append(step)
        # else:
        if True:
            pacman_total_reward_list.append(pacman.total_reward)
            ghost_1_total_reward_list.append(ghost_1.total_reward)
            ghost_2_total_reward_list.append(ghost_2.total_reward)
            step_each_episode.append(step)
        
        save_vars = [pacman_total_reward_list,
                    ghost_1_total_reward_list, 
                    ghost_2_total_reward_list,
                    step_each_episode]

        if i % 200 == 1:
            pacman.save_q_table(version_name)
            save_total_rewards(save_vars, version_name)
            plot_total_rewards(save_vars, version_name)

    return save_vars

def plot_total_rewards(save_vars, version_name):
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(list(range(len(save_vars[0])))[::2], save_vars[0][::2])
    axs[1].plot(list(range(len(save_vars[3])))[::2], save_vars[3][::2])
    
    plt.savefig(version_name+'.png')
    plt.close(fig)


    # fig, axs = plt.subplots(4)
    # fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(list(range(len(save_vars[0]))), save_vars[0])
    # axs[1].plot(list(range(len(save_vars[1]))), save_vars[1])
    # axs[2].plot(list(range(len(save_vars[2]))), save_vars[2])
    # axs[3].plot(list(range(len(save_vars[2]))), save_vars[3])
    
    # plt.savefig(version_name+'.png')
    # plt.close(fig)


def save_total_rewards(save_vars, version_name):
    var = dict()
    var['pacman_total_reward_list'] = save_vars[0]
    var['ghost_1_total_reward_list'] = save_vars[1]
    var['ghost_2_total_reward_list'] = save_vars[2]
    var['step_each_episode'] = save_vars[3]

    with open(version_name+'.json', 'w') as outfile:
        json.dump(var, outfile)


if __name__ == '__main__':
    # layout_path = 'layouts/smallClassic.lay'
    # layout_path = 'layouts/super_small.lay'
    layout_path = 'layouts/ultra_small.lay'



    env = layout.load_env(layout_path)

    pacman = agents.PacmanAgent_QLearning(env)
    # pacman = agents.PacmanAgent_Cheating_QLearning(env)
    ghost_1 = agents.GhostAgent_Random(env, 0)
    ghost_2 = agents.GhostAgent_Random(env, 1)

    episodes = 5000

    version_name = 'output/7_normal_pacman_no_ghost_ultrasmall'

    train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name)

    print('hello_world')