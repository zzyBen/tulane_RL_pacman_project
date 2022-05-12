# import imp
import layout
# from agents import PacmanAgent_QLearning, GhostAgent_Random
import agents
import json
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
# from stable_baselines3.common.env_checker import check_env

def train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name):
    pacman_total_reward_list = []
    ghost_1_total_reward_list = []
    ghost_2_total_reward_list = []
    step_each_episode = []
    pacman_evaluation_reward_list = []
    food_left_list = []

    env.set_train_who('pacman')

    for i in range(episodes):
        print('Episode {}'.format(i+1))

        # Reset env
        env.reset()

        total_reward = 0

        step = 0
        step_max = 8000
        while True:
            step += 1
            if step == step_max:
                info = 'Too much steps, episode interrupted'
                break

            # Train pacman
            next_state, reward, done, info = pacman.step_q_learning()
            total_reward += reward
            if done:
                break


        print('\t{}\n\t train total reward = {}'.format(info, total_reward))

        pacman_total_reward_list.append(total_reward)
        step_each_episode.append(step)
        food_left_list.append(len(env.food.asList()))

        evaluation_reward = evaluate(env, pacman, ghost_1, ghost_2)
        pacman_evaluation_reward_list.append(evaluation_reward)
        
        save_vars = dict()
        save_vars['pacman_total_reward_list'] = pacman_total_reward_list
        save_vars['step_each_episode'] = step_each_episode
        save_vars['pacman_evaluation_reward_list'] = pacman_evaluation_reward_list
        save_vars['food_left_list'] = food_left_list

        if i % 500 == 1:
            # pacman.save_q_table(version_name)
            save_total_rewards(save_vars, version_name)
            plot_total_rewards(save_vars, version_name)

    return save_vars

def evaluate(env, pacman, ghost_1, ghost_2):
    # Reset env
    env.reset()
    total_reward = 0

    step = 0
    step_max = 8000
    while True:
        step += 1
        if step == step_max:
            info = 'Too much steps, episode interrupted'
            break

        # Predict pacman
        action = pacman.predict(env.get_pacman_state(), deterministic=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print('\t{}\n\tevaluation total reward = {}'.format(info, total_reward))

    return total_reward


def plot_total_rewards(save_vars, version_name):
    # [::2]

        # save_vars['pacman_total_reward_list'] = pacman_total_reward_list
        # save_vars['step_each_episode'] = step_each_episode
        # save_vars['pacman_evaluation_reward_list'] = pacman_evaluation_reward_list
        # save_vars['food_left_list'] = food_left_list

    interval = 10

    fig, axs = plt.subplots(4, figsize=(15, 15))
    axs[0].plot(list(range(len(save_vars['pacman_total_reward_list'])))[::interval], save_vars['pacman_total_reward_list'][::interval])
    axs[1].plot(list(range(len(save_vars['step_each_episode'])))[::interval], save_vars['step_each_episode'][::interval])
    axs[2].plot(list(range(len(save_vars['pacman_evaluation_reward_list'])))[::interval], save_vars['pacman_evaluation_reward_list'][::interval])
    axs[3].plot(list(range(len(save_vars['food_left_list'])))[::interval], save_vars['food_left_list'][::interval])
    
    axs[0].set_title('Pacman train total reward')
    axs[1].set_title('Number of steps')
    axs[2].set_title('Pacman evaluation total reward')
    axs[3].set_title('Food left')
    plt.savefig(version_name+'.png')
    plt.close(fig)


    # fig, axs = plt.subplots(5)
    # fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(list(range(len(save_vars[0]))), save_vars[0])
    # axs[1].plot(list(range(len(save_vars[1]))), save_vars[1])
    # axs[2].plot(list(range(len(save_vars[2]))), save_vars[2])
    # axs[3].plot(list(range(len(save_vars[3]))), save_vars[3])
    # axs[4].plot(list(range(len(save_vars[4]))), save_vars[4])

    # axs[0].set_title('Pacman train total reward')
    # axs[1].set_title('Pacman train total reward')
    # axs[2].set_title('Pacman train total reward')
    # axs[3].set_title('Number of steps')
    # axs[4].set_title('Pacman evaluation total reward')
    # plt.savefig(version_name+'.png')
    # plt.close(fig)


def save_total_rewards(save_vars, version_name):

    with open(version_name+'.json', 'w') as outfile:
        json.dump(save_vars, outfile)


if __name__ == '__main__':
    layout_path = 'layouts/small.lay'
    # layout_path = 'layouts/super_small.lay'
    # layout_path = 'layouts/ultra_small.lay'
    # layout_path = 'layouts/ultra_small_v2.lay'
    # layout_path = 'layouts/extreme_small.lay'
    # layout_path = 'layouts/extreme_small_2_food.lay'
    # layout_path = 'layouts/extreme_small_3_food.lay'

    env = layout.load_env(layout_path)

    # print(check_env(env))

    pacman = agents.PacmanAgent_QLearning(env)
    # pacman = DQN("MlpPolicy", env, verbose=1)
    # pacman = agents.PacmanAgent_Cheating_QLearning(env)
    ghost_1 = agents.GhostAgent_Random(env, 0)
    ghost_2 = agents.GhostAgent_Random(env, 1)

    env.set_pacman_ghost(pacman, ghost_1, ghost_2)

    episodes = 50000

    # version_name = 'output/1_normal_pacman_no_ghost_supersmall_evaluation'
    # version_name = 'output/1_normal_pacman_no_ghost_ultrasmall_evaluation'
    # version_name = 'output/2_normal_pacman_no_ghost_ultrasmall_0.2lr'
    # version_name = 'output/2_normal_pacman_no_ghost_ultrasmall_0.2lr_newstate'

    # version_name = 'output/4_normal_pacman_no_ghost_extremesmall_0.2lr'
    # version_name = 'output/5_normal_pacman_no_ghost_extremesmall_0.2lr_notseefood'

    # version_name = 'output/6_normal_pacman_no_ghost_extremesmall_2_food_0.2lr'
    # version_name = 'output/7_normal_pacman_no_ghost_extremesmall_2_food_0.2lr_notseefood'
    # version_name = 'output/8_normal_pacman_no_ghost_extremesmall_2_food_0.2lr_seefood_1'
    # version_name = 'output/9_normal_pacman_no_ghost_extremesmall_3_food_0.2lr_seefood_3'
    # version_name = 'output/10_normal_pacman_no_ghost_extremesmall_3_food_0.2lr_seefood_2'
    # version_name = 'output/11_normal_pacman_no_ghost_extremesmall_3_food_0.2lr_seefood_1'
    # version_name = 'output/1_new_pacman_no_ghost_extremesmall_3_food_0.2lr'
    # version_name = 'output/2_new_pacman_no_ghost_small_0.2lr'
    # version_name = 'output/3_new_pacman_1_random_ghost_small_0.2lr'
    # version_name = 'output/4_new_pacman_2_random_ghosts_small_0.2lr'
    # version_name = 'output/5_new_pacman_2_random_ghosts_small_0.2lr'
    # version_name = 'output/6_new_pacman_2_random_ghosts_ultrasmall_0.2lr'
    # version_name = 'output/7_new_pacman_2_random_ghosts_ultrasmall_v2_0.2lr'

    # version_name = 'output/1_new_pacman_2_random_ghosts_extremesmall_3_food_v2_0.2lr'
    version_name = 'output/2_new_pacman_2_random_ghosts_small_v2_0.2lr'
    # version_name = 'output/3_DQN_2_random_ghosts_small_v2_0.2lr'

    train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name)

    print('hello_world')