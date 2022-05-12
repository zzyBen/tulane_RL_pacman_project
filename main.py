# import imp
import layout
# from agents import PacmanAgent_QLearning, GhostAgent_Random
import agents
import json
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

def train_stable_baseline(env, pacman, ghost_1, ghost_2, episodes, version_name):

    total_timesteps = 1000

    evaluation_step_list = []
    evaluation_reward_list = []
    evaluation_food_left_list = []
    total_timesteps_list = []

    for i in range(episodes):
        print('Episode {}'.format(i+1))

        # Reset env
        env.set_train_who('pacman')
        env.reset()

        print('-------------------------- Learn Start')
        pacman.learn(total_timesteps=total_timesteps, log_interval=4)
        print('-------------------------- Learn Stop')

        # pacman_total_reward_list.append(total_reward)
        # step_each_episode.append(step)
        # food_left_list.append(len(env.food.asList()))

        evaluation_result = evaluate(env, pacman, ghost_1, ghost_2, sb3=True)
        evaluation_step_list.append(evaluation_result['step'])
        evaluation_reward_list.append(evaluation_result['total_reward'])
        evaluation_food_left_list.append(evaluation_result['food_left'])

        total_timesteps_list.append((i+1))

        if i % 5 == 1:
            save_vars = dict()
            save_vars['evaluation_step_list'] = evaluation_step_list
            save_vars['evaluation_reward_list'] = evaluation_reward_list
            save_vars['evaluation_food_left_list'] = evaluation_food_left_list
            save_vars['total_timesteps_list'] = total_timesteps_list

            # pacman.save_q_table(version_name)
            save_total_rewards(save_vars, version_name)
            # plot_total_rewards(save_vars, version_name)
            plot_sb3_rewards(save_vars, version_name)

    return save_vars


def train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name):
    pacman_total_reward_list = []
    step_each_episode = []
    pacman_evaluation_reward_list = []
    food_left_list = []

    for i in range(episodes):
        print('Episode {}'.format(i+1))

        # Reset env
        env.set_train_who('pacman')
        env.reset()
        

        total_reward = 0

        step = 0
        step_max = 2000
        while True:
            step += 1
            if step == step_max:
                info = 'Too much steps, episode interrupted'
                break

            # Train pacman
            next_state, reward, done, info = pacman.step_q_learning()
            info = info['info']
            total_reward += reward
            if done:
                break


        print('\t{}\n\t train total reward = {}'.format(info, total_reward))

        pacman_total_reward_list.append(total_reward)
        step_each_episode.append(step)
        food_left_list.append(len(env.food.asList()))

        evaluation_result = evaluate(env, pacman, ghost_1, ghost_2)
        evaluation_reward = evaluation_result['total_reward']
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

def train_naive_alternative(env, pacman, ghost_1, ghost_2, episodes, version_name):
    pacman_total_reward_list = []
    step_each_episode = []
    pacman_evaluation_reward_list = []
    food_left_list = []

    for i in range(episodes):
        print('Episode {}'.format(i+1))

        # Reset env
        env.set_train_who('pacman')
        env.reset()
        

        total_reward = 0

        step = 0
        step_max = 2000
        while True:
            step += 1
            if step == step_max:
                info = 'Too much steps, episode interrupted'
                break

            # Train pacman
            next_state, reward, done, info = pacman.step_q_learning()
            info = info['info']
            total_reward += reward
            if done:
                break


        print('\t{}\n\t train total reward = {}'.format(info, total_reward))

        pacman_total_reward_list.append(total_reward)
        step_each_episode.append(step)
        food_left_list.append(len(env.food.asList()))

        evaluation_result = evaluate(env, pacman, ghost_1, ghost_2)
        evaluation_reward = evaluation_result['total_reward']
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
        
        
        if i % 100 == 1:
            train_ghost_2(env, pacman, ghost_1, ghost_2, 100)

    return save_vars

def train_ghost_2(env, pacman, ghost_1, ghost_2, episodes):
    for i in range(episodes):
        # print('Episode {}'.format(i+1))

        # Reset env
        env.set_train_who('ghost_2')
        env.reset()

        step = 0
        step_max = 2000
        while True:
            step += 1
            if step == step_max:
                break

            # Train pacman
            next_state, reward, done, info = ghost_2.step_q_learning()
            if done:
                break


def evaluate(env, pacman, ghost_1, ghost_2, sb3=False):
    # Reset env
    env.set_train_who('evaluate')
    env.reset()
    
    total_reward = 0

    step = 0
    step_max = 2000
    while True:
        step += 1
        if step == step_max:
            info = 'Too much steps, episode interrupted'
            break

        action, _states = pacman.predict(env.get_pacman_state(), deterministic=True)
        next_state, reward, done, info = env.step(action)
        info = info['info']
        total_reward += reward
        if done:
            break

    print('\t{}\n\tevaluation total reward = {}'.format(info, total_reward))

    evaluation_result = dict()
    evaluation_result['total_reward'] = total_reward
    evaluation_result['step'] = step
    evaluation_result['food_left'] = len(env.food.asList())
    
    return evaluation_result

def plot_total_rewards(save_vars, version_name):
    # [::2]

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



def plot_sb3_rewards(save_vars, version_name):
    # [::2]

        # save_vars['pacman_total_reward_list'] = pacman_total_reward_list
        # save_vars['step_each_episode'] = step_each_episode
        # save_vars['pacman_evaluation_reward_list'] = pacman_evaluation_reward_list
        # save_vars['food_left_list'] = food_left_list

    interval = 1

    fig, axs = plt.subplots(3, figsize=(15, 15))
    axs[0].plot(save_vars['total_timesteps_list'][::interval], save_vars['evaluation_reward_list'][::interval])
    axs[1].plot(save_vars['total_timesteps_list'][::interval], save_vars['evaluation_step_list'][::interval])
    axs[2].plot(save_vars['total_timesteps_list'][::interval], save_vars['evaluation_food_left_list'][::interval])
    
    axs[0].set_title('Evaluation total reward')
    axs[1].set_title('Evaluation steps')
    axs[2].set_title('Evaluation food left')
    plt.savefig(version_name+'.png')
    plt.close(fig)

def save_total_rewards(save_vars, version_name):

    with open(version_name+'.json', 'w') as outfile:
        json.dump(save_vars, outfile)


if __name__ == '__main__':
    layout_path = 'layouts/small.lay'
    # layout_path = 'layouts/super_small.lay'
    # layout_path = 'layouts/ultra_small.lay'
    # layout_path = 'layouts/ultra_small_v2.lay'
    # layout_path = 'layouts/ultra_small_v3.lay'
    # layout_path = 'layouts/extreme_small.lay'
    # layout_path = 'layouts/extreme_small_2_food.lay'
    # layout_path = 'layouts/extreme_small_3_food.lay'

    env = layout.load_env(layout_path)


    pacman = agents.PacmanAgent_QLearning(env)
    # pacman = PPO("MultiInputPolicy", env, verbose=0)
    ghost_1 = agents.GhostAgent_Random(env, 0)
    # ghost_2 = agents.GhostAgent_Random(env, 1)
    ghost_2 = agents.GhostAgent_QLearning(env, 1)

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


    # version_name = 'output/1_new_pacman_2_random_ghosts_ultra_small_v2_0.2lr'
    # version_name = 'output/2_new_pacman_2_random_ghosts_small_v2_0.2lr'
    # version_name = 'output/3_new_pacman_2_random_ghosts_small_0.2lr'
    # version_name = 'output/4_PPO_1000steps_2_random_ghosts_small_0.2lr'
    # version_name = 'output/5_PPO_1000steps_2_random_ghosts_ultra_small_v2_0.2lr'
    # version_name = 'output/6_PPO_1000steps_2_random_ghosts_ultra_small_v3_0.2lr'
    # version_name = 'output/7_Qlearning_2_random_small_0.2lr'
    # version_name = 'output/8_alternative_Qlearning_1_random_small_0.2lr'
    version_name = 'output/8_alternative_evaluate_first_Qlearning_1_random_small_0.2lr'

    env.set_train_who('pacman')
    print(check_env(env))

    # train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name)
    train_naive_alternative(env, pacman, ghost_1, ghost_2, episodes, version_name)
    # train_stable_baseline(env, pacman, ghost_1, ghost_2, episodes, version_name)

    print('hello_world')