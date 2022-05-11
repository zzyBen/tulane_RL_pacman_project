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
    pacman_evaluation_reward_list = []
    food_left_list = []

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

        print('\t{}\n\t train total reward = {}'.format(info, pacman.total_reward))

        pacman_total_reward_list.append(pacman.total_reward)
        ghost_1_total_reward_list.append(ghost_1.total_reward)
        ghost_2_total_reward_list.append(ghost_2.total_reward)
        step_each_episode.append(step)
        food_left_list.append(len(new_env.food.asList()))

        evaluation_reward = evaluate(env, pacman, ghost_1, ghost_2)
        pacman_evaluation_reward_list.append(evaluation_reward)
        print('\tevaluation total reward = {}'.format(evaluation_reward))
        
        save_vars = [pacman_total_reward_list,
                    ghost_1_total_reward_list, 
                    ghost_2_total_reward_list,
                    step_each_episode,
                    pacman_evaluation_reward_list,
                    food_left_list]

        if i % 20 == 1:
            pacman.save_q_table(version_name)
            save_total_rewards(save_vars, version_name)
            plot_total_rewards(save_vars, version_name)

    return save_vars

def evaluate(env, pacman, ghost_1, ghost_2):
    # Reset env
    new_env = env.deep_copy_new()
    pacman.init_episode(new_env)
    ghost_1.init_episode(new_env)
    ghost_2.init_episode(new_env)

    step = 0
    step_max = 8000
    while True:
        step += 1
        if step == step_max:
            info = 'Too much steps, episode interrupted'
            break

        # Train pacman
        reward, done, info = pacman.step_optimal_policy()
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

    return pacman.total_reward


def plot_total_rewards(save_vars, version_name):
    # [::2]
    fig, axs = plt.subplots(4, figsize=(15, 15))
    axs[0].plot(list(range(len(save_vars[0]))), save_vars[0])
    axs[1].plot(list(range(len(save_vars[3]))), save_vars[3])
    axs[2].plot(list(range(len(save_vars[4]))), save_vars[4])
    axs[3].plot(list(range(len(save_vars[5]))), save_vars[5])
    
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
    var = dict()
    var['pacman_total_reward_list'] = save_vars[0]
    var['ghost_1_total_reward_list'] = save_vars[1]
    var['ghost_2_total_reward_list'] = save_vars[2]
    var['step_each_episode'] = save_vars[3]
    var['pacman_evaluation_reward_list'] = save_vars[4]
    var['food_left_list'] = save_vars[5]

    with open(version_name+'.json', 'w') as outfile:
        json.dump(var, outfile)


if __name__ == '__main__':
    # layout_path = 'layouts/small.lay'
    # layout_path = 'layouts/super_small.lay'
    # layout_path = 'layouts/ultra_small.lay'
    # layout_path = 'layouts/extreme_small.lay'
    # layout_path = 'layouts/extreme_small_2_food.lay'
    layout_path = 'layouts/extreme_small_3_food.lay'

    env = layout.load_env(layout_path)

    pacman = agents.PacmanAgent_QLearning(env)
    # pacman = agents.PacmanAgent_Cheating_QLearning(env)
    ghost_1 = agents.GhostAgent_Random(env, 0)
    ghost_2 = agents.GhostAgent_Random(env, 1)

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
    version_name = 'output/11_normal_pacman_no_ghost_extremesmall_3_food_0.2lr_seefood_1'

    train_naive(env, pacman, ghost_1, ghost_2, episodes, version_name)

    print('hello_world')