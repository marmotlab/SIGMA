'''create test set and test model'''
import random
import pickle
import multiprocessing as mp
from typing import Union
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
from model import Network
import configs

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = torch.device('cpu')
torch.set_num_threads(1)

def create_test(test_env_settings, num_test_cases):

    for map_length, num_agents, density in test_env_settings:

        name = './test_set/{}length_{}agents_{}density.pth'.format(map_length, num_agents, density)
        print('-----{}length {}agents {}density-----'.format(map_length, num_agents, density))

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)

def create_test_house(test_env_settings, num_test_cases):

    for map_length, num_agents, density in test_env_settings:

        name = './house_test_set/{}length_{}agents_{}density.pth'.format(map_length, num_agents, density)
        print('-----{}length {}agents {}density-----'.format(map_length, num_agents, density))

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)


# def render_test_case(model, test_case, number):

#     network = Network()
#     network.eval()
#     network.to(device)

#     with open(test_case, 'rb') as f:
#         tests = pickle.load(f)

#     model_name = model
#     while os.path.exists('./models/{}.pth'.format(model_name)):
#         state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
#         network.load_state_dict(state_dict)
#         env = Environment()

#         case = 2
#         show = False
#         show_steps = 100

#         fail = 0
#         steps = []

#         start = time.time()
#         for i in range(test_num):
#             env.load(tests[i][0], tests[i][1], tests[i][2])
            
#             done = False
#             network.reset()

#             while not done and env.steps < configs.max_episode_length:
#                 if i == case and show and env.steps < show_steps:
#                     env.render()

#                 obs, pos = env.observe()

#                 actions, _, _, _ = network.step(torch.FloatTensor(obs).to(device), torch.FloatTensor(pos).to(device))

#                 _, _, done, _ = env.step(actions)

#             steps.append(env.steps)

#             if not np.array_equal(env.agents_pos, env.goals_pos):
#                 fail += 1
#                 if show:
#                     print(i)

#             if i == case and show:
#                 env.close(True)
        
#         f_rate = (test_num-fail)/test_num
#         mean_steps = sum(steps)/test_num
#         duration = time.time()-start

#         print('--------------{}---------------'.format(model_name))
#         print('finish: %.4f' %f_rate)
#         print('mean steps: %.2f' %mean_steps)
#         print('time spend: %.2f' %duration)

#         model_name -= configs.save_interval


def test_model(model_range):
    '''
    test model in 'models' file with model number 
    '''
    network = Network()
    network.eval()
    network.to(device)
    if configs.test_senario == 'house':
        test_set = configs.house_test_env_settings
    else:
        test_set = configs.test_env_settings

    pool = mp.Pool(mp.cpu_count())

    def write_to_file(filename, content, mode='a'):
        with open(filename, mode) as file:
            file.write(content + '\n')

    # if isinstance(model_range, int):
    #     filename = f"./models/result/{model_range}.txt"
    #     state_dict = torch.load(f'./models/{model_range}.pth', map_location=device)
    #     network.load_state_dict(state_dict)
    #     network.eval()
    #     network.share_memory()

    #     write_to_file(filename, f'----------test model {model_range}----------', 'w')

    #     for case in test_set:
    #         write_to_file(filename, f"test set: {case[0]} length {case[1]} agents {case[2]} density")
    #         if configs.test_senario == 'house':
    #             with open(f'./house_test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
    #                 tests = pickle.load(f)
    #         else:
    #             with open(f'./test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
    #                 tests = pickle.load(f)

    #         tests = [(test, network) for test in tests]
    #         ret = pool.map(test_one_case, tests)

    #         success = 0
    #         avg_step = 0
    #         for i, j in ret:
    #             success += i
    #             avg_step += j

    #         write_to_file(filename, f"success rate: {success/len(ret)*100:.2f}%")
    #         write_to_file(filename, f"average step: {avg_step/len(ret)}")
    #         write_to_file(filename, "")

    # elif isinstance(model_range, tuple):
    for model_name in model_range:
        filename = f"./models/result/{model_name}.txt"
        state_dict = torch.load(f'./models/{model_name}.pth', map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print(f'----------test model {model_name}----------')
        write_to_file(filename, f'----------test model {model_name}----------', 'w')

        for case in test_set:
            write_to_file(filename, f"test set: {case[0]} length {case[1]} agents {case[2]} density")
            if configs.test_senario == 'house':
                with open(f'./house_test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
                    tests = pickle.load(f)
            else:
                with open(f'./test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
                    tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success = 0
            avg_step = 0
            section = 0
            for i, j, k in ret:
                success += i
                avg_step += j
                section += k

            write_to_file(filename, f"success rate: {success/len(ret)*100:.2f}%")
            write_to_file(filename, f"average step: {avg_step/len(ret)}")
            write_to_file(filename, f"section map value: {section/len(ret)}")
            write_to_file(filename, "")

        write_to_file(filename, '\n')

def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    section = 0
    while not done and env.steps < configs.max_episode_length:
        actions, _, _, _, sheaf_section = network.step_test(torch.as_tensor(obs.astype(np.float32)), torch.as_tensor(pos.astype(np.float32)))
        (obs, pos), _, done, _ = env.step(actions)
        step += 1
        section += sheaf_section
    section = section / step
    return np.array_equal(env.agents_pos, env.goals_pos), step, section

def make_animation(model_name: int, test_set_name: tuple, test_case_idx: int, steps: int = 25, env_name: str = 'test_set/40length_64agents_0.3density.pth'):
    '''
    visualize running results
    model_name: model number in 'models' file
    test_set_name: (length, num_agents, density)
    test_case_idx: int, the test case index in test set
    steps: how many steps to visualize in test case
    '''
    color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green

    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load('models/{}.pth'.format(model_name), map_location=device)
    network.load_state_dict(state_dict)

    test_name = env_name

    with open(test_name, 'rb') as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

    fig = plt.figure()
            
    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
            text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)


        actions, _, _, _ = network.step(torch.from_numpy(obs.astype(np.float32)).to(device), torch.from_numpy(pos.astype(np.float32)).to(device))
        (obs, pos), _, done, _ = env.step(actions)
        # print(done)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps-env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)
                text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)


    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True, repeat_delay=1000)

    ani.save('videos/{}_{}_{}_{}.mp4'.format(model_name, *test_set_name, test_case_idx))


if __name__ == '__main__':
    
    # create maze test set
    # create_test(test_env_settings=configs.test_env_settings, num_test_cases=configs.num_test_cases)
    
    # create room like test set
    # create_test_house(test_env_settings=configs.house_test_env_settings, num_test_cases=configs.num_test_cases)

    # test model
    # test_model(['502000'])

    # visualize result
    model_name = '84000_house'
    test_env_map = 'house_test_set/60length_128agents_0.3density.pth'
    make_animation(model_name, (60, 128), 1, 512, test_env_map)