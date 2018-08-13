import os
import torch
import numpy as np
import torch.optim as optim
from model import QNet
from utils.utils import *
from hparams import HyperParams as hp
from dqn_agent import train_model
from copy import deepcopy
from minecraft_env import env
from memory import Memory


if __name__=="__main__":
    env = env.MinecraftEnv()
    env.init(allowDiscreteMovement=None, 
             videoResolution=[800, 600])
    env.seed(500)
    torch.manual_seed(500)
    render_map = False

    num_inputs = env.observation_space.shape
    num_actions = len(env.action_names[0])

    print('state size:', num_inputs)
    print('action size:', num_actions)

    model = QNet(num_actions)
    model.apply(weights_init)
    target_model = QNet(num_actions)
    update_target_model(model, target_model)
    model.train()
    target_model.train()

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, 
                           weight_decay=hp.l2_rate)

    memory = Memory(100000)
    if render_map:
        root, canvas = init_map()


    steps = 0
    scores = []
    epsilon = 1.0
    for episode in range(hp.num_episodes):
        state = env.reset()
        state = pre_process(state)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (84, 84, 4))

        for i in range(3):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            next_state = pre_process(next_state)
            next_state = np.reshape(next_state, (84, 84, 1))
            history = np.append(next_state, history[:, :, :3], axis=2)

        score = 0
        prev_life = 20
        while True:
            env.render(mode='rgb_array')
            steps += 1

            qvalue = model(to_tensor(history).unsqueeze(0))
            action = get_action(epsilon, qvalue, num_actions)
            next_state, reward, done, info = env.step(action)

            observation = info['observation']

            next_state = pre_process(next_state)
            next_state = np.reshape(next_state, (84, 84, 1))
            next_history = np.append(next_state, history[:, :, :3], axis=2)
            reward *= 0.1
            reward += 0.1

            if done:
                mask = 0
                reward = 0
            else:
                mask = 1


            memory.push(history, next_history, action, reward, mask)

            score += reward
            history = deepcopy(next_history)

            if steps > hp.initial_exploration:
                if epsilon < 0.1:
                    epsilon -= 0.00001
                batch = memory.sample()
                train_model(model, target_model, batch, optimizer)

            if steps % hp.update_target:
                update_target_model(model, target_model)

            if done:
                print('episode: ', episode, 'steps: ', steps, 'epsilon: ', round(epsilon, 4), 
                      ' score: ', score)
                break


        if episode % hp.save_freq:
            score = int(score)
            directory = 'save_model/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), 'save_model/' + str(score) +
                       'model.pt')