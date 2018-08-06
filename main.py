import os
import gym
import marlo
import torch
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils import get_action, pre_process
from collections import deque
from hparams import HyperParams as hp
from ppo_agent import train_model
from copy import deepcopy

if __name__=="__main__":
    env = gym.make('MinecraftCliffWalking1-v0')
    env.init(allowContinuousMovement=["move", "turn"],
             continuous_discrete=False,
             videoResolution=[800, 600])
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape
    num_actions = len(env.action_names[0])

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_actions)
    critic = Critic()

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    episodes = 0

    for iter in range(15000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        while steps < 500:
            episodes += 1
            state = env.reset()
            state = pre_process(state)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (84, 84, 4))

            score = 0
            while True:
                env.render(mode='rgb_array')
                steps += 1

                mu, std, _ = actor(torch.Tensor(history).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, info = env.step(action)
                # print(info['number_of_rewards_since_last_state'])
                reward = np.clip(reward, -1, 1)
                next_state = pre_process(next_state)
                next_state = np.reshape(next_state, (84, 84, 1))
                next_history = np.append(next_state, history[:, :, :3],
                                         axis=2)
                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([history, action, reward*0.1, mask])

                score += reward
                history = deepcopy(next_history)

                if done:
                    print(steps)
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)

        if iter % 100:
            score_avg = int(score_avg)
            directory = 'save_model/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(actor.state_dict(), 'save_model/' + str(score_avg) +
                       'actor.pt')
            torch.save(critic.state_dict(), 'save_model/' + str(score_avg) +
                       'critic.pt')