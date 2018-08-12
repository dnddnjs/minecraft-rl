import os
import torch
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import get_action, pre_process, init_map, drawMobs
from hparams import HyperParams as hp
from ppo_agent import train_model
from copy import deepcopy
from minecraft_env import env
from memory import Memory


if __name__=="__main__":
    env = env.MinecraftEnv()
    env.init(allowContinuousMovement=["move", "turn"],
             continuous_discrete=False,
             videoResolution=[800, 600])
    env.seed(500)
    torch.manual_seed(500)
    render_map = False

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
    if render_map:
        root, canvas = init_map()

    for iter in range(15000):
        actor.eval(), critic.eval()
        memory = Memory()
        steps = 0
        scores = []
        while steps < 1000:
            episodes += 1
            state = env.reset()
            state = pre_process(state)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (84, 84, 4))

            for i in range(3):
                action = env.action_space.sample()
                _, reward, done, info = env.step(action)
                # next_state = pre_process(next_state)
                # next_state = np.reshape(next_state, (84, 84, 1))
                # next_history = np.append(next_state, history[:, :, :3],
                 #                         axis=2)

            observation = info['observation']
            if observation is not None:
                if "entities" in observation:
                    entities = observation["entities"]
                    map = drawMobs(entities)
                    map = np.array(map)
                    state = pre_process(map)
                    # map = np.reshape(map, (84, 84, 1))

            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (84, 84, 4))
            # input = np.append(history, map, axis=2)

            score = 0
            prev_life = 20
            while True:
                env.render(mode='rgb_array')
                steps += 1

                mu, std, _ = actor(torch.Tensor(history).unsqueeze(0))
                action = get_action(mu, std)[0]
                _, reward, done, info = env.step(action)

                observation = info['observation']
                if observation is not None:
                    if "entities" in observation:
                        entities = observation["entities"]
                        map = drawMobs(entities)
                        map = np.array(map)
                        map = pre_process(map)
                        state = np.reshape(map, (84, 84, 1))
                    life = observation['entities'][0]['life']
                    if life < prev_life:
                        reward = reward + (life - prev_life)

                # next_state = pre_process(next_state)
                # next_state = np.reshape(next_state, (84, 84, 1))
                next_history = np.append(state, history[:, :, :3],
                                         axis=2)
                # input = np.append(next_history, map, axis=2)
                reward *= 0.1
                reward += 0.1

                if done:
                    mask = 0
                    reward = 0
                else:
                    mask = 1


                memory.push(next_history, np.array(action), reward, mask)

                score += reward
                history = deepcopy(next_history)

                if done:
                    print('steps: ', steps, ' score: ', score)
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        batch = memory.sample()
        train_model(actor, critic, batch, actor_optim, critic_optim)

        if iter % 100:
            score_avg = int(score_avg)
            directory = 'save_model/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(actor.state_dict(), 'save_model/' + str(score_avg) +
                       'actor.pt')
            torch.save(critic.state_dict(), 'save_model/' + str(score_avg) +
                       'critic.pt')