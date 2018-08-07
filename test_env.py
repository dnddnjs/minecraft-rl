import numpy as np
from minecraft_env import env
from utils.utils import drawMobs


env = env.MinecraftEnv()
env.init(
    allowContinuousMovement=["move", "turn"],
    continuous_discrete=False,
    videoResolution=[800, 600]
    )

done = False
for i in range(1):
    score = 0
    count = 0
    env.reset()
    while True:
        count += 1
        env.render(mode="rgb_array")
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        num_states = info['number_of_observations_since_last_state']
        num_rewards = info['number_of_rewards_since_last_state']
        observation = info['observation']
        print(num_states, num_rewards)

        if "entities" in observation:
            entities = observation["entities"]
            map = drawMobs(entities)
            map = np.array(map)
            print(map.shape)

        score += reward
        obs = np.reshape(obs, (600, 800, 3))

        if done:
            print(str(i) + 'th episode score is ' + str(score))
            break

env.close()