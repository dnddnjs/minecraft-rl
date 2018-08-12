import numpy as np
from minecraft_env import env
from utils.utils import drawMobs
import cv2


env = env.MinecraftEnv()
env.init(
    allowContinuousMovement=["move", "turn"],
    continuous_discrete=False,
    videoResolution=[800, 600]
    )

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('record/mob-fun.mp4', fourcc, 20, (800, 600))

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
            drawMobs(entities)
        score += reward
        obs = np.reshape(obs, (600, 800, 3))
        video.write(obs)

        if done:
            cv2.destroyAllWindows()
            video.release()
            print(str(i) + 'th episode score is ' + str(score))
            break

env.close()