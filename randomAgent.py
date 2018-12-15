import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
import envs
#import gridworld_env
from gym import wrappers, logger
import numpy as np
import copy
import pandas as pd

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        print(self.action_space)

    def act(self, observation, reward, done):
        f = 2#self.action_space.sample()
        return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='gridworld-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    envx = gym.make(args.env_id)

    outdir = 'gridworld-v0/random-agent-results'

    env = wrappers.Monitor(envx, directory=outdir, force=True, video_callable=False)

    env.seed(0)

    episode_count = 1000
    reward = 0
    done = False
    envx.verbose = False
    tot_moves = []
    tot_score = []
    for num in range(0, 1):
        if num != 9:
            envx.setPlan("gridworldPlans/plan" + str(num) + ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

            agent = RandomAgent(envx)
            # np.random.seed(5)
            rsum = 0
            results = []
            moves = []
            for i in range(episode_count):
                ob = env.reset()

                '''if i % 100 == 0 and i > 0:
                    envx.verbose = True
                else:
                    envx.verbose = False'''

                if envx.verbose:
                    envx.render(1)
                j = 0

                # print(str(ob))
                while True:
                    # sprint(ob)
                    action = agent.act(ob, reward, done)
                    ob, reward, done, _ = env.step(action)
                    # print(done)
                    rsum += reward
                    j += 1
                    if envx.verbose:
                        envx.render()
                    if done:
                        results.append(rsum)
                        moves.append(j)
                        # print(str(i)+" rsum="+str(rsum)+", "+str(j)+" actions")
                        rsum = 0
                        break
            print('score: ' + str(np.mean(results)))
            print('moves: ' + str(np.mean(moves)))
            plt.hist(results)
            plt.show()
            plt.hist(moves)
            plt.show()
            tot_score.append(np.mean(results))
            tot_moves.append(np.mean(moves))
            print("done")

    env.close()
    dataframe = pd.DataFrame([tot_moves, tot_score], index=['movement', 'score'],
                             columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
    writer = pd.ExcelWriter('randomAgent'+str(episode_count) + '.xlsx')
    dataframe.to_excel(writer, 'Sheet2')
    writer.save()