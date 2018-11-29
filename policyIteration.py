#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:33:37 2018

@author: 3673760
"""

import argparse
import sys
import matplotlib

matplotlib.use("TkAgg")
import gym
import envs
# import gridworld_env
from gym import wrappers, logger
import numpy as np
import copy


class PolicyIteration(object):
    """The world's simplest agent!"""

    def __init__(self, env, gamma, eps=0.8):

        self.env = env
        self.states, self.p = env.getMDP()
        pi_new = [0]*len(self.states)
        pi_old = [1] * len(self.states)

        while pi_new != pi_old :
            pi_old = copy.copy(pi_new)
            v_new = [0] * len(self.states)
            v_old = [1] * len(self.states)
            norm = eps +1

            while norm > eps:
                v_old = copy.copy(v_new)
                for s in self.states:
                    if s in self.p:
                        a = pi_old[self.states[s]]
                        v_new[self.states[s]] =sum ([dest[0] * (dest[2] + gamma *
                                                v_old[self.states[dest[1]]])
                                                for dest in self.p[s][a]])
                norm = np.sum([(n - o)**2 for n, o in zip(v_new, v_old)])

            for s in self.states:
                if s in self.p:
                    pi_new[self.states[s]] = np.argmax([sum ([dest[0] * (dest[2] + gamma *
                                    v_new[self.states[dest[1]]])
                                    for dest in self.p[s][a] ])
                                    for a in self.p[s] ])

        self.pi = pi_new




    def act(self, observation, reward, done):
        # returner si c une valeur finale
        state = self.states[observation.dumps()]
        '''print(state)
        print(self.pi)
        print(self.value_new)'''
        #print(self.pi[state])

        return self.pi[state]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='gridworld-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    envx = gym.make(args.env_id)

    outdir = 'gridworld-v0/value-iteration-results'

    env = wrappers.Monitor(envx, directory=outdir, force=True, video_callable=False)

    env.seed(0)

    episode_count = 12
    reward = 0
    done = False
    envx.verbose = True

    envx.setPlan("gridworldPlans/plan2.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    agent = PolicyIteration(envx, 0.5)
    # np.random.seed(5)
    rsum = 0
    results = []
    for i in range(episode_count):
        ob = env.reset()

        envx.verbose = False
        '''if i % 100 == 0 and i > 0:
            envx.verbose = True
        else:
            envx.verbose = False'''

        if envx.verbose:
            envx.render(1)
        j = 0
        # print(str(ob))
        while True:
            # print(agent.states[ob.dumps()])
            # print(agent.p[ob.dumps()])
            # print(ob)
            # sprint(ob)
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if envx.verbose:
                envx.render()
            if done:
                results.append(rsum)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                rsum = 0
                break
    print(np.mean(results))
    print("done")
    env.close()
