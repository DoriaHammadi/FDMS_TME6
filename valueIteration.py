#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:33:37 2018

@author: 3673760
"""

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

class ValueIteration(object):
    """The world's simplest agent!"""
    def __init__(self, env,gamma,eps = 0.001):
        
        self.env = env 
        self.states, self.p = env.getMDP()
        #sprint(p)
        value_old = np.zeros(len(self.states) )
        self.value_new = np.zeros(len(self.states) ) 
        self.pi = np.zeros(len(self.states) ) 
        while True :
            value_old = copy.copy(self.value_new)
            delta = 0
            for s in self.states :
                if s in self.p.keys(): 
                    self.value_new[self.states[s]] = max ([sum ([dest[0] * (dest[2] + gamma * value_old[self.states[dest[1]]]) 
                                                        for dest in self.p[s][a] ])
                                                  for a in self.p[s] ])
                    
                    delta = max(delta, abs(self.value_new[self.states[s]] - value_old[self.states[s]]))                
            if ( delta < eps):               
                break 
                
        #### Meilleur Politique ####
        for s in self.states :
            if s in self.p.keys():
                act_s = { a : sum ([dest[0] * (dest[2] + gamma * self.value_new[self.states[dest[1]]])
                            for dest in self.p[s][a] ])
                        for a in self.p[s]}
                self.pi[self.states[s]] = max(act_s, key=act_s.get)
                
                
    def act(self, observation, reward, done):
        #returner si c une valeur finale
        state = self.states[observation.dumps()]
        #print(state)
        #print(self.pi, self.pi[state])
        #ssprint(self.value_new)
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

    num = 0
    gamma = 0.5
    eps = 0.001

    lnum = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    lgamma = [i/10 for i in range(1,10)]
    leps = [0.001]#[0.1,0.01,0.001,0.0001]

    episode_count = 1000
    reward = 0
    done = False
    envx.verbose = False
    tot_moves = []
    tot_score = []
    for eps in leps:
        if num != 9:
            envx.setPlan("gridworldPlans/plan" + str(num) + ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

            agent = ValueIteration(envx, gamma,eps)
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
                             columns=leps)
    writer = pd.ExcelWriter('valueIteration' + str(episode_count) + '.xlsx')
    dataframe.to_excel(writer, 'Sheet2')
    writer.save()
