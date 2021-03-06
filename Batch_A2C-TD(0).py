import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import envs
from gym import wrappers, logger
import numpy as np
import copy
import Box2D 
import torch
from torch import nn
from torch.autograd import Variable
from models_A2C import Pi,V
import matplotlib.pyplot as plt
import pandas as pd


class ActorCritic(object):
    """The world's simplest agent!"""
    def __init__(self, env, inSize, outSize,layers,gamma, lrPi, lrV ):
        self.inSize = inSize
        self.outSize = outSize
        self.pi = Pi(inSize, outSize, layers)
        self.V = V(inSize, outSize, layers)
        self.gamma = gamma
        self.env = env
        self.loss_V = nn.SmoothL1Loss()
        self.loss_Pi = nn.CrossEntropyLoss()
        self.optim_V = torch.optim.Adam(self.V.parameters(), lr = lrV)
        self.optim_Pi = torch.optim.Adam(self.pi.parameters(), lr = lrPi)
        self.memory_counter = 0  
        
        self.MEMORY_CAPACITY = 3000
# for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, inSize * 2 + 2))
        
    def act(self, observation):
        
        #Trouve l'action
        self.s = torch.FloatTensor(observation)
        self.prob_a = self.pi.forward(self.s)

        self.a = torch.distributions.Categorical(self.prob_a).sample()
        return self.a.tolist()

    def store_transition(self, s, a, r, s_):
        '''
        enregister la transaction dans la matrice memory
        '''
#print("########################save transaction")
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.MEMORY_CAPACITY
        #print(transition)
        self.memory[index, :] = transition
        #print("memory ", self.memory[index, :])
        
        self.memory_counter += 1
    def learn(self, ob, reward, done):
        '''
        mise a jour de V et Pi
        '''
        
        b_memory = self.memory[ :self.memory_counter, :]
        

        self.memory_counter = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, inSize * 2 + 2))

        b_s = torch.FloatTensor(b_memory[:, :self.inSize])
        b_a = torch.LongTensor(b_memory[:, self.inSize:self.inSize+1].astype(int))
        #y_onehot = torch.FloatTensor(len(b_a), self.outSize)
        #y_onehot.zero_()
        #b_a_onehot = y_onehot.scatter(1,b_a,1)
        
        
        b_r = torch.FloatTensor(b_memory[:, self.inSize+1:self.inSize+2])
        b_s1 = torch.FloatTensor(b_memory[:, -self.inSize:])

 
        #Met à jour V
        y1 = Variable(reward + self.gamma * self.V.forward(b_s1).detach(), requires_grad = False)
        
        y = self.V.forward(b_s)
        loss_V = self.loss_V(y, y1)
        self.optim_V.zero_grad()
        loss_V.backward()
  
        self.optim_V.step()
         
        #Met à jour Pi
        A = float(reward) + self.gamma* (self.V.forward(b_s1)) - (self.V.forward(b_s)) 
        A = A.detach()
      
        loss_Pi = self.loss_Pi(torch.log(self.pi.forward(b_s)) * A, b_a.view(-1))
        print('***********', loss_Pi)
        #loss_Pi = loss * (A)
               
        self.optim_Pi.zero_grad()
        loss_Pi.backward()
        self.optim_Pi.step()
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('mode', nargs='?', default=1, help='Select the environment to run')
    parser.add_argument('--jeu', type=int, default=2
                        , metavar='N')
    args = parser.parse_args()
    mode=args.jeu

    if mode == 0:
        args.env_id='gridworld-v0'
        

        # You can set the level to logger.DEBUG or logger.WARN if you
        # want to change the amount of output.
        logger.set_level(logger.INFO)

        envx = gym.make(args.env_id)



        outdir = 'gridworld-v0/random-agent-results'

        env = wrappers.Monitor(envx, directory=outdir, force=True, video_callable=False)

        env.seed(0)


        episode_count = 1000000
        reward = 0
        done = False
        envx.verbose = True

        envx.setPlan("gridworldPlans/plan0.txt",{0:-0.001,3:1,4:1,5:-1,6:-1})

        inSize= env.observation_space.shape[0]
        outSize= env.action_space.n
        layers= [30, 30]
        gamma = 0.99
        lrPi = 0.001
        lrV = 0.001
        episode_count = 1000000
        reward = 0
        done = False
        envx.verbose = True
        agent = ActorCritic(env, inSize, outSize,layers,gamma, lrPi, lrV)
        #np.random.seed(5)
        rsum=0

        for i in range(episode_count):
            ob = env.reset()

            if i % 100 == 0 and i > 0:
                envx.verbose = True
            else:
                envx.verbose = False

            if envx.verbose:
                envx.render(1)
            j = 0
            #print(str(ob))
            while True:

                action = agent.act(ob, reward, done)
                ob, reward, done, _ = env.step(action)
                
                agent.store_transition(s, action, reward, s1)                

                rsum+=reward
                j += 1
                if envx.verbose:
                    envx.render()
                if done:
                    #
                    print(str(i)+" rsum="+str(rsum)+", "+str(j)+" actions")
                    agent.learn(ob, reward, done)

                    rsum=0
                    
                    break

        print("done")
        env.close()
    if mode==1:
        args.env_id = 'CartPole-v1'
        

        # You can set the level to logger.DEBUG or logger.WARN if you
        # want to change the amount of output.
        logger.set_level(logger.INFO)

        envx = gym.make(args.env_id)
        # print(str(envx))

        # You provide the directory to write to (can be an existing
        # directory, including one with existing data -- all monitor files
        # will be namespaced). You can also dump to a tempdir if you'd
        # like: tempfile.mkdtemp().
        outdir = 'cartpole-v0/random-agent-results'
        env = envx
        env = wrappers.Monitor(envx, directory=outdir, force=True, video_callable=False)

        env.seed(0)
        inSize= env.observation_space.shape[0]
        outSize= env.action_space.n
        layers= [200]
        gamma = 0.99
        lrPi = 0.001
        lrV = 0.001
        episode_count = 30000
        reward = 0
        done = False
        envx.verbose = True
        agent = ActorCritic(env, inSize, outSize,layers,gamma, lrPi, lrV)
        np.random.seed(5)
        rsum = 0

        for i in range(episode_count):
            ob = env.reset()
            if i % 1 == 0 and i >= 0:
                envx.verbose = True
                # agent.restart=0.0
            else:
                envx.verbose = False
                # agent.restart = restart
                # ob=agent.restart()

            #if i % 1 == 0:
            #    torch.save(agent, os.path.join(outdir, "mod_last"))

            j = 0
            # agent.explo=explo
            if envx.verbose:
                env.render(1)
            # print(str(ob))
            while True:
                if envx.verbose:
                    env.render(1)
                action = agent.act(ob)
                #print("action "+str(action))
                j += 1
                
                ob, reward, done, _ = env.step(action)
                agent.learn(ob, reward, done)
                # if done:
                #    prs("rsum before",rsum)
                #    prs("reward ",reward)
                rsum += reward
                # if done:
                #    prs("rsum after",rsum)
                # if not reward == 0:
                #    print("rsum="+str(rsum))
                if done:

                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    rsum = 0
                    break

                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        # Close the env and write monitor result info to disk
        print("done")
        env.close()

    if mode==2:
        args.env_id = 'LunarLander-v2'
        
        # You can set the level to logger.DEBUG or logger.WARN if you
        # want to change the amount of output.
        logger.set_level(logger.INFO)

        envx = gym.make(args.env_id)
        # print(str(envx))

        # You provide the directory to write to (can be an existing
        # directory, including one with existing data -- all monitor files
        # will be namespaced). You can also dump to a tempdir if you'd
        # like: tempfile.mkdtemp().
        outdir = 'LunarLander-v2/results'
        env = envx
        env = wrappers.Monitor(envx, directory=outdir, force=True, video_callable=False)

        env.seed(0)


        
        inSize= env.observation_space.shape[0]
        outSize= env.action_space.n
        layers= [30, 30]
        catarget = 1
        gamma = 0.99
        lrPi = 0.001
        lrV = 0.001
        episode_count = 1000
        reward = 0
        done = False
        envx.verbose = True
        agent = ActorCritic(env, inSize, outSize,layers,gamma, lrPi, lrV)
        env._max_episode_steps = 200

        np.random.seed(5)
        rsum = 0
        rsums = []
        moves = []
        
        lnum = []
        results = []
        
        
        for i in range(episode_count):
            ob = env.reset()
            j = 0

            if i % 1 == 0 and i >= 0:
                envx.verbose = True
                # agent.restart=0.0
            else:
                
                envx.verbose = False
                # agent.restart = restart
            # ob=agent.restart()

            #if i % 1 == 0:
            #    torch.save(agent, os.path.join(outdir, "mod_last"))

            # agent.explo=explo
            if envx.verbose:
                envx.render()
            # print(str(ob))
            while True:
                if envx.verbose:
                    envx.render()
                action = agent.act(ob)
                #print("action "+str(action))
                j += 1
                
                ob_, reward, done, _ = env.step(action)
                agent.store_transition(ob, action, reward, ob_)
                # if done:
                #    prs("rsum before",rsum)
                #    prs("reward ",reward)
                rsum += reward

                # if done:
                #    prs("rsum after",rsum)
                # if not reward == 0:
                #    print("rsum="+str(rsum))
                if done:


                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    agent.learn(ob, reward, done)

                    rsums.append(rsum)
                    results.append(rsum)
                    moves.append(j)
                    lnum.append(i)
                    rsum = 0
                    break

        print("done")

        dataframe = pd.DataFrame([results, moves], index=['movement', 'score'],
                                 columns=lnum)
        #writer = pd.DataFrame.to_csv('OnlineA2CCCCCCCCCCCCCC.xlsx')
                                     
        dataframe.to_csv('OnlineA2CTD0.csv', sep='\t')
        #dataframe.to_excel(writer, 'Sheet2')
        #
        #writer.save()
