# TME 3 Juliette ORTHOLAND - Doria HAMMADI

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
from torch.autograd import Variable
from modelDQN import *

#steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
loss_list = []
class DQN(object):
    
    def __init__(self, env, MEMORY_CAPACITY, N_STATES, N_ACTIONS, GAMMA, LR , BATCH_SIZE):
        self.eval_net, self.target_net = NN(N_STATES, N_ACTIONS, [10, 10]), NN(N_STATES, N_ACTIONS, [100, 100])
        
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        
        
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.GAMMA = GAMMA
        self.LR = LR       
        self.BATCH_SIZE = BATCH_SIZE
        self.steps_learn = 0
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.SmoothL1Loss()
        
    
    def act(self, x, steps_done):
        '''
        return l'action a jouer 
        '''
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        eps_threshold = EPS_START
            
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)

        i = np.random.uniform()

        if i > eps_threshold:   
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]   
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        '''
        enregister la transaction dans matrice memory
        '''
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.MEMORY_CAPACITY
        #print(transition)
        self.memory[index, :] = transition
        #print("memory ", self.memory[index, :])
        
        self.memory_counter += 1

    def learn(self):
        '''
        choisir un batch aleatoire
            et apprednre sur ce batch
        '''
        if(self.memory_counter < self.MEMORY_CAPACITY):
            return
        self.steps_learn += 1   
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        mean_reward = b_r.mean()
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a) 
        q_ = self.eval_net(b_s)# shape (batch, 1)
        q_next = self.target_net(b_s_).detach()    
                
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        loss_list.append(loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if (self.steps_learn % 100 == 0):
            q_next = q_eval
        print("epoch: %d loss: %.3f reward_mean %.1f" %(i , loss, mean_reward)) 
        for param in self.eval_net.parameters():
                param.grad.data.clamp_(-1, 1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('mode', nargs='?', default=1, help='Select the environment to run')
    parser.add_argument('--jeu', type=int, default=1, metavar='N')
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
        BATCH_SIZE = 1
        
        LR = 0.01                   
        EPSILON = 0.9999              
        GAMMA = 0.5                 
        TARGET_REPLACE_ITER = 100   
        MEMORY_CAPACITY = 2000
        N_ACTIONS = env.action_space.n
        
        N_STATES = env.observation_space.shape[0]
        episode_count = 1000000
        reward = 0
        done = False
        envx.verbose = True

        envx.setPlan("gridworldPlans/plan0.txt",{0:-0.001,3:1,4:1,5:-1,6:-1})

        agent = DQN(env, MEMORY_CAPACITY, N_STATES, N_ACTIONS, GAMMA, LR, BATCH_SIZE)
        #np.random.seed(5)
        rsum=0

        for i in range(episode_count):
            s = env.reset()

            if i % 100 == 0 and i > 0:
                envx.verbose = True
            else:
                envx.verbose = False

            if envx.verbose:
                envx.render(1)
            j = 0
            #print(str(ob))
            while True:

                action = agent.act(s , i)
                s1, reward, done, _ = env.step(action)
                agent.store_transition(s, action, reward, s1)                
                s = s1                
                rsum+=reward
                j += 1
                agent.learn()
                if envx.verbose:
                    envx.render()
                    
                if done:
                    print(str(i)+" rsum="+str(rsum)+", "+str(j)+" actions")
                    rsum=0
                    break

        print("done")
        env.close()
        
    if mode==1:
        
        print("je suis dans cartePole")
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
        BATCH_SIZE = 1
        LR = 0.01                   
        EPSILON = 0.9999              
        GAMMA = 0.9                 
        TARGET_REPLACE_ITER = 100   
        MEMORY_CAPACITY = 2000
        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]

        episode_count = 1000000
        reward = 0
        done = False
        envx.verbose = True

        agent = DQN(env, MEMORY_CAPACITY, N_STATES, N_ACTIONS, GAMMA, LR, BATCH_SIZE)
        agent.learn()
        print("end learn")
        np.random.seed(5)
        rsum = 0

        for i in range(episode_count):
            s = env.reset()
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
                action = agent.act(s, i )
                j += 1
                s1, reward, done, _ = env.step(action)
                agent.store_transition(s, action, reward, s1)
                
                s = s1

                # if done:
                #    prs("rsum before",rsum)
                #    prs("reward ",reward)
                rsum += reward
                # if done:
                #    prs("rsum after",rsum)
                # if not reward == 0:
                #    print("rsum="+str(rsum))
                agent.learn()

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
        
        BATCH_SIZE = 32
        LR = 0.01                   
        EPSILON = 0.5
        
        GAMMA = 0.9                 
        TARGET_REPLACE_ITER = 100   
        MEMORY_CAPACITY = 2000
        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]
        
        episode_count = 10000
        
        reward = 0
        
        done = False
        envx.verbose = True

        env._max_episode_steps = 200
        agent = DQN(env, MEMORY_CAPACITY, N_STATES, N_ACTIONS, GAMMA, LR, BATCH_SIZE)
        agent.learn()

        np.random.seed(5)
        rsum = 0

        for i in range(episode_count):
            s = env.reset()
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
                envx.render()
            # print(str(ob))
            while True:
                if envx.verbose:
                    pass
                    #envx.render()
                action = agent.act(s,i)
                j += 1
                s1, reward, done, _ = env.step(action)
                agent.store_transition(s, action, reward, s1)

                # if done:
                #    prs("rsum before",rsum)
                #    prs("reward ",reward)
                rsum += reward
                # if done:
                #    prs("rsum after",rsum)
                # if not reward == 0:
                #    print("rsum="+str(rsum))
                agent.learn()
                envx.render()
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
        
        plt.figure(figsize=(10,7))
        plt.plot(loss_list, c="red")
        plt
        
        plt.show()
