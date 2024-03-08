#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys


class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=memory_size)
        # END STUDENT SOLUTION
        


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        minibatch = random.sample(self.buffer, self.batch_size)
        return minibatch
        # END STUDENT SOLUTION



    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.buffer.append(transition)
        # END STUDENT SOLUTION



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, hidden_layer_size), 
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.Q_w = q_net_init()
        self.Q_w = self.Q_w.to(device)
        self.Q_w_optim = optim.Adam(self.Q_w.parameters(), lr=lr_q_net)

        self.Q_target = q_net_init()
        self.Q_target = self.Q_target.to(device)
        self.Q_target.load_state_dict(self.Q_w.state_dict())
       
        self.replay_buffer = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        q_value = self.Q_w(state)

        # BEGIN STUDENT SOLUTION
        if stochastic:
            if random.random() < self.epsilon:
                return random.choice(range(self.action_size))
            else:
                return torch.argmax(q_value).item()
        else:
            return torch.argmax(q_value).item()
        # END STUDENT SOLUTION
        pass


    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        # minibatch = self.replay_buffer.sample_batch()
        # y = torch.zeros(self.replay_buffer.batch_size, device=self.device)
        # q = torch.zeros(self.replay_buffer.batch_size, device=self.device)
        # for i in range(self.replay_buffer.batch_size):
        #     state, action, reward, next_state, done = minibatch[i]
        #     q[i] = self.Q_w(torch.tensor(state, dtype=torch.float32, device=self.device))[action]
        #     if done:
        #         y[i] = reward
        #     else:
        #         y[i] = reward + self.gamma * torch.max(self.Q_target(torch.tensor(next_state, dtype=torch.float32, device=self.device))).item()
        # loss = torch.mean((y.detach() - q)**2)
        # self.Q_w_optim.zero_grad()
        # loss.backward()
        # self.Q_w_optim.step()
        minibatch = self.replay_buffer.sample_batch()
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.Q_w(states)
        next_q_values = self.Q_target(next_states)

        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = next_q_values.max(1)[0]

        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets.detach())
        self.Q_w_optim.zero_grad()
        loss.backward()
        self.Q_w_optim.step()
        # END STUDENT SOLUTION
        


    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []
        c = 0
        # initialize replay buffer
        if (init_buffer):
            state, _ = env.reset()
            for i in range(self.burn_in):
                action = env.action_space.sample()
                next_state, reward, done, _, _ = env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    state, _ = env.reset()
            
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_rewards = []

            for step in range(max_steps):
                action = self.get_action(torch.tensor(state, dtype=torch.float32, device=self.device), train)
                
                next_state, reward, done, _, _ = env.step(action)

                episode_rewards.append(reward)

                self.replay_buffer.append((state, action, reward, next_state, done))

                state = next_state

                if train:
                    c = c + 1
                    if (c % 50 == 0):
                        self.Q_target.load_state_dict(self.Q_w.state_dict())
                    self.train()  
                
                if done:
                    break
            total_rewards.append(sum(episode_rewards))
        # END STUDENT SOLUTION
        return(total_rewards)



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    D = []
    for agent in agents:
        temp = []
        for count in tqdm(range(0, 1000, 100)):
            if (count == 0):
                print(f'Running new agent\n')
                agent.run(env, max_steps, 100, True, True)
                test_returns = agent.run(env, max_steps, 20, False, False)
                print(f'Average Return: {np.mean(test_returns)}\n')
                temp += [np.mean(test_returns)]
            else:
                agent.run(env, max_steps, 100, True, False)
                test_returns = agent.run(env, max_steps, 20, False, False)
                print(f'Average Return: {np.mean(test_returns)}\n')
                temp += [np.mean(test_returns)]
        D.append(temp)
    D = np.array(D)
    average_total_rewards = np.mean(D, axis=0)
    min_total_rewards = np.min(D, axis=0)
    max_total_rewards = np.max(D, axis=0)
    graph_every = 100
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    agents = []
    for i in range(args.num_runs):
        agents.append(DeepQNetwork(env.observation_space.shape[0], env.action_space.n))
    graph_agents(args.env_name, agents, env, args.max_steps, args.num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
