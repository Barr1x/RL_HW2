#! python3

import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm



class PolicyGradient(nn.Module):
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, mode='REINFORCE', n=128, gamma=0.99, device='cpu'):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.Softmax(dim=-1)
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.baseline = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

        self.actor = self.actor.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.baseline = self.baseline.to(device) #baseline
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr = lr_actor) #baseline
        
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.actor(state), self.critic(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        action_probs = self.actor(state)
        if stochastic:
            action = torch.distributions.Categorical(action_probs).sample()
        else:
            action = torch.argmax(action_probs)
        return action
        # END STUDENT SOLUTION
    


    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        action_probs = self.actor(states)
        log_probs = torch.log(torch.gather(action_probs, 1, actions.unsqueeze(1)))
        log_probs = log_probs.squeeze()

        b_t = self.baseline(states) #baseline
        b_t = torch.squeeze(b_t) #baseline
        
        #print(log_probs)
        
        returns = torch.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            returns[t] = cumulative
        
        print("log_prob", log_probs)
        print("returns", returns)
        print("b_t", b_t)
        sys.exit(1)
        #print(log_probs*returns)
        actor_loss = -torch.mean(log_probs * returns.detach() - log_probs * b_t.detach()) #baseline
        b_loss = torch.mean((returns.detach() - b_t)**2) #baseline
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.baseline_optimizer.zero_grad() #baseline
        b_loss.backward() #baseline
        self.baseline_optimizer.step() #baseline
        # END STUDENT SOLUTION
        


    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            

            for step in range(max_steps):
                action = self.get_action(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0), train)
                action = action.item()

                next_state, reward, done, _, _ = env.step(action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)

                state = next_state

                if done:
                    break

            if train:
                self.train(episode_states, episode_actions, episode_rewards)  

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
        print(f'Running new agent\n')
        for count in tqdm(range(0, 3500, 100)):
            agent.run(env, max_steps, 100, True)
            test_returns = agent.run(env, max_steps, 20, False)
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
    mode_choices = ['REINFORCE', 'REINFORCE_WITH_BASELINE', 'A2C']

    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--mode', type=str, default='REINFORCE', choices=mode_choices, help='Mode to run the agent in')
    parser.add_argument('--n', type=int, default=64, help='The n to use for n step A2C')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=3500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    agents = []
    for i in range(args.num_runs):
        agents.append(PolicyGradient(env.observation_space.shape[0], env.action_space.n, mode="REINFORCE_WITH_BASELINE", n=args.n))
    graph_agents(args.env_name, agents, env, args.max_steps, args.num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
