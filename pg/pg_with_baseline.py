#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from collections import namedtuple
import os

EPS = 1e-12

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.hid_dim = 16

        self.input_layer = nn.Sequential(nn.Linear(self.state_size, self.hid_dim), nn.ReLU())
        self.p_layer1 = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU())
        self.p_layer2 = nn.Linear(self.hid_dim, self.action_size)

        self.v_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU())
             for _ in range(2)])
        self.v_output_layer = nn.Linear(self.hid_dim, 1)

        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.to(device)

        # END STUDENT SOLUTION


    # def forward(self, state): # ori
    #     return(self.actor(state), self.critic(state))

    def forward(self, x):

        x = self.input_layer(x)
        out = self.p_layer1(x)
        out = self.p_layer2(out)
        action_prob = F.softmax(out, dim=-1)

        for layer in self.v_layers:
            value = layer(x)
        state_value = self.v_output_layer(value)

        return action_prob, state_value


    def get_action(self, state, stochastic=True):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        state = torch.tensor(state).float().to(device)
        # print("2:", state)
        action_prob, state_value = self.forward(state)
        dist = Categorical(action_prob)  # convert to a distribution todo
        action = dist.sample()  # choose action from the distribution

        self.saved_actions.append(SavedAction(dist.log_prob(action), state_value))  # save to action buffer

        return action.item()
        # END STUDENT SOLUTION

    def saved_rewards(self, reward):
        self.rewards.append(reward)

    def calculate_loss(self, gamma=0.999):
        saved_actions = self.saved_actions  # list of actions
        rewards = self.rewards  # list of rewards
        policy_losses = []
        state_value_list = []
        returns = []
        adv_list = []

        for t in range(len(rewards) - 1, -1, -1):  # calculate disounted returns in each time step
            disc_returns = (returns[0] if len(returns) > 0 else 0)
            G_t = gamma * disc_returns + rewards[t]
            returns.insert(0, G_t)  # insert in the beginning of the list
            state_value = saved_actions[t][1]
            state_value_list.append(state_value)
            adv_list.insert(0, G_t - state_value)

        adv_list = torch.tensor(adv_list)
        adv_list = (adv_list - adv_list.mean()) / (adv_list.std() + EPS)  # for stability

        for step in range(len(saved_actions)):
            log_prob = saved_actions[step][0]
            adv = adv_list[step]
            policy_losses.append(adv * log_prob)

        value_loss = F.mse_loss(torch.tensor(state_value_list), torch.tensor(returns))
        policy_loss = torch.stack(policy_losses, dim=0).sum()
        loss = -policy_loss + value_loss

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]



    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        r_decay = 0.99
        # model = PolicyGradient(4, 2).to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=r_decay)  # ?


        ewma_reward = 0  # EWMA reward for tracking the learning progress

        for episode in range(num_episodes):
            # reset environment and episode reward
            state = env.reset()[0]

            ep_reward = 0
            t = 0

            steps = 9999
            for t in range(steps):
                action = self.get_action(state=state)
                # env_res = env.step(action)
                # print(env_res)
                state, reward, done, _, _ = env.step(action)
                self.saved_rewards(reward)
                ep_reward += reward
                if done: break

            loss = self.calculate_loss(gamma=r_decay)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.clear_memory()

            # update EWMA reward and log the results
            ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}\tlength: {t + 1}\treward: {ep_reward}\t ewma reward: {ewma_reward}")

            if ewma_reward > env.spec.reward_threshold or episode == num_episodes - 1:
                if not os.path.isdir("./models"):
                    os.mkdir("./models")
                torch.save(self.state_dict(), f"./models/REINFORCE_WITH_BASELINE_baseline.pth")
                break


    def test(self, env, model_name):
        # model = PolicyGradient(4, 2).to(device)
        self.load_state_dict(torch.load(f"./models/{model_name}"))

        max_episode_len = 10000

        state = env.reset()[0]
        running_reward = 0
        for t in range(max_episode_len + 1):
            action = self.get_action(state)
            state, reward, done, info, _ = env.step(action)
            running_reward += reward
            if done:
                break
        print(f"Testing: Reward: {running_reward}")
        env.close()
        pass



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
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
    max_steps = args.max_steps

    train = "train"

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    pg_network = PolicyGradient(4, 2)
    env = gym.make("CartPole-v1")
    num_episodes = args.num_episodes
    pg_network.run(env, max_steps, num_episodes, train)

    pg_network.test(env, "REINFORCE_WITH_BASELINE_baseline.pth")

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
