#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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

        self.fc1 = nn.Linear(4, 10) # this is a 2 layer network, 10-middle layer, like a Q-table
        nn.init.normal_(self.fc1.weight, 0, 0.3)
        nn.init.constant_(self.fc1.bias, 0.1)

        self.fc2 = nn.Linear(10, 2)
        nn.init.normal_(self.fc2.weight, 0, 0.3)
        nn.init.constant_(self.fc2.bias, 0.1)

        # END STUDENT SOLUTION


    # def forward(self, state): # ori
    #     return(self.actor(state), self.critic(state))

    def forward_for_pg(self, x):
        # convert numpy array to pytorch tensor
        out = torch.from_numpy(x).float()

        out = self.fc1(out)
        out = F.sigmoid(out)

        out = self.fc2(out)
        scores = out
        out = F.softmax(out, dim=0)

        return out, scores


    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass


    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        probs, scores = self.forward_for_pg(states)
        log_prob = self.loss_func(scores, torch.tensor(actions))

        # accumulating reward
        acc_reward = []
        for i in range(len(rewards)):
            acc_r = 0
            for j in range(i, len(rewards)):
                acc_r += self.r_decay ** (j - i) * rewards[j]
            acc_reward.append(acc_r)
        acc_reward = torch.tensor(acc_reward)
        acc_reward -= acc_reward.mean()
        acc_reward /= acc_reward.std()  # todo check

        log_reward = log_prob * acc_reward
        loss = log_reward.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # END STUDENT SOLUTION
        pass

    def get_action_for_pg(self, probs):
        action = np.random.choice(a=2, p=probs.detach().numpy())
        return action

    def policy(self, env, ob):
        env.reset()

        is_done = False
        states = []
        rewards = []
        action_took = []

        while (is_done != True):
            # using current policy network to compute action
            action_probs, _ = self.forward_for_pg(ob)
            # print(action_probs)
            cur_action = self.get_action_for_pg(action_probs)
            # take the action & record
            res = env.step(int(cur_action))
            # print(res)
            state, reward, done, info, _ = res  # dn
            # add an additional dim for later concatenate
            # print(state) # [ 0.04109825  0.1922446  -0.02631517 -0.33133468]
            state_store = np.expand_dims(state, axis=0)  #?
            # print(state_store) # [[ 0.04109825  0.1922446  -0.02631517 -0.33133468]]

            env.render()
            action_took.append(cur_action)
            rewards.append(reward)
            states.append(state_store)

            # updates for next iteration
            ob = state
            is_done = done

        return np.concatenate(states, axis=0), action_took, rewards


    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        self.r_decay = 0.99
        lr = 0.02
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        init_state = env.reset()
        init_state = init_state[0]
        for i_train in range(num_episodes): # todo minibatch /  5 IID trials?
            states, actions, rewards = self.policy(env, init_state)
            # print(states)
            self.train(states, actions, rewards)  # learn
            print("Episode {} is finished, with total rewards {} ...".format(i_train, torch.tensor(rewards).sum()))
        # END STUDENT SOLUTION
        return(total_rewards)



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
    num_episodes = args.num_episodes
    train = "train"

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    pg_network = PolicyGradient(4, 2)
    env = gym.make("CartPole-v1")
    pg_network.run(env, max_steps, num_episodes, train)

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
