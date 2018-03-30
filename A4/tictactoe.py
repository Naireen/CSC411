from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import pickle
import os

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done




class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=40, output_size=9):
        super(Policy, self).__init__()
        # TODO
        self.hidden_layer_size = hidden_size
        # input is 27 dim vector, output is 9 dim vector
        self.features = nn.Sequential(
            # an affine operation: y = Wx + b
            nn.Linear(27, hidden_size),
            #nn.LogSoftmax(),
            nn.SELU(),
            nn.Linear(hidden_size, 9), # each of the nine represent where they should move
        )
        return

    def forward(self, x):
        # TODO
        x = self.features(x)
        return x


def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    #print("Unsqueeze state,", state)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    #print("Scattered state", state)
    pr = policy(Variable(state))
    #print(type(pr))
    pr = 1./(1. +torch.exp(-1*pr))
    #pr = 1./(1 + np.exp(-1*pr))
    #pr =
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob


# preallocate empty array and assign slice by chrisaycock
def shift5(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    total = len(rewards)
    rewards = np.asarray(rewards)
    total_rewards = np.zeros(total)
    dist_f = [gamma ** (i) for i in range(total)]
    for j in range(total):
        discount_factor = np.sum(dist_f * rewards)
        dist_f = shift5(dist_f, +1)
        total_rewards[j] = discount_factor
    return total_rewards


def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 5, # TODO
            Environment.STATUS_INVALID_MOVE: -40, # -20
            Environment.STATUS_WIN         : 40,
            Environment.STATUS_TIE         : 5,
            Environment.STATUS_LOSE        : 0
    }[status]


def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    #player_label = env.turn
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    ep_length = 200000
    final_rewards = np.zeros(ep_length)
    player_ids = np.zeros(ep_length)
    final_statuses = np.zeros_like(player_ids)
    path = "ttt_%d" % policy.hidden_layer_size
    #check if saved directory exists, if not, create it
    try:
        os.stat(path)
    except:
        os.mkdir(path)

    for i_episode in range(ep_length):
        saved_rewards = []
        saved_logprobs = []
        #np.random.seed(i_episode) # varying this will vary new player when env.reset is called
        #pass random initial player start
        state = env.reset()
        player_label = env.turn
        #print(env.turn)
        #print("Initial State", state  )
        done = False
        while not done:
            if player_label ==1:
                #print("Play original", player_label)
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
            else: # player id is 2, so it goes second
                #print("enf turn", env.turn)
                state , status, done = env.play_against_random_after(policy)
                if not done:
                    #print("pl", env.turn)
                    action, logprob = select_action(policy, state)
                    state, status, done = env.step(action)


            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R
        final_rewards[i_episode] = R
        player_ids[i_episode] = player_label
        final_statuses[i_episode] = get_reward(status) # last status in the game
        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == (log_interval-1):
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt_%d/policy-%d.pkl" % (policy.hidden_layer_size ,i_episode))

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    #save very last model
    torch.save(policy.state_dict(),
                       "ttt_%d/policy-%d.pkl" % (policy.hidden_layer_size ,i_episode))

    total_data = np.vstack((player_ids, final_rewards, final_statuses))
    np.savetxt("rewards_new_{}_{}_iterations.txt".format(policy.hidden_layer_size, ep_length), total_data)
    #np.savetxt("rewards_new_100.txt", final_rewards)




def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode, folder):
    """Load saved weights"""
    weights = torch.load("{}/policy-{}.pkl".format(folder, episode))
    policy.load_state_dict(weights)


if __name__ == '__main__':
    import sys
    policy = Policy(hidden_size=40)
    env = Environment()
    print("Policy", policy)

    #'''
    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))
    #'''
    def print_state():
        state = env.grid
        state = torch.from_numpy(state).long().unsqueeze(0)
        state = torch.zeros(3, 9).scatter_(0, state, 1)
        print(state)

    #run human coded steps
    #'''
    env.step(1)
    env.render()
    print_state()
    env.step(2)
    env.render()
    print_state()
    env.step(0)
    env.render()
    print_state()
    env.step(3)
    env.render()
    print_state()
    env.step(5)
    env.render()
    print_state()
    env.step(4)
    env.render()
    print_state()
    env.step(6)
    env.render()
    print_state()
    env.step(7)
    env.render()
    print_state()
    env.step(8)
    env.render()
    print_state()
    #cd
    # '''

    # run policy against random
    '''
    policy = Policy(hidden_size=40)
    episode = 49000
    load_weights(policy, episode, "ttt_40")
    print(policy)
    
    #random.seed(0)
    for i_episode in range(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        #print("Initial State", state  )
        done = False
        random.seed(0)
        while not done:
            #random.setstate(0)
            action, logprob = select_action(policy, state)
            env.render()
            state, status, done = env.play_against_random(action)
            env.render()
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
    '''
    '''
    #view initial distribution of model
    policy = Policy(hidden_size=40)
    episode = 0
    load_weights(policy, episode, "ttt_40")
    print(first_move_distr(policy, env))
    '''