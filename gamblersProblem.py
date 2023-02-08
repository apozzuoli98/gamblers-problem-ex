import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt



""" Implementation of the Gambler's Problem in R. Sutton & A. Bartow, Reinforcement Learning: An Introduction (2nd Edition). 2020

A Gambler has the opportunity to make bets on a sequence of coin flips. 
If the coin comes up heads, they win what they staked on that flip. If tails, they lose the stake. 
Game ends when the gambler reaches $100 or $0.
On each flip, the gambler must decide what portion of their capital to stake, in integer number of dollars.

"""
class GamblersProblem:

    def __init__(self, ph):
        self.ph = ph
        self.S = np.arange(101) # All possible states (the gamblers capital)
        self.V = np.zeros(101) # values for each state
        self.pi = np.zeros(101) # 'optimal' policy
        self.val_iter()
        self.plot_value()
        self.plot_policy()

    """ Value iteration """
    def val_iter(self):
        theta = 0.01
        gamma = 1 # discount of future value
        sweep = 0
        while True:
            sweep += 1
            delta = 0
            for s in self.S:
                if s == 0 or s == 100:
                    continue
                v = self.V[s]
                A = np.array([x for x in range(0,min(s+1, 100-s+1))]) # All actions from current state (+1 since range is exclusive upper bound)
                R = s + A # All next states
                R = np.where(R == 100, 1, 0) # Reward for next state
                policy_eval = self.ph*(R + gamma*self.V[s+A]) + (1.0-self.ph)*(gamma*self.V[s-A]) # evaluate the action from current state
                self.V[s] = np.amax(policy_eval) # Set value to policy evaluation
                self.pi[s] = np.argmax(policy_eval) # set policy to action that yielded max reward
                delta = max(delta, abs(v - self.V[s]))
            if delta < theta: break


    """ Plot estimated values of each state """
    def plot_value(self):
        plt.plot(self.V)
        plt.xlim([1,99])
        xticks = np.arange(0,99, step=25)
        xticks = np.append(xticks, 99)
        plt.xticks(xticks)
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.show()

    """ Plot policy for each state """
    def plot_policy(self):
        plt.step(np.arange(0,101), self.pi)
        plt.xlim([1,99])
        xticks = np.arange(0,99, step=25)
        xticks = np.append(xticks, 99)
        plt.xticks(xticks)
        plt.xlabel('Capital')
        plt.ylabel('Final policy (Stake)')
        plt.show()

def main():
    GamblersProblem(0.4)


if __name__ == '__main__':
    main()
