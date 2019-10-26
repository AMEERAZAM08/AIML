"""
Dice Experiment
===============
two fair dice are thrown in a series of trials and the experiment 
sums up the distibution of the situation of 
the sum of the value of 2 dice and what distibution it follows

we will be taking 100 sample of the distribution to see if the means converge

"""

from numpy.random import randint, seed
import numpy as np
import matplotlib.pyplot as plt


def experiment(seed_val, samples):
    seed(seed_val)
    exp_size = 10000
    dice1 = [randint(1, 7) for _ in range(0, exp_size)]
    dice2 = [randint(1, 7) for _ in range(0, exp_size)]
    tally = np.sum([dice1 ,dice2], axis=0)
    print(dice1)
    return tally

def draw(tally):
    plt.hist(tally, bins=[i - 0.5 for i in range(0, 15)])
    plt.title("Dice Experiment")
    plt.xticks(range(0, 15))
    plt.xlabel("sum of the dice values")
    plt.ylabel("Frequency")
    plt.show()

exp1 = experiment(42, 100)
draw(exp1)
