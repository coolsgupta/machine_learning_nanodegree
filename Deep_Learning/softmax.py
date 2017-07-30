import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_list = np.exp(L)
    total = sum(exp_list)
    prob_l = []
    for x in exp_list:
        y = (x*1.0/total)
        prob_l.append(y)

    return prob_l
