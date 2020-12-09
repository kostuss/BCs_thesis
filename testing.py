import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid_second_prime(z):
    """ Second order derivative of the sigmoid function """
    return -(1/(np.exp(z)+1))+(3/((np.exp(z)+1)**2))-(2/((np.exp(z)+1)**3))



def simulation(B):
	for i in range(10):
		B +=1




