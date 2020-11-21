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


x_range=[x for x in np.arange(-5,5,0.05)]
sig=[sigmoid(x) for x in np.arange(-5,5,0.05)]
sig_prime=[sigmoid_prime(x) for x in np.arange(-5,5,0.05)]
sig_2prime=[sigmoid_second_prime(x) for x in np.arange(-5,5,0.05)]


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.plot(x_range, sig, color='r')
ax.plot(x_range, sig_prime, color='b')
ax.plot(x_range, sig_2prime, color='y')
ax.set_title('Sigmoid with derivative')
plt.show()

