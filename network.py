"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network and Optimal Brain Damage algorithm 
for network prunning.  
Gradients are calculated using backpropagation.

Implementation based on: 
- Michael A.Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015
- Yann LeCun, John S.Denker, Sara A. Solla, "Optimal Brain Damage", Advances in Neural Information Processing Systems, 1989
"""

#### Libraries
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        arg ``sizes`` contains the number of neurons in the
        respective layers of the network. e.g sizes=[30,50,10]
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1. 
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #saliencies for each parameter (weight) used in OBD algorithm 
        self.saliencies = [np.zeros((y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def feedforward_epsilon(self, a, weights):
        """Return the output of the network with chosen weights. 
        Used for calculating symetric derivative.
        """
        for b, w in zip(self.biases, weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.
        If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def show_accuracy(self, test_data):
        """
        Print to console accuracy for the given test data
        Method uses 'evaluate' method
        """
        n_test=len(test_data)
        correct=self.evaluate(test_data)
        print("Accuracy : {} / {} = {:.2f}".format(correct,n_test, correct/n_test))


    def cost_function(self, output_activations, y):
        """Return cost for a single training example  """
        return 1/2 * np.linalg.norm(y - output_activations)**2

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


    def second_symetric_derivative(self, wt_index, training_example):
        """
        Calculate second derivative (diagonal of the Hesian) for the given parameter 
        (weight) using second symetric derivative  
        arg wt_index - tuple for weight identification (layer, to i-th neuron, from j-th neuron)
        """
        
        x = training_example[0]
        y = training_example[1]

        epsilon = np.finfo(np.float64).eps

        # positive epsilon change
        weights_epsilon = [np.copy(cop) for cop in self.weights]
        #print("Weight =:" , weights_epsilon[wt_index[0]][wt_index[1]][wt_index[2]])
        #print("Weight  :" , self.weights[wt_index[0]][wt_index[1]][wt_index[2]])

        weights_epsilon[wt_index[0]][wt_index[1]][wt_index[2]] += epsilon
        #print("Weight +:",weights_epsilon[wt_index[0]][wt_index[1]][wt_index[2]])
        #print("Weight  :" , self.weights[wt_index[0]][wt_index[1]][wt_index[2]])
        f_xh_pos = self.cost_function(self.feedforward_epsilon(x, weights_epsilon), y)
        print("Value +:" , f_xh_pos)

        # negative epsilon change
        weights_epsilon = [np.copy(cop) for cop in self.weights]
        weights_epsilon[wt_index[0]][wt_index[1]][wt_index[2]] -= epsilon
        #print("Weight -:" , weights_epsilon[wt_index[0]][wt_index[1]][wt_index[2]])
        #print("Weight  :" , self.weights[wt_index[0]][wt_index[1]][wt_index[2]])
        f_xh_neg = self.cost_function(self.feedforward_epsilon(x, weights_epsilon), y)
        print("Value -:" , f_xh_neg)

        # without epsilon change
        f_x = self.cost_function(self.feedforward(x), y)
        print("Value =:" , f_x)
        return (f_xh_pos - (2 * f_x) + f_xh_neg) / epsilon**2


    def evaluate_second_derivative(self, wt_index, training_example):

        symetric_derivative = self.second_symetric_derivative(wt_index, training_example)

        x = training_example[0]
        y = training_example[1]
        backprop_derivative = self.backpropOBD(training_example[0], training_example[1])[ wt_index[0] ][ wt_index[1] ][ wt_index[2] ]

        return (symetric_derivative , backprop_derivative)

        
        ####### OBD module ########
        """
        weighted sum:    z vectors (in code) -> a (in LeCun)
        activations:     activation (in code) -> x (in LeCun)
        """
    def OBD(self, training_data):
        
        """Calculate saliencies based on second derivatives h"""
        nabla_h = [np.zeros(w.shape) for w in self.weights]

        """ Iterate over training examples """
        for x, y in training_data:
            delta_nabla_h = self.backpropOBD(x,y)
            nabla_h = [nh+(dnh/len(training_data))
                         for nh, dnh in zip(nabla_h, delta_nabla_h)]

        self.saliencies = [(h * w**2)/2 
                            for w, h in zip(self.weights, nabla_h)]

    def backpropOBD(self, x, y):

        #second_derivaties = [np.zeros(b.shape) for b in self.biases]
        h_vector = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_second_prime(zs[-1])

        delta2_z = self.boundry_OBD_derivative(zs[-1], y)

        h_vector[-1] = np.dot(delta2_z, activations[-2].transpose()**2)

        #iterate over layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            #backpropagate second term of (7) in LeCun
            delta = np.dot(self.weights[-l+1].transpose(), delta) * \
             sigmoid_second_prime(z)
            
            #backpropagate second derivative (7) in LeCun
            delta2_z = np.dot(self.weights[-l+1].transpose()**2, delta2_z) * sp**2 + \
             delta

            #second_derivaties[-l] = delta2_z
            h_vector[-l] = np.dot(delta2_z, activations[-l-1].transpose()**2)
           
        return h_vector

    def boundry_OBD_derivative(self, weighted_sums, y):
        """
        Boundry condition at the output layer 
        (8) in LeCun 
        """
        return 2 * sigmoid_prime(weighted_sums)**2 - \
            2 * (y - sigmoid(weighted_sums)) * sigmoid_second_prime(weighted_sums)



#### Sigmoid function and its derivatives
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid_second_prime(z):
    """ Second order derivative of the sigmoid function """
    return -(1/(np.exp(z)+1))+(3/((np.exp(z)+1)**2))-(2/((np.exp(z)+1)**3))