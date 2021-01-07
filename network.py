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
        argument sizez zwiera listę prezentujaca 
        liczbe neuronow w kolejnych warstwach sieci. 
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #saliencies for each parameter (weight) used in OBD algorithm 
        self.saliencies = [np.zeros((y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]

        #mask for retraining network with freezed weights 
        self.mask = [np.ones((y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.cost_delta_epsilon = 0.000005

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

    def SGD(self, training_data, epochs_limit, mini_batch_size, eta,
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
        total_train_cost = []
        total_test_cost = []
        prev_cost = np.inf
        prev_delta_cost_list = np.full((3,),np.inf)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs_limit):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if (j+1)%50 == 0: 
                print("Epoch {} complete".format(j+1))

            current_cost = self.total_cost(training_data)
            total_train_cost.append(current_cost)

            if test_data:
                total_test_cost.append(self.total_cost(test_data))
            prev_delta_cost_list = np.delete(np.insert(prev_delta_cost_list, 0, np.abs(prev_cost - current_cost)),-1)
            prev_cost = current_cost

            #stopping rule
            if all(prev_delta_cost_list < self.cost_delta_epsilon):
                print("Training ended at epoch {}".format(j+1))
                return total_train_cost, total_test_cost 

        return total_train_cost, total_test_cost 

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
        self.weights = [(w-(eta/len(mini_batch))*nw) * mask
                        for w, nw, mask in zip(self.weights, nabla_w, self.mask)]
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
        print("Accuracy : {} / {} = {:.3f}".format(correct,n_test, correct/n_test))


    def total_cost(self, data):
        """Return quadratic cost for all examples"""
        cost = 0.0

        for x, y in data: 
            a = self.feedforward(x)
            cost += self.cost_function(a, y)/len(data)

        return cost

    def cost_function(self, a, y):
        """Return quadratic cost for a single example  """
        return 0.5*np.linalg.norm(a-y)**2

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

        #TODO : przetestowac rozne epsilony
        epsilon = np.finfo(np.float32).eps

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

    def OBD(self, train_data, test_data):
        
        """Calculate saliencies based on second derivatives h"""
        nabla_h = [np.zeros(w.shape) for w in self.weights]
        par_number = sum([w.shape[0]*w.shape[1] for w in self.weights])

        test_cost = []
        train_cost = []
        prev_cost = np.inf
        prev_delta_cost_list = np.full((4,),np.inf)

        for limit in range(int(0.9*par_number)):
            """ Iterate over training examples """
            for x, y in train_data:
                delta_nabla_h = self.backpropOBD(x,y)
                nabla_h = [nh + dnh
                             for nh, dnh in zip(nabla_h, delta_nabla_h)]

            self.saliencies = [(h * w**2)/(2 * len(train_data))
                                for w, h in zip(self.weights, nabla_h)]

            self.cut_weights(limit+1)
            self.SGD(train_data, 200, 10, 3.0)        
            test_cost.append(self.total_cost(test_data))
            train_cost.append(self.total_cost(train_data))

            current_cost = self.total_cost(test_data)
            
            prev_delta_cost_list = np.delete(np.insert(prev_delta_cost_list, 0, prev_cost - current_cost),-1)
            prev_cost = current_cost

            #stopping rule
            if all(prev_delta_cost_list < self.cost_delta_epsilon/200):
                print("OBD ended after cut of {} out of {} weights".format(limit+1, par_number))
                return train_cost, test_cost 

        return train_cost, test_cost

    def OBD_full(self, train_data, test_data):
        
        """Calculate saliencies based on second derivatives h"""
        nabla_h = [np.zeros(w.shape) for w in self.weights]
        par_number = sum([w.shape[0]*w.shape[1] for w in self.weights])

        test_cost = []
        train_cost = []
        for limit in range(int(0.95*par_number)):
            """ Iterate over training examples """
            for x, y in train_data:
                delta_nabla_h = self.backpropOBD(x,y)
                nabla_h = [nh + dnh
                             for nh, dnh in zip(nabla_h, delta_nabla_h)]

            self.saliencies = [(h * w**2)/(2 * len(train_data))
                                for w, h in zip(self.weights, nabla_h)]

            self.cut_weights(limit+1)
            self.SGD(train_data, 200, 10, 3.0)        
            test_cost.append(self.total_cost(test_data))
            train_cost.append(self.total_cost(train_data))

        return train_cost, test_cost

    def backpropOBD(self, x, y):
        
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

        #wsteczna propagacja drugich pochodnych
        delta = self.cost_derivative(activations[-1], y)
        delta2_z = self.boundry_OBD_derivative(zs[-1], y)
        h_vector[-1] = np.dot(delta2_z, activations[-2].transpose()**2)

        #iteracja po kolejnych warstwach
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            sp2 = sigmoid_second_prime(z)

            #(7) in LeCun
            delta = np.dot(self.weights[-l+1].transpose(), delta * sigmoid_prime(zs[-l+1]))        
            #backpropagate second derivative (7) in LeCun
            delta2_z =  sp**2 * np.dot(self.weights[-l+1].transpose()**2, delta2_z) 
            # + \
            #for testing approximation stated in LeCun
            # sp2 * delta 

            #second_derivaties[-l] = delta2_z
            h_vector[-l] = np.dot(delta2_z, activations[-l-1].transpose()**2)
           
        return h_vector

    def boundry_OBD_derivative(self, weighted_sums, y):
        """
        Boundry condition at the output layer 
        (8) in LeCun 
        """
        return 2 * sigmoid_prime(weighted_sums)**2 #- \
            #2 * (y - sigmoid(weighted_sums)) * sigmoid_second_prime(weighted_sums)

    def cut_weights(self, limit):
        """Cuts given part of weights by setting it to zero and freezing """

        #List of tuples (index of weight , saliency )
        saliencies_list = []

        for i_layer , saliencies in enumerate(self.saliencies):
            for i_row, row in enumerate(saliencies.tolist()):
                for i_column, value in enumerate(row):
                    saliencies_list.append(( [i_layer, i_row, i_column], value))

        saliencies_list.sort(key = lambda x: x[1])

        to_cut = [element[0] for element in saliencies_list[:limit]]

        print("{} out of {} weights cut".format(len(to_cut), len(saliencies_list)))

        self.restore_mask()
        for wt_index in to_cut:
            self.weights[wt_index[0]][wt_index[1]][wt_index[2]] = 0.0
            self.mask[wt_index[0]][wt_index[1]][wt_index[2]] = 0.0

    def restore_mask(self):

        for i in range(len(self.mask)):
            self.mask[i].fill(1.0)


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