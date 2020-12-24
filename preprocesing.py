import numpy as np
import matplotlib.pyplot as plt
import joblib

import network
import obiekt_regulacji
import Scaler
import simulation


def plot_cost_epoch(train_cost, test_cost, start):
	sim_length = len(train_cost)
	time=[i for i in range(start,sim_length)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	x_ticks = np.arange(0, sim_length, 10)
	x_major_ticks = np.arange(0, sim_length, 20)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)

	# And a corresponding grid
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	plt.plot(time, train_cost[start:], color='r', label = 'dane treningowe')
	plt.plot(time, test_cost[start:], color='b', label = 'dane testowe')

	plt.title(f"Zależność funkcji kosztu od liczby epok")
	plt.xlabel("Liczba epok")
	plt.legend()
	plt.show()

def plot_cost_neurons(train_cost, test_cost, neuron_numbers):
	
	sim_length = len(train_cost)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	x_ticks = np.arange(0, neuron_numbers[-1], 10)
	x_major_ticks = np.arange(0, neuron_numbers[-1], 50)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)

	# And a corresponding grid
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	plt.plot(neuron_numbers, train_cost, color='r', label = 'dane treningowe')
	plt.plot(neuron_numbers, test_cost, color='b', label = 'dane testowe')

	plt.title(f"Zależność kosztu od liczby neuronów w warstwie ukrytej")
	plt.xlabel("Liczba neuronów warstwy ukrytej")
	plt.legend()
	plt.show()



###Load Data
file_train = "dane/dmc_regulation_data_1_train.pkl"
file_test = "dane/dmc_regulation_data_1_test.pkl"

train_data = joblib.load(file_train)
test_data = joblib.load(file_test)
full_data = train_data + test_data
print("Train:{} Test:{} All:{}".format(len(train_data),len(test_data), len(full_data)))

scaler_object = Scaler.Scaler(full_data)
scaled_train = scaler_object.scale_data(train_data)
scaled_test = scaler_object.scale_data(test_data)
print("Data loaded")

#cost for single network
net = network.Network([31, 150, 1])
cost_train, cost_test = net.SGD(scaled_train, 300, 10, 3, scaled_test)

#print("Train cost: {}".format(cost_train[-1]))
#print("Test cost: {}".format(cost_test[-1]))
plot_cost_epoch(cost_train, cost_test, 0)
plot_cost_epoch(cost_train, cost_test, 30)


simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów", 5.21, reference=True)
simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów", 3.0, reference=False)

cost_train, cost_test = net.OBD(scaled_train,scaled_test)
plot_cost_epoch(cost_train, cost_test, 0)

'''
#iteration by neuron number in hidden layer
cost_by_neuron_train = []
cost_by_neuron_test = []
neuron_numbers = [i for i in range(5,30,5)]+[i for i in range(30,100,10)]+[i for i in range(100,210,50)]

for neuron_number in neuron_numbers:
	net = network.Network([31, neuron_number, 1])
	cost_train, cost_test = net.SGD(scaled_train, 200, 10, 3, scaled_test)
	cost_by_neuron_train.append(cost_train[-1])
	cost_by_neuron_test.append(cost_test[-1])
	print("Net with {} complited".format(neuron_number))

#print("Train cost: {}".format(cost_train[-1]))
#print("Test cost: {}".format(cost_test[-1]))
plot_cost_neurons(cost_by_neuron_train, cost_by_neuron_test,neuron_numbers)
'''


