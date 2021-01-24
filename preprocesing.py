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
	x_ticks = np.arange(0, sim_length+1, 10)
	x_major_ticks = np.arange(0, sim_length+1, 20)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)

	# And a corresponding grid
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	plt.plot(time, train_cost[start:], color='r', label = 'dane treningowe')
	#plt.plot(time, test_cost[start:], color='b', label = 'dane testowe')

	plt.title(f"Zależność funkcji kosztu od liczby epok")
	plt.xlabel("Liczba epok")
	plt.ylabel("Wartość funkcji kosztu")
	plt.legend()
	plt.show()

def plot_cost_OBD(train_cost, test_cost, test = False):
	sim_length = len(train_cost)
	time=[i for i in range(sim_length)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	x_ticks = np.arange(0, sim_length, 100)
	x_major_ticks = np.arange(0, sim_length, 500)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)

	# And a corresponding grid
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	if test:
		plt.plot(time, test_cost[:], color='b', label = 'dane testowe')
	else:
		plt.plot(time, train_cost[:], color='r', label = 'dane treningowe')

	plt.title(f"Zależność funkcji kosztu od liczby zredukowanych wag")
	plt.xlabel("Liczba zredukowanych wag")
	plt.ylabel("Wartość funkcji kosztu")
	plt.legend()
	plt.show()

def plot_cost_neurons(train_cost, test_cost, neuron_numbers, test = False):
	
	sim_length = len(train_cost)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	x_ticks = np.arange(0, neuron_numbers[-1]+1, 5)
	x_major_ticks = np.arange(0, neuron_numbers[-1]+1, 20)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)

	# And a corresponding grid
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	if test:
		plt.plot(neuron_numbers, test_cost, color='b', label = 'dane testowe')
	else:
		plt.plot(neuron_numbers, train_cost, color='r', label = 'dane treningowe')

	#plt.title(f"Zależność kosztu od liczby neuronów w warstwie ukrytej")
	plt.xlabel("Liczba neuronów warstwy ukrytej")
	plt.ylabel("Wartość funkcji kosztu")
	plt.legend()
	plt.show()

def perform_comparison(net, scaler_object, y_zad_list, show = False):
	MSE_vector_dmc = []
	MSE_vector_net = []
	for zad in y_zad_list:
		mse_net, mse_dmc = simulation.perform_simulation_comp(net, scaler_object, zad)
		MSE_vector_dmc.append(mse_dmc)
		MSE_vector_net.append(mse_net)
		if show:
			simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów z OBD", zad, reference=True)
	return MSE_vector_net, MSE_vector_dmc

if __name__ == "__main__":

	####Load Data
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
	#####

	'''
	#comparison of two methods   
	MSE_vector_dmc = [[] for i in range(5)]
	MSE_vector_net = [[] for i in range(5)]
	exit_vector = []
	cost_vector = []
	#y_zad = [2.5 , 4, 5.21 , 6.1 , 7]
	y_zad = [5.21, 2.5]
	for iteration in range(4):
		net = network.Network([31, 150, 1])
		cost_train, cost_test = net.SGD(scaled_train, 300, 10, 3, scaled_test)
		cost_vector.append(cost_train[-1])
		for idx, zad in enumerate(y_zad):
			#mse_net, mse_dmc = simulation.perform_simulation_comp(net, scaler_object, zad)
			#MSE_vector_dmc[idx].append(mse_dmc)
			#MSE_vector_net[idx].append(mse_net)
			if iteration==0:
				simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów", zad, reference=True)
			else:
				simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów", zad, reference=False)


		print({"Iteration {} finished".format(iteration)})

	print('simulation finished')
	#porowananie wynikow
	#print([[round(np.mean(x),3),round(np.mean(y),3)] for x,y in zip(MSE_vector_net, MSE_vector_dmc)])
	'''

	'''
	#cost for single network
	net = network.Network([31, 150, 1])
	cost_train, cost_test = net.SGD(scaled_train, 300, 10, 3, scaled_test)
	print("Train cost: {}".format(cost_train[-1]))
	print("Test cost: {}".format(cost_test[-1]))
	#plot_cost_epoch(cost_train, cost_test, 0)
	#plot_cost_epoch(cost_train, cost_test, 50)
	#simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów", 5.21, reference=True)
	#simulation.perform_simulation(net, scaler_object, "sieć 150 neuronów", 3.0, reference=False)
	cost_train_OBD, cost_test_OBD = net.OBD(scaled_train,scaled_test)
	print("Train cost after OBD: {}".format(cost_train_OBD[-1]))
	print("Test cost after OBD: {}".format(cost_test_OBD[-1]))
	plot_cost_OBD(cost_train_OBD, cost_test_OBD, True)
	plot_cost_OBD(cost_train_OBD, cost_test_OBD, False)
	
	#plot_cost_OBD(cost_train[:-100], cost_test[:-100], True)
	#plot_cost_OBD(cost_train[:-100], cost_test[:-100], False)
	#simulation.perform_simulation(net, scaler_object, "sieć OBD", 5.21, reference=True)

	y_zad_list = [x for x in np.arange(2, 9, 0.1)]
	MSE_vector_net, MSE_vector_dmc = perform_comparison(net, scaler_object, y_zad_list)

	y_zad_list_2 = [2.5 , 4, 5.21 , 6.1 , 7]
	MSE_vector_net_2, MSE_vector_dmc_2 = perform_comparison(net, scaler_object, y_zad_list_2, True)

	print('simulation finished')
	#porowananie wynikow
	#print([[round(np.mean(x),3),round(np.mean(y),3)] for x,y in zip(MSE_vector_net, MSE_vector_dmc)])
	'''
	for i in range(3):
		#iteration by neuron number in hidden layer
		cost_by_neuron_train = []
		cost_by_neuron_test = []
		neuron_numbers = [i for i in range(5,30,5)]+[i for i in range(30,100,10)]+[i for i in range(100,210,25)]

		for neuron_number in neuron_numbers:
			net = network.Network([31, neuron_number, 1])
			cost_train, cost_test = net.SGD(scaled_train, 200, 10, 3, scaled_test)
			cost_by_neuron_train.append(cost_train[-1])
			cost_by_neuron_test.append(cost_test[-1])
			print("Net with {} complited".format(neuron_number))

		#print("Train cost: {}".format(cost_train[-1]))
		#print("Test cost: {}".format(cost_test[-1]))
		plot_cost_neurons(cost_by_neuron_train, cost_by_neuron_test,neuron_numbers)
		plot_cost_neurons(cost_by_neuron_train, cost_by_neuron_test,neuron_numbers, True)



