import numpy as np
import pickle
import network
import obiekt_regulacji
import Scaler


def perform_simulation(net, scaler_object, title, zad, reference = False):
	#Control simulation 
	#Object identification
	iterations = 40
	T1 = 5
	T2 = 2 
	K = 1
	TD = 0
	D = 60

	###DMC Control
	# wektor odpowiedzi skokowych
	s_list = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD))
	#obiekt regulacji
	sim_object_dmc = obiekt_regulacji.SimObject(T1, T2, K, TD)
	dmc = obiekt_regulacji.DMC(D, D, s_list, 1, 1)

	u_value_dmc = 0.0
	y_k_dmc = 0.0

	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, 1,
	 y_zad, u_value_dmc , y_k_dmc)

	#zmiana wartosci zadanej i regulacja
	y_zad = zad
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, iterations,
	 y_zad, u_value_dmc , y_k_dmc)


	###Network Control
	sim_object_net = obiekt_regulacji.SimObject(T1, T2, K, TD)
	net_controller = obiekt_regulacji.NetController(net, scaler_object)

	u_value_net = 0.0
	y_k_net = 0.0

	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, 1,
	 y_zad, u_value_net , y_k_net)

	#poczatkowa petla przy wartosci zadanej i regulacja
	y_zad = zad
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, iterations,
	 y_zad, u_value_net , y_k_net)


	#plot simulations
	if reference:
		obiekt_regulacji.plot_simulation(sim_object_dmc, "DMC")
		print("MSE dmc: {}".format(sim_object_dmc.get_mse()))
	obiekt_regulacji.plot_simulation(sim_object_net, title)
	print("MSE net: {}".format(sim_object_net.get_mse()))

def perform_simulation_comp(net, scaler_object, zad):
	#Control simulation 
	#Object identification
	iterations = 40
	T1 = 5
	T2 = 2 
	K = 1
	TD = 0
	D = 60

	###DMC Control
	# wektor odpowiedzi skokowych
	s_list = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD))
	#obiekt regulacji
	sim_object_dmc = obiekt_regulacji.SimObject(T1, T2, K, TD)
	dmc = obiekt_regulacji.DMC(D, D, s_list, 1, 1)

	u_value_dmc = 0.0
	y_k_dmc = 0.0

	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, 1,
	 y_zad, u_value_dmc , y_k_dmc)

	#zmiana wartosci zadanej i regulacja
	y_zad = zad
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, iterations,
	 y_zad, u_value_dmc , y_k_dmc)


	###Network Control
	sim_object_net = obiekt_regulacji.SimObject(T1, T2, K, TD)
	net_controller = obiekt_regulacji.NetController(net, scaler_object)

	u_value_net = 0.0
	y_k_net = 0.0

	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, 1,
	 y_zad, u_value_net , y_k_net)

	#poczatkowa petla przy wartosci zadanej i regulacja
	y_zad = zad
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, iterations,
	 y_zad, u_value_net , y_k_net)

	return sim_object_net.get_mse(), sim_object_dmc.get_mse()

def perform_simulation_2(net, scaler_object, title, zad, reference = False):
	#Control simulation 
	#Object identification
	iterations=30
	T1=5
	T2=3
	K=1
	TD=4
	D=60

	###DMC Control
	# wektor odpowiedzi skokowych
	s_list = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD))
	#obiekt regulacji
	sim_object_dmc = obiekt_regulacji.SimObject(T1, T2, K, TD)
	dmc = obiekt_regulacji.DMC(D, D, s_list, 1, 1)

	###Network Control
	sim_object_net = obiekt_regulacji.SimObject(T1, T2, K, TD)
	net_controller = obiekt_regulacji.NetController(net, scaler_object)

	#DMC
	u_value_dmc = 0.0
	y_k_dmc = 0.0
	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, 1,
	 y_zad, u_value_dmc , y_k_dmc)

	#NET
	u_value_net = 0.0
	y_k_net = 0.0
	#poczatkowa petla przy wartosci zadanej 0
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, 1,
	 y_zad, u_value_net , y_k_net)

	for y_zad in zad:
		#DMC	
		u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, iterations,
	 		y_zad, u_value_dmc , y_k_dmc)
		#NET
		u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, iterations,
	 		y_zad, u_value_net , y_k_net)
		

	#plot simulations
	if reference:
		obiekt_regulacji.plot_simulation(sim_object_dmc, "DMC")
	obiekt_regulacji.plot_simulation(sim_object_net, title)