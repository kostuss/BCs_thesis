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
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(
		dmc, sim_object_dmc, 1, y_zad, u_value_dmc , y_k_dmc)

	#zmiana wartosci zadanej i regulacja
	y_zad = zad
	u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(
		dmc, sim_object_dmc, iterations, y_zad, u_value_dmc , y_k_dmc)

	###Network Control
	sim_object_net = obiekt_regulacji.SimObject(T1, T2, K, TD)
	net_controller = obiekt_regulacji.NetController(net, scaler_object)

	u_value_net = 0.0
	y_k_net = 0.0

	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(
		net_controller, sim_object_net, 1, y_zad, u_value_net , y_k_net)

	#poczatkowa petla przy wartosci zadanej i regulacja
	y_zad = zad
	u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(
		net_controller, sim_object_net, iterations, y_zad, u_value_net , y_k_net)

	return sim_object_net.get_mse(), sim_object_dmc.get_mse()

def perform_simulation_comp_TD(net, scaler_object):
	#Control simulation 
	#Object identification
	iterations = 45
	T1 = 5
	T2 = 2 
	K = 1
	TD = 0
	D = 60
	MSE_vector = []	
	# wektor odpowiedzi skokowych dla obiektu referencyjnego
	s_list = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD))

	TD_list = [1,2,3,4,5,6]
	y_zad_list = [round(x,1) for x in np.arange(2, 9, 0.1)]
	for TD_temp in TD_list:
		MSE_vector_net = []
		MSE_vector_dmc = []
		MSE_vector_ref = []
		s_list_ref = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD_temp))

		for zad in y_zad_list:
			### Reference Control
			sim_object_dmc_ref = obiekt_regulacji.SimObject(T1, T2, K, TD_temp)
			dmc_ref = obiekt_regulacji.DMC(D, D, s_list_ref, 1, 1)
			u_value_dmc = 0.0
			y_k_dmc = 0.0
			#poczatkowa petla przy wartosci zadanej 0
			y_zad = 0.0
			u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(
				dmc_ref, sim_object_dmc_ref, 1, y_zad, u_value_dmc , y_k_dmc)
			#zmiana wartosci zadanej i regulacja
			y_zad = zad
			u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(
				dmc_ref, sim_object_dmc_ref, iterations, y_zad, u_value_dmc , y_k_dmc)
			###DMC Control
			sim_object_dmc = obiekt_regulacji.SimObject(T1, T2, K, TD_temp)
			dmc = obiekt_regulacji.DMC(D, D, s_list, 1, 1)
			u_value_dmc = 0.0
			y_k_dmc = 0.0
			#poczatkowa petla przy wartosci zadanej 0
			y_zad = 0.0
			u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object_dmc, 1,
			 y_zad, u_value_dmc , y_k_dmc)
			#zmiana wartosci zadanej i regulacja
			y_zad = zad
			u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(
				dmc, sim_object_dmc, iterations, y_zad, u_value_dmc , y_k_dmc)

			###Network Control
			sim_object_net = obiekt_regulacji.SimObject(T1, T2, K, TD_temp)
			net_controller = obiekt_regulacji.NetController(net, scaler_object)
			u_value_net = 0.0
			y_k_net = 0.0

			#poczatkowa petla przy wartosci zadanej 0
			y_zad = 0.0
			u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(
				net_controller, sim_object_net, 1, y_zad, u_value_net , y_k_net)
			#petla przy wartosci zadanej i regulacja
			y_zad = zad
			u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(
				net_controller, sim_object_net, iterations, y_zad, u_value_net , y_k_net)

			#zapisanie MSE dla pojedynczego przypadku y_zad
			MSE_vector_net.append(sim_object_net.get_mse())
			MSE_vector_dmc.append(sim_object_dmc.get_mse())
			MSE_vector_ref.append(sim_object_dmc_ref.get_mse())

			#wyswietlenie wynikow dla kazdego TD dla y_zad 5
			if zad == 5.0 and TD_temp<3:
				obiekt_regulacji.plot_simulation(sim_object_net, 'sieć dla Td={}'.format(TD_temp))
				obiekt_regulacji.plot_simulation(sim_object_dmc, 'DMC dla Td={}'.format(TD_temp))

		#zapisanie MSE dla jednego Td
		MSE_vector.append(
			(np.mean(MSE_vector_net), np.mean(MSE_vector_dmc), np.mean(MSE_vector_ref)))

	return MSE_vector

def perform_simulation_comp_TI(net, scaler_object):
	#Control simulation 
	#Object identification
	iterations = 45 
	T1 = 5
	T2 = 2 
	K = 1
	TD = 0
	D = 60
	MSE_vector = []	
	# wektor odpowiedzi skokowych dla obiektu referencyjnego
	s_list = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD))

	TI_list = zip([6,5,7,4,7,3,8],[2,3,3,1,6,2,1])
	y_zad_list = [round(x,1) for x in np.arange(2, 9, 0.1)]
	for T1_temp, T2_temp in TI_list:
		#wketory MSE dla pojedynczych wart zad
		MSE_vector_net = []
		MSE_vector_dmc = []
		MSE_vector_ref = []

		s_list_ref = np.array(obiekt_regulacji.generate_s_vaules(D, T1_temp, T2_temp, K, TD))
		for zad in y_zad_list:
			### Reference Control
			sim_object_dmc_ref = obiekt_regulacji.SimObject(T1_temp, T2_temp, K, TD)
			dmc_ref = obiekt_regulacji.DMC(D, D, s_list_ref, 1, 1)
			u_value_dmc = 0.0
			y_k_dmc = 0.0
			#poczatkowa petla przy wartosci zadanej 0
			y_zad = 0.0
			u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc_ref, sim_object_dmc_ref, 1,
			 y_zad, u_value_dmc , y_k_dmc)
			#zmiana wartosci zadanej i regulacja
			y_zad = zad
			u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc_ref, sim_object_dmc_ref, iterations,
			 y_zad, u_value_dmc , y_k_dmc)

			###DMC Control
			sim_object_dmc = obiekt_regulacji.SimObject(T1_temp, T2_temp, K, TD)
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
			sim_object_net = obiekt_regulacji.SimObject(T1_temp, T2_temp, K, TD)
			net_controller = obiekt_regulacji.NetController(net, scaler_object)
			u_value_net = 0.0
			y_k_net = 0.0

			#poczatkowa petla przy wartosci zadanej 0
			y_zad = 0.0
			u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, 1,
				y_zad, u_value_net , y_k_net)
			#zmiana wartosci zadanej i regulacja
			y_zad = zad
			u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, iterations,
			 y_zad, u_value_net , y_k_net)

			#zapisanie MSE dla pojedynczego przypadku y_zad
			MSE_vector_net.append(sim_object_net.get_mse())
			MSE_vector_dmc.append(sim_object_dmc.get_mse())
			MSE_vector_ref.append(sim_object_dmc_ref.get_mse())


			#wyswietlenie wynikow dla kazdego TD dla y_zad 5
			if zad == 5.0:
				obiekt_regulacji.plot_simulation(sim_object_net, 'sieć dla T1={} T2={}'.format(T1_temp, T2_temp))
				obiekt_regulacji.plot_simulation(sim_object_dmc, 'DMC dla T1={} T2={}'.format(T1_temp, T2_temp))

		#zapisanie MSE dla jednego Td
		MSE_vector.append((np.mean(MSE_vector_net), np.mean(MSE_vector_dmc), np.mean(MSE_vector_ref)))

	return MSE_vector

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