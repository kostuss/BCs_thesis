'''
A modul to generate regulation examples for futher network learning. 

'''
import numpy as np
import pickle
import joblib
import sys
import obiekt_regulacji


def get_size_MB(object):
	return sys.getsizeof(object)/(1024*1024)

def print_size(object): 
	print("Rozmiar: {:.5f} MB ".format(get_size_MB(object)))


#lista przykladow uczacych
learning_data = []

# regulacja DMC
iterations = 40
D = 60
K = 1


#iteracja po T1 i T2
for T1 in np.arange(2,9,1):
	for T2 in np.arange(1,T1,1):
		#iteracja po opoznieniu
		for TD in np.arange(1,7,1):
			# wektor odpowiedzi skokowych
			s_list = np.array(obiekt_regulacji.generate_s_vaules(D, T1, T2, K, TD))
			#iteracja po wartosciach zadanych
			for zad in np.arange(1,10,0.1):
				#obiekt regulacji
				sim_object= obiekt_regulacji.SimObject(T1, T2, K, TD)
				dmc = obiekt_regulacji.DMC(D, D, s_list, 1, 1)
				#poczatkowe wartosci sterowania
				u_value_dmc = 0.0
				y_k_dmc = 0.0
				#poczatkowa petla przy wartosci zadanej 0
				y_zad = 0.0
				u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object, 5,
				 y_zad, u_value_dmc , y_k_dmc)

				#zmiana wartosci zadanej i regulacja
				y_zad = zad
				u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object, iterations,
				 y_zad, u_value_dmc , y_k_dmc)

				#powrot wartosci zadanej do 0
				#y_zad = 0.0
				#u_value_dmc, y_k_dmc = obiekt_regulacji.simulation_loop(dmc, sim_object, iterations,
				# y_zad, u_value_dmc , y_k_dmc)

				sim_object.calculate_e_values()
				#obiekt_regulacji.plot_simulation(sim_object)

				for i in range(len(sim_object.u_list)):
					#pojedynczy przyklad uczacy
					X = np.zeros(31)

					#start = i - 29 if i- 29 > 0 else 0
					##przepisanie przeszylych 29 sygnalow sterowania lub kr
					#for idx, u in enumerate(np.flip(sim_object.u_list[start:i])):
					#	X[idx] = u

					start = i - 29 if i - 29 > 0 else 0
					#przepisanie obecnej i przeszlych 29 wartosci uchybu
					for idx, e in enumerate(np.flip(sim_object.e_list[start:i+1])):
						X[idx] = e
					
					#dodanie obecnego wyjscia
					X[30] = sim_object.y_zad_list[i]

					#dodanie przykladu uczacego do listy
					learning_data.append((X.reshape(-1,1), sim_object.u_list[i]))


print(len(learning_data))
print_size(learning_data)
print(type(learning_data[0]))
print(learning_data[0][0].shape)

joblib.dump(learning_data, 'dane/dmc_regulation_examples_3.pkl',1)  
#zapisanie danych w formacie pickle
#pickle.dump(learning_data, open("dane/dmc_regulation_examples.p", "wb"))


'''
print("Dlugosc: ", len(sim_object.y_zad_list))
print("Uchyb regulacji: \n", sim_object.e_list)
print("Wartosci sterowania: \n",sim_object.u_list)
print("Wyjscia obiektu: \n", sim_object.y_list)
'''