import numpy as np
import joblib
import network
import obiekt_regulacji
import Scaler

###Load Datasca
file = "dane/dmc_regulation_examples_1.pkl"
train_dmc = joblib.load(file)
scaler_object = Scaler.Scaler(train_dmc)
scaled_train = scaler_object.scale_data(train_dmc)

print("Data loaded")

###Load Network
net = network.Network([80, 100, 1])

#test different net sizes and learning rates 
net.SGD(scaled_train, 30, 10, 1.5) 

'''
#Control simulation 
#Object identification
iterations=40
T1=5.5
T2=2.5
K=1
TD=3
D=60

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
y_zad = 3.2
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
y_zad = 3.2
u_value_net, y_k_net = obiekt_regulacji.simulation_loop_network(net_controller, sim_object_net, iterations,
 y_zad, u_value_net , y_k_net)


#plot simulations
obiekt_regulacji.plot_simulation(sim_object_dmc, "DMC")
obiekt_regulacji.plot_simulation(sim_object_net, "Siec neuronowa")
'''


