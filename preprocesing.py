import numpy as np
import joblib
import network
import obiekt_regulacji
import Scaler
import simulation

###Load Datasca
file = "dane/dmc_regulation_examples_3.pkl"
train_dmc = joblib.load(file)
scaler_object = Scaler.Scaler(train_dmc)
scaled_train = scaler_object.scale_data(train_dmc)
print("Data loaded")

'''
###Load Network
net = network.Network([40, 100, 1])

#test different net sizes and learning rates 
net.SGD(scaled_train, 10, 10, 1.5) 
simulation.perform_simulation(net, scaler_object, "10 epoch")

net.SGD(scaled_train, 10, 10, 1.5) 
simulation.perform_simulation(net, scaler_object, "20 epoch")

net.SGD(scaled_train, 10, 10, 1.5) 
simulation.perform_simulation(net, scaler_object, "30 epoch")

net.OBD(scaled_train, 0.2)
net.SGD(scaled_train, 3, 10, 1.5) 
simulation.perform_simulation(net, scaler_object, "OBD 0.2")

net.OBD(scaled_train , 0.25)
net.SGD(scaled_train, 3, 10, 1.5) 
simulation.perform_simulation(net, scaler_object, "OBD 0.25")

net.OBD(scaled_train , 0.3)
net.SGD(scaled_train, 3, 10, 1.5) 
simulation.perform_simulation(net, scaler_object, "OBD 0.3")
'''


