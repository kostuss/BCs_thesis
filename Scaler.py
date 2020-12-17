import numpy as np 
import pickle


class Scaler:

	def __init__(self, learning_data):
		self.min_input, self.max_input = 0.0, 0.0
		self.min_output, self.max_output = 0.0, 0.0
		self.find_min_max(learning_data)

	def find_min_max(self, learning_data):

		temp_min_input = 0.0 
		temp_max_input = 0.0 
		temp_min_output = 0.0
		temp_max_output = 0.0 

		for X, Y in learning_data:
			temp_min_input = min(X.min(), temp_min_input)
			temp_min_output = min(Y.min(), temp_min_output)
			temp_max_input = max(X.max(), temp_max_input)
			temp_max_output = max(Y.max(), temp_max_output)

		self.min_input = temp_min_input
		self.min_output = temp_min_output 
		self.max_input = temp_max_input
		self.max_output = temp_max_output

	def scale_input_value(self, value):
		return (value - self.min_input)/(self.max_input - self.min_input)

	def scale_output_value(self, value): 
		return (value - self.min_output)/(self.max_output - self.min_output)

	def inv_scale_input_value(self, value): 
		return value*(self.max_input - self.min_input) + self.min_input

	def inv_scale_output_value(self, value):
		return value*(self.max_output - self.min_output) + self.min_output

	def scale_data(self, learning_data):

		scaled_data = []
		for X, Y in learning_data:
			scaled_data.append((self.scale_input_value(X), self.scale_output_value(Y)))

		return scaled_data


def preproces_data(file):

	train_dmc = pickle.load(open(file, "rb"))
	scaler_object = Scaler(train_dmc)
	scaled_train = scaler_object(train_dmc)

	return scaled_train



if __name__ == "__main__":

	scaler = Scaler([0])
	print(scaler.min_input, scaler.max_input)
