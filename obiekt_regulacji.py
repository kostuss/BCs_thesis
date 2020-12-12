import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv


class SimObject: 

	def __init__(self, T1, T2, K , TD):

		#parameters defininig regulation object
		self.T1 = T1
		self.T2 = T2
		self.K = K
		self.TD = TD

		#constatns for equation
		self.alpha_1 = math.exp(-(1/T1))
		self.alpha_2 = math.exp(-(1/T2))
		self.a_1 = -self.alpha_1-self.alpha_2
		self.a_2 = self.alpha_1*self.alpha_2
		self.b_1 = (K/(T1-T2))*(T1*(1-self.alpha_1)-T2*(1-self.alpha_2))
		self.b_2 = (K/(T1-T2))*(self.alpha_1*T2*(1-self.alpha_2) - \
			self.alpha_2*T1*(1-self.alpha_1))

		self.u_list = np.zeros(0)
		self.y_list = np.zeros(0)
		self.y_zad_list = np.zeros(0)
		self.e_list = np.zeros(0)

	def get_lag_u(self, lag):

		if lag>len(self.u_list):
			return 0.0
		else:
			return self.u_list[-lag]

	def get_lag_y(self, lag):

		if lag>len(self.y_list):
			return 0.0
		else: 
			return self.y_list[-lag]

	def make_simulation_step(self, u, y_zad):
		#first lag
		u_lag1=self.get_lag_u(self.TD+1)
		u_lag2=self.get_lag_u(self.TD+2)
		y_lag1=self.get_lag_y(1)
		y_lag2=self.get_lag_y(2)

		y_current=self.b_1*u_lag1 + self.b_2*u_lag2 - self.a_1*y_lag1 - self.a_2*y_lag2
		self.y_list = np.append(self.y_list, y_current)
		self.u_list = np.append(self.u_list, u)
		self.y_zad_list = np.append(self.y_zad_list, y_zad)
		return y_current

	def calculate_e_values(self):
		self.e_list = self.y_zad_list - self.y_list

class DMC:

	def __init__(self, n_u, n, S_list, psi_const, lambda_const):
		
		#horyzont sterwania - ile przyszlych wartosci sterowania wyznaczanych jest w kazdej iteracji algorytmu
		self.n_u = n_u
		#horyzont predykcji - 
		self.n = n 
		#horyzont dynamiki obiektu - ile s wyzaczamy (10, 20, 30)
		self.d = len(S_list)

		self.Y_k = np.zeros(n)
		#wektor wartosci zadanej - stala na calym horyzoncie N
		self.Y_zad = np.zeros(n)
		#wektor opisujacy trajektorie sygnalu wyjsciowego 
		self.Y_hat = np.zeros(n)

		#wektor wyznaczanych przyrostow wartosci sterowania 
		self.U_delta = np.zeros(n_u)

		#wektor przeszlych przyrostow sterowania
		self.U_delta_P = np.zeros(len(S_list)-1)


		self.Psi = np.diag([psi_const for i in range(n)])
		self.Lambda = np.diag([lambda_const for i in range(n_u)])
		
		#macierz
		self.M_p = self.init_M_p(S_list, n)

		#macierz
		self.M = self.init_M(S_list, n, n_u)

		self.K = inv(self.M.T @ self.Psi @ self.M + self.Lambda) @ self.M.T @ self.Psi


	def init_M_p(self, S_list, N):

		arr=np.zeros((N,len(S_list)-1))
		for i in range(N):
			for j in range(len(S_list)-1):
				arr[i][j] = (S_list[i+j+1] if i+j+1+1<=len(S_list) else S_list[-1]) - S_list[j]
		return arr

	def init_M(self, S_list, n, n_u):

		arr=np.zeros((n,n_u))
		for i in range(n):
			for j in range(i+1 if i+1<=n_u else n_u):
				arr[i][j]= S_list[i-j] if i-j+1<=len(S_list) else S_list[-1]

		return arr

	#ustawienie wartosi zadanej
	def set_Y_zad(self, Y_zad):
		#wartosc zadana stala na calym horyzoncie
		self.Y_zad.fill(Y_zad)

	def set_Y_k(self, Y_k):
		self.Y_k.fill(Y_k)

	def update_U_delta_P(self, u_delta_current):
		self.U_delta_P = np.delete(np.insert(self.U_delta_P, 0, u_delta_current),-1)

	#wyznaczenie kolejnych wartosci sterowania
	def calculate_U_delta(self, Y_current): 

		self.Y_k.fill(Y_current)
		self.U_delta = self.K @ (self.Y_zad - self.Y_k - (self.M_p @ self.U_delta_P))
		delta_u = self.U_delta[0]
		self.update_U_delta_P(delta_u)

		return delta_u

class SimplePID:

	def __init__(self, P=0.2, I=0.1, D=0.1):

		self.Kp = P
		self.Ki = I
		self.Kd = D
		self.errors=[0]
		self.setPoint=0

	def get_u(self, output):
		error=self.setPoint-output
		self.errors.append(error)
		u_t=self.Kp*error + self.Ki*sum(self.errors) + self.Kd*(error-self.errors[-2])
		return u_t

	def set_setPoint(self,setPoint):
		self.setPoint=setPoint


def generate_s_vaules(D, T1, T2, K, TD):
	sim_object = SimObject(T1,T2,K,TD)

	for i in range(D):
		sim_object.make_simulation_step(1.0, 1.0)
	return sim_object.y_list[:]

def simulation_loop(controller, sim_object, iterations, y_zad, u_value, y_value):
	controller.set_Y_zad(y_zad)
	for i in range(iterations):
		u_value += controller.calculate_U_delta(y_value)
		y_value = sim_object.make_simulation_step(u_value, y_zad)

	return u_value, y_value

def plot_simulation(sim_object):
	#wyswietlenie przebiegu regulacji
	sim_length = len(sim_object.y_list)
	time=[i for i in range(sim_length)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	# Major ticks every 20, minor ticks every 5
	#major_ticks = np.arange(0, D+5, 10)
	x_ticks = np.arange(0, sim_length, 1)
	x_major_ticks = np.arange(0, sim_length, 10)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)


	# And a corresponding grid
	ax.grid(which='both')
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	plt.step(time, sim_object.y_list, color='r', label = 'wyjście obiektu', where='post')
	plt.step(time, sim_object.u_list, color='b', label = 'sterowanie', where='post')
	plt.step(time, sim_object.y_zad_list, color='y', label = 'wartość zadana', where='post')
	plt.step(time, sim_object.e_list, color='g', label = 'uchyb_regulacji', where='post')

	plt.title('Regulacja DMC')
	plt.xlabel("k")
	plt.legend()
	plt.show()


def plot_step_response(s_list):
	#wyswietlenie odpowiedzi skokowej obiektu regulacji
	sim_length = len(s_list)
	time=[i for i in range(sim_length)]
	one_vector=[1 for i in range(sim_length)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	# Major ticks every 20, minor ticks every 5
	#major_ticks = np.arange(0, D+5, 10)
	x_ticks = np.arange(0, sim_length, 1)
	x_major_ticks = np.arange(0, sim_length, 10)

	ax.set_xticks(x_ticks, minor=True)
	ax.set_xticks(x_major_ticks)

	# And a corresponding grid
	ax.grid(which='both')
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	plt.step(time, s_list, color='r', where='post')
	plt.step(time, one_vector, color='b', where='post')
	plt.title('Wykres odpowiedzi skokowej')
	plt.show()


if __name__ == "__main__":


	# regulacja DMC
	iterations=40
	T1=5.5
	T2=2.5
	K=1
	TD=8
	D=60

	# wektor odpowiedzi skokowych
	s_list = np.array(generate_s_vaules(D, T1, T2, K, TD))
	plot_step_response(s_list)

	#obiekt refulacji
	sim_object = SimObject(T1, T2, K, TD)
	dmc = DMC(D, D, s_list, 1, 1)
	
	#poczatkowe wartosci sterowania
	u_value_dmc = 0.0
	y_k_dmc = 0.0
	
	#poczatkowa petla przy wartosci zadanej 0
	y_zad = 0.0
	u_value_dmc, y_k_dmc = simulation_loop(dmc, sim_object, 1,
	 y_zad, u_value_dmc , y_k_dmc)
	
	#zmiana wartosci zadanej i regulacja
	y_zad = 10.0
	u_value_dmc, y_k_dmc = simulation_loop(dmc, sim_object, iterations,
	 y_zad, u_value_dmc , y_k_dmc)

	#zmiana wartosci zadanej i regulacja
	y_zad = 0.0
	u_value_dmc, y_k_dmc = simulation_loop(dmc, sim_object, iterations,
	 y_zad, u_value_dmc , y_k_dmc)
	
	#zmiana wartosci zadanej i regulacja
	'''
	y_zad = 0.0
	u_value_dmc, y_k_dmc = simulation_loop(dmc, sim_object, iterations,
	 y_zad, u_value_dmc , y_k_dmc)
	'''

	sim_object.calculate_e_values()
	plot_simulation(sim_object)

