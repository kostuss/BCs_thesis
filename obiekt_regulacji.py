import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv


class SimObject: 
	def __init__(self, T1, T2, K , TD):

		#parameters defininig regulation object
		self.T1=T1
		self.T2=T2
		self.K=K
		self.TD=TD

		#constatns for equation
		self.alpha_1=math.exp(-(1/T1))
		self.alpha_2=math.exp(-(1/T2))
		self.a_1=-self.alpha_1-self.alpha_2
		self.a_2=self.alpha_1*self.alpha_2
		self.b_1=(K/(T1-T2))*(T1*(1-self.alpha_1)-T2*(1-self.alpha_2))
		self.b_2=(K/(T1-T2))*(self.alpha_1*T2*(1-self.alpha_2) - \
			self.alpha_2*T1*(1-self.alpha_1))

		self.u_list=[]
		self.y_list=[]

	def get_lag_u(self, lag):

		if lag>len(self.u_list):
			return 0
		else:
			return self.u_list[-lag]

	def get_lag_y(self, lag):

		if lag>len(self.y_list):
			return 0
		else: 
			return self.y_list[-lag]

	def make_simulation_step(self, u):
		#first lag
		u_lag1=self.get_lag_u(self.TD+1)
		u_lag2=self.get_lag_u(self.TD+2)
		y_lag1=self.get_lag_y(1)
		y_lag2=self.get_lag_y(2)

		y_current=self.b_1*u_lag1 + self.b_2*u_lag2 - self.a_1*y_lag1 - self.a_2*y_lag2
		self.y_list.append(y_current)
		self.u_list.append(u)
		return y_current

def generate_s_vaules(Sim_class, D, T1, T2, K, TD):
	sim_object=Sim_class(T1,T2,K,TD)
	for i in range(D+1):
		sim_object.make_simulation_step(1)
	return sim_object.y_list[1:]


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
		self.M = self.init_M(S_list,n,n_u)

		self.K = inv(self.M.T @ self.Psi @ self.M + self.Lambda) @ self.M.T @ self.Psi


	def init_M_p(self, S_list, N):

		arr=np.zeros((N,len(S_list)-1))
		for i in range(N):
			for j in range(len(S_list)-1):
				arr[i][j]= (S_list[i+j+1] if i+j+1+1<=len(S_list) else S_list[-1]) - S_list[j]
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


if __name__ == "__main__":
	'''
	#initialize simulation object
	sim_object1=SimObject(3 ,4, 1, 2)
	sim_object2=SimObject(1, 20, 1, 2)
	#perform simulation

	for i in range(80):
		sim_object1.make_simulation_step(1)
		sim_object2.make_simulation_step(1)
		#print('Step: ',i)

	print(sim_object1.y_list)

	#plot results
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	time=[i for i in range(80)]
	ax.plot(time, sim_object1.y_list, color='r')
	ax.plot(time, sim_object2.y_list, color='y')
	ax.plot(time, sim_object1.u_list, color='b')
	ax.set_title('Simulation')
	plt.show()
	

	s_list=[1,2,4,7]
	dmc=DMC(5, 7, s_list, 1, 1)

	#print(dmc.Psi)
	#print(dmc.Lambda)

	#print(dmc.M_p)
	#print(dmc.M)
	'''

	# regulacja DMC
	y_zad=2
	iterations=150
	T1=5
	T2=4
	K=1
	TD=0

	#DMC regulation
	s_list=generate_s_vaules(SimObject, 100, T1, T2, K, TD)

	print(type(s_list))
	print(len(s_list))
	print(s_list)


	sim_object1=SimObject(T1, T2, K, TD)
	dmc=DMC(50, 50, s_list, 1, 1)
	dmc.set_Y_zad(0)

	print(dmc.M)
	
	#PID
	#pid=SimplePID(P=0.5, I=0.1, D=0.1)
	#pid.set_setPoint(y_zad)
	#sim_object2=SimObject(T1, T2, K, TD)

	u_value_dmc=0
	u_value_pid=0
	y_k_dmc=0

	for i in range(5):
		y_k_dmc=sim_object1.make_simulation_step(u_value_dmc)
		u_value_dmc += dmc.calculate_U_delta(y_k_dmc)

	dmc.set_Y_zad(y_zad)

	for i in range(iterations):
		y_k_dmc = sim_object1.make_simulation_step(u_value_dmc)
		u_value_dmc += dmc.calculate_U_delta(y_k_dmc)
		
		#y_k_pid=sim_object2.make_simulation_step(u_value_pid)
		#u_value_pid=pid.get_u(y_k_pid)

	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	time=[i for i in range(iterations+5)]
	y_zad_list=[y_zad for i in range(iterations+5)]

	ax.plot(time, sim_object1.y_list, color='r')
	ax.plot(time, sim_object1.u_list, color='b')
	ax.plot(time, y_zad_list, color='y')
	ax.set_title('Simulation')
	plt.show()

 	

