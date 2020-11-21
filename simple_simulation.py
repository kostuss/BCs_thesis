import time
import numpy as np
import matplotlib.pyplot as plt


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
		u_t=self.Kp*error + self.Ki*sum(self.errors) + self.Kd*(self.errors[-2]-error)
		return u_t

	def set_setPoint(self,setPoint):
		self.setPoint=setPoint

class SimulationObject: 

	def __init__(self):

		self.output_list=[0]
		self.setPoint_list=[0]
		self.u_list=[0]

	def make_simulation_step(self, u, setPoint):
		y=0.9*self.output_list[-1] + 0.1*self.u_list[-1]
		self.u_list.append(u)
		self.setPoint_list.append(setPoint)
		self.output_list.append(y)
		return y

	def get_output_list(self):
		return self.output_list

	def get_u_list(self):
		return self.u_list

class DMC:

	def __init__(self, S_list, psi_list, lambda_list, N,):
		
		self.Psi=np.diag(psi_list)
		self.Lambda=np.diag(lambda_list)
		self.D=len(S_list)
		self.N=N
		self.M_p=




if __name__ == "__main__":

	PID = SimplePID()
	sim = SimulationObject()
	target = 5
	PID.set_setPoint(target)
	output=None
	u_value=0
	for i in range(99):
		output= sim.make_simulation_step(u_value, target)
		print(output)
		u_value=PID.get_u(output)

	print(len(sim.output_list))

	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	time=[i for i in range(100)]
	ax.scatter(time, sim.output_list, color='r')
	ax.scatter(time, sim.setPoint_list, color='b')
	#ax.set_xlabel('Grades Range')
	#ax.set_ylabel('Grades Scored')
	ax.set_title('scatter plot')
	plt.show()
