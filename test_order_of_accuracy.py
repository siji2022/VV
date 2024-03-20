# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

n_agents=1
nx_system=4

def init(type='random'):
    # initial condition with all zeros
    if type == 'zero':
        x = np.zeros((n_agents,nx_system))
    elif type == 'random':
        x = np.random.rand(n_agents,nx_system)
    elif type == 'base':
        x=np.array([0.,0,1,0]).reshape(n_agents,nx_system)
    return x

def exact_solution_state(x, u, t):
    # update the state using exact solution
    # u is constant
    # deep copy the state
    x=copy.deepcopy(x)
    x[:,2:4] = x[:,2:4] + u*t
    x[:,0:2] = x[:,0:2] + x[:,2:4]*t + (1/2)*u*t**2
    return x

def numerical_solution_state(x, u, dt):
    # update the state using numerical solution
    # u is constant
    # x is iterativly updated
    x[:, 2:] += u*dt
    x[:, :2] += x[:, 2:]*dt
    x[:, :2] += u*dt*dt*0.5
    
    return x

# start testing the exact solution and numerical solution
ts=np.linspace(0,1,20)
x=init()
u=np.array([0.,-1]).reshape(1,2)
trajectory_exact=np.array([x])
trajectory_numerical=np.array([x])

for t in ts:
    x_update = exact_solution_state(x, u, t)
    trajectory_exact=np.append(trajectory_exact,[x_update],axis=0)

for _ in ts:
    dt=ts[1]-ts[0]
    x = numerical_solution_state(x, u, dt)
    trajectory_numerical=np.append(trajectory_numerical,[x],axis=0)

print(trajectory_exact.shape)
print(trajectory_numerical.shape)
plt.plot(trajectory_exact[:,0,0],trajectory_exact[:,0,1],'.',label='exact solution')
plt.plot(trajectory_numerical[:,0,0],trajectory_numerical[:,0,1],'.',label='numerical solution')
plt.plot(trajectory_numerical[0,0,0],trajectory_numerical[0,0,1],'o',color='r',markersize=10,label='start')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exact solution and numerical solution')
plt.legend()
plt.grid()
plt.savefig('./test_order_of_accuracy.png')

plt.clf()