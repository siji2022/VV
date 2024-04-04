# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *


# fix the random seed
np.random.seed(0)
n_agents=4
nx_system=4
dt=0.05
initial_sep=3.1

valid_init=False
while not valid_init:
    x_init=init(n_agents,nx_system, type='random')
    x_init[:,:2]=x_init[:,:2]*20
    # check if the initial condition is valid
    valid_init=True
    for i in range(n_agents):
        for j in range(i+1,n_agents):
            if np.linalg.norm(x_init[i,:2]-x_init[j,:2])<initial_sep:
                valid_init=False
                break


iterations=200
x=copy.deepcopy(x_init)
diff_history=[]
for i in range(iterations):
    # calculate the input for control
    diff = x.reshape((n_agents, 1, nx_system)) - x.reshape(
                (1, n_agents, nx_system))
    diff_history.append(diff)
    r2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1],diff[:, :, 1])

    u=controller_centralized(diff, r2)
    # upate the state
    x=numerical_solution_state1(x, u, dt)
    if (i+1)%100==0:
        # visualize the end station
        plt.plot(x[:,0],x[:,1],'o',label=f'at iteration {i}')
        # add quiver
        plt.quiver(x[:,0],x[:,1],x[:,2],x[:,3],color='b')

# visualize the initial condition
plt.plot(x_init[:,0],x_init[:,1],'o',label='initial')
# add quiver
plt.quiver(x_init[:,0],x_init[:,1],x_init[:,2],x_init[:,3],color='r',label='initial velocity')


plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flocking')
plt.savefig('./flocking.png')
plt.clf()
diff_history=np.array(diff_history)
# make two subplots
fig,axs = plt.subplots(2,1,figsize=(10,8))
axs = np.ravel(axs)
for node_i in range(n_agents):
    for node_j in range(n_agents):
        if node_i!=node_j and node_i<node_j:
            # norm position difference
            axs[0].plot(np.linalg.norm(diff_history[:,node_i,node_j,:2],axis=1),label=f'position diff {node_i}_{node_j}')
            axs[1].plot(np.linalg.norm(diff_history[:,node_i,node_j,2:],axis=1),'.', label=f'velocity diff {node_i}_{node_j}')

axs[0].legend()
axs[1].legend()
axs[0].set_title('position diff')
axs[1].set_title('velocity diff')

plt.xlabel('iterations')
axs[0].set_ylabel('diff norm')
axs[1].set_ylabel('diff norm')
axs[0].grid()
axs[1].grid()
plt.savefig('./trajectory_convergence.png')


