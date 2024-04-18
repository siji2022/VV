# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *
from time import time

# fix the random seed
# initialize
start_time = time()
np.random.seed(0)
n_agents=10
nx_system=4
dt=0.02
initial_sep=3.1
random_init_position_range=20

# define destination
Destination=(15,15,0,0)
u_gamma_position=1.
u_gamma_velocity=1.0
r_scale=4.0
base=1.0
u_scale=10.0


valid_init=False
while not valid_init:
    x_init=init(n_agents,nx_system, type='random')
    x_init[:,:2]=x_init[:,:2]*random_init_position_range
    # check if the initial condition is valid
    valid_init=True
    for i in range(n_agents):
        for j in range(i+1,n_agents):
            # for each pair of agents, check if the initial separation is valid
            if np.linalg.norm(x_init[i,:2]-x_init[j,:2])<initial_sep:
                valid_init=False
                break
    # agent must have at least one neighbor
    # diff = x_init.reshape((n_agents, 1, nx_system)) - x_init.reshape( (1, n_agents, nx_system))
    # diff=np.linalg.norm(diff[:,:,:2],axis=2)
    # np.fill_diagonal(diff,np.inf)
    # if np.max(np.min(diff,axis=1))>4:
    #     valid_init=False
    #     break

# start the simulation
iterations=int(100./dt)
x=copy.deepcopy(x_init)
diff_history=[]
u_history=[]
state_history=[]
for i in range(iterations):
    # calculate the input for control
    state_history.append(x)
    diff = x.reshape((n_agents, 1, nx_system)) - x.reshape(
                (1, n_agents, nx_system))
    diff_history.append(diff)
    r2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1],diff[:, :, 1])

    u=controller_centralized(diff, r2/r_scale**2)
    u_gamma=controller_gamma(x,Destination,u_gamma_position,u_gamma_velocity,r_scale, base)
    u=(u+u_gamma)*u_scale
    u_history.append(u)
    
    # upate the state
    x=numerical_solution_state1(x, u, dt)
    # if (i+1)%200==0:
    #     # visualize the end station
    #     plt.plot(x[:,0],x[:,1],'o',label=f'at iteration {i}')
    #     # add quiver
    #     plt.quiver(x[:,0],x[:,1],x[:,2],x[:,3],color='b')

diff_history=np.array(diff_history)
u_history=np.array(u_history)
state_history=np.array(state_history)

end_time=time()
print(f'Elapsed time: {end_time-start_time} seconds')
plt.plot(state_history[:,:,0],state_history[:,:,1])
# visualize the initial condition
plt.plot(x_init[:,0],x_init[:,1],'o',label='initial')
# add quiver
plt.quiver(x_init[:,0],x_init[:,1],x_init[:,2],x_init[:,3],color='r',label='initial velocity')

# visualize the end station
plt.plot(state_history[-1,:,0],state_history[-1,:,1],'o',label=f'at iteration {i}')
# add quiver
plt.quiver(state_history[-1,:,0],state_history[-1,:,1],state_history[-1,:,2],state_history[-1,:,3],color='b')


plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Flocking for N={n_agents}')
plt.savefig('./plots/flocking.png',bbox_inches='tight',dpi=300)
plt.clf()

# # visualize the trajectory convergence


# fig,axs = plt.subplots(3,1,figsize=(10,12))
# axs = np.ravel(axs)
# for node_i in range(n_agents):
#     for node_j in range(n_agents):
#         if node_i!=node_j and node_i<node_j:
#             axs[0].plot(np.linalg.norm(diff_history[:,node_i,node_j,:2],axis=1),label=f'position diff {node_i}_{node_j}')
#             axs[1].plot(np.linalg.norm(diff_history[:,node_i,node_j,2:],axis=1),'.', label=f'velocity diff {node_i}_{node_j}')
#     axs[1].plot(np.linalg.norm(state_history[:,node_i,2:],axis=1),label=f'velocity norm {node_i}')
#     axs[2].plot(u_history[:,node_i,0],'.',label=f'u_x {node_i}')
#     axs[2].plot(u_history[:,node_i,1],'.',label=f'u_y {node_i}')
#     axs[2].plot(np.linalg.norm(u_history[:,node_i,:],axis=1),label=f'u_norm {node_i}')

# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# axs[0].set_title('position diff')
# axs[1].set_title('velocity diff')
# axs[2].set_title('control input')

# plt.xlabel('iterations')
# axs[0].set_ylabel('diff norm')
# axs[1].set_ylabel('diff norm')
# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
# plt.savefig('./plots/trajectory_convergence.png',dpi=300, bbox_inches='tight')

# print(calc_SRQs(state_history,u_history,diff_history))

