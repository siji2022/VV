# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *


# fix the random seed
# initialize
np.random.seed(0)
n_agents=4
nx_system=4
dt=0.02
initial_sep=3.1
random_init_position_range=20
total_time=100.

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

# start the simulation

dts=[ np.power(2, i)/25. for i  in range(2)]
print(dts)
# store the results
results={}
for dt in dts:
    iterations=int(total_time/dt)
    x_axis=np.arange(0,iterations)*dt
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

    diff_history=np.array(diff_history)
    u_history=np.array(u_history)
    state_history=np.array(state_history)
    # print(calc_SRQs(state_history,u_history,diff_history))
    results[dt]=(diff_history,u_history,state_history,x_axis)


# visualize the trajectory convergence for node 0

fig,axs = plt.subplots(1,1,figsize=(8,5))
axs = np.ravel(axs)
node_i=0

# Richardson extrapolation
u1=results[dts[0]][1]
u2=results[dts[1]][1]
x_axis_u1=results[dts[0]][3]
x_axis_u2=results[dts[1]][3]


# need to downsample u1 so u1 and u2 has the same length
u1_downsample=np.array([u1[i] for i in range(len(x_axis_u1)) if x_axis_u1[i] in x_axis_u2])
u_bar=u1_downsample+(u1_downsample-u2)/3.0


# GCI
fs=3.
error_gci=u1_downsample-u2
error_gci=error_gci* fs/(np.power(2,0.667)-1.)
error_gci_norm=np.linalg.norm(error_gci[:,node_i,2:],axis=1)
u1_downsample_norm=np.linalg.norm(u1_downsample[:,node_i,:],axis=1)
# find the max error
max_error_gci=np.max(error_gci_norm)
print(f'max error gci: {max_error_gci}')
# find the location of the max error
max_error_gci_location=np.argmax(error_gci_norm)
print(f'max error gci location: {max_error_gci_location}')

# plot error_gci as band
plt.fill_between(x_axis_u2, u1_downsample_norm-error_gci_norm,
                 u1_downsample_norm+error_gci_norm, alpha=0.3, color='r',label='GCI uncertainty band')
plt.plot(x_axis_u2,u1_downsample_norm,label=f'dt = 0.02 ', alpha=0.7)
plt.legend()
plt.title('Velocity norm with Uncertainty')
plt.xlabel('iterations')
plt.grid()
# plt.xlim([0,6])
plt.savefig('./plots/hw4_gci.png',dpi=300, bbox_inches='tight')







