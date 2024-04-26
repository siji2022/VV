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

dts=[ np.power(2, i)/50. for i  in range(4)]

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
    print(calc_SRQs(state_history,u_history,diff_history))
    results[dt]=(diff_history,u_history,state_history,x_axis)


# visualize the trajectory convergence for node 0

fig,axs = plt.subplots(1,1,figsize=(8,5))
axs = np.ravel(axs)
node_i=0
for dt in results:
    diff_history,u_history,state_history, x_axis=results[dt]
    axs[0].plot(x_axis,np.linalg.norm(state_history[:,node_i,2:],axis=1),label=f'velocity norm {node_i} at dt={dt}', alpha=0.7, linestyle='--')
    #plot with dash line style
# Richardson extrapolation
u1=results[dts[0]][2]
u2=results[dts[1]][2]
u3=results[dts[2]][2]
u4=results[dts[3]][2]
x_axis_u1=results[dts[0]][3]
x_axis_u2=results[dts[1]][3]
x_axis_u3=results[dts[2]][3]
x_axis_u4=results[dts[3]][3]

# need to downsample u1 so u1 and u2 has the same length
u1_downsample=[u1[i] for i in range(len(x_axis_u1)) if x_axis_u1[i] in x_axis_u2]
u_bar=u1_downsample+(u1_downsample-u2)/3.0
#plot u_bar
axs[0].plot(x_axis_u2,np.linalg.norm(u_bar[:,node_i,2:],axis=1),label=f'velocity norm {node_i} at standard extrapolation', alpha=1, linestyle='dotted',linewidth=4.0)

axs[0].legend()
axs[0].set_title('velocity norm')
plt.xlabel('iterations')
axs[0].set_ylabel('velocity norm')
axs[0].grid()

plt.savefig('./plots/trajectory_convergence_node0.png',dpi=300, bbox_inches='tight')
plt.clf()

# plot the error terms
# downsample 
u_bar_downsample=np.array([u_bar[i] for i in range(len(x_axis_u2)) if x_axis_u2[i] in x_axis_u4])
u1_downsample=np.array([u1[i] for i in range(len(x_axis_u1)) if x_axis_u1[i] in x_axis_u4])
u2_downsample=np.array([u2[i] for i in range(len(x_axis_u2)) if x_axis_u2[i] in x_axis_u4])
u3_downsample=np.array([u3[i] for i in range(len(x_axis_u3)) if x_axis_u3[i] in x_axis_u4])
u4_downsample=u4
error_u1=u1_downsample-u_bar_downsample
error_u2=u2_downsample-u_bar_downsample
error_u3=u3_downsample-u_bar_downsample
error_u4=u4_downsample-u_bar_downsample

# plt.plot(x_axis_u4,np.linalg.norm(error_u1[:,node_i,2:],axis=1),label=f'error u1- RE', alpha=0.7, )
# plt.plot(x_axis_u4,np.linalg.norm(error_u2[:,node_i,2:],axis=1),label=f'error u2- RE', alpha=0.7, linestyle='--')
# plt.plot(x_axis_u4,np.linalg.norm(error_u3[:,node_i,2:],axis=1),label=f'error u3- RE', alpha=0.7, linestyle='--')
# plt.plot(x_axis_u4,np.linalg.norm(error_u4[:,node_i,2:],axis=1),label=f'error u4- RE', alpha=0.7, linestyle='--')
de_estimate_u1 = error_u1/(np.power(2,0.667)-1)
de_estimate_u2 = error_u2
de_estimate_u3 = error_u3
plt.plot(x_axis_u4,np.linalg.norm(de_estimate_u1[:,node_i,2:],axis=1),label=f'DE estimate dt = 0.02', alpha=0.7)
plt.plot(x_axis_u4,np.linalg.norm(de_estimate_u2[:,node_i,2:],axis=1),label=f'DE estimate dt = 0.04', alpha=0.7)
plt.plot(x_axis_u4,np.linalg.norm(de_estimate_u3[:,node_i,2:],axis=1),label=f'DE estimate dt = 0.08', alpha=0.7)
plt.legend()
plt.title('Velocity norm Error Estimation Zoom')
plt.xlabel('iterations')
plt.grid()
# plt.xlim([0,6])
plt.savefig('./plots/hw4_error.png',dpi=300, bbox_inches='tight')
# plt.savefig('./plots/hw4_error_zoom.png',dpi=300, bbox_inches='tight')

# GCI
fs=3.
error_gci=u1_downsample-u2_downsample
error_gci=error_gci* fs/(np.power(2,0.667)-1.)

# plot error_gci as band
plt.fill_between(x_axis_u4, np.linalg.norm(u1_downsample[:,node_i,2:],axis=1)-np.linalg.norm(error_gci[:,node_i,2:],axis=1),
                 np.linalg.norm(u1_downsample[:,node_i,2:],axis=1)+np.linalg.norm(error_gci[:,node_i,2:],axis=1), alpha=0.3, color='r',label='GCI uncertainty band')
plt.plot(x_axis_u4,np.linalg.norm(u1_downsample[:,node_i,2:],axis=1),label=f'dt = 0.02 ', alpha=0.7)
plt.legend()
plt.title('Velocity norm with Uncertainty')
plt.xlabel('iterations')
plt.grid()
# plt.xlim([0,6])
plt.savefig('./plots/hw4_gci.png',dpi=300, bbox_inches='tight')



# axs[1].plot(u_history[:,node_i,0],label=f'u_x {node_i}')
# axs[1].plot(u_history[:,node_i,1],label=f'u_y {node_i}')
# axs[1].plot(np.linalg.norm(u_history[:,node_i,:],axis=1),label=f'u_norm {node_i}')
# axs[1].legend()
# axs[1].set_title('control input')
# axs[1].grid()





