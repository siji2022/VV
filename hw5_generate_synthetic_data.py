import numpy as np
import matplotlib.pyplot as plt
# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *
from time import time

# synthetic data
# choose model input parameters to be an aleatory uncertainty --> u; u is the acceleration commands on each robots; the execution on each robot can add a random noise because of the hardware
# what is the range of u

# choose model input parameters to be an aleatory uncertainty --> diff_x; diff_x is the measurement of the distance between the robots; the measurement can add a random noise because of the hardware
# what is the range of diff_x

# pick mean and std for u or diff_x
# u=u+mu-sigma   --> SRQ alhpa
# u = u + mu+sigma  -->SRQ beta


############################################
# Analyze the distribution of u
############################################

# read data from ./data/state_history_4.npy
n_agents=4
state_history=np.load('./data/state_history_4.npy') # 2500, 4, 4
u_history=np.load('./data/u_history_4.npy') # 2500, 4, 2
diff_history=np.load('./data/diff_history_4.npy') # 2500, 4, 4, 4

u_history_x=u_history[:,:,0]
u_history_y=u_history[:,:,1]
u_history_norm=np.linalg.norm(u_history,axis=2)
diff_history_norm=np.linalg.norm(diff_history[:500,:,:,:2],axis=3).flatten()
# remove 0 from diff_history_norm
diff_history_norm=diff_history_norm[diff_history_norm>0]

print(f'mean:{np.mean(diff_history_norm)}, std:{np.std(diff_history_norm.flatten())}')
# mean:5.042931140860366, std:2.4936196830838084

# print(f'mean:{np.mean(u_history_norm, axis=0)}, std:{np.std(u_history_norm,axis=0)}') 
# print(f'mean:{np.mean(u_history_x,axis=0)}, std:{np.std(u_history_x,axis=0)}')
# print(f'mean:{np.mean(u_history_y, axis=0)}, std:{np.std(u_history_y, axis=0)}') 

# mean:[0.03612341 0.03151994 0.03036318 0.04116625], std:[0.32474749 0.30205528 0.26815454 0.41930351]
# mean:[-0.00102763  0.00062413 -0.00291774  0.00429013], std:[0.22251163 0.26536243 0.06792329 0.34750721]
# mean:[-0.00044912 -0.00391744 -0.00028895  0.00412871], std:[0.2392763  0.14764125 0.26116395 0.23814571]

# print the distribution of u_history
fig, axs = plt.subplots(1, 1, figsize=(8,5))
axs = np.ravel(axs)

axs[0].hist(diff_history_norm.flatten(),bins=10,label='distance')
axs[0].set_yscale('log')
axs[0].set_title('Distance Distribusion (norm)')
axs[0].set_xlabel('Distance to others(including self)')
axs[0].set_ylabel('frequency')
axs[0].legend()
# axs[1].hist(u_history_y.flatten(),bins=10,label='u_y')
# axs[1].set_title('u_y')
# axs[1].set_xlabel('u_y')
# # axs[1].set_ylabel('frequency')
# axs[1].set_yscale('log')
# axs[1].legend()

# axs[2].hist(u_history_norm.flatten(),bins=10,label='u_y')
# axs[2].set_title('u_norm')
# axs[2].set_yscale('log')
# axs[2].legend()

plt.savefig('./plots/hw5_u_distribution.png',dpi=300,bbox_inches='tight')
plt.clf()

# fix the random seed
# initialize
start_time = time()
np.random.seed(0)
n_agents=4
nx_system=4
dt=0.04
initial_sep=3.1
random_init_position_range=30

# define destination
Destination=(15,15,0,0)
u_gamma_position=1.
u_gamma_velocity=1.0
r_scale=4.0
base=1.0
u_scale=10.0
############################################
# add noise on u and measure the SRQ(v_norm)
# noise is added at each time stemp consistently following the same normal distribution noise ~ (0, sigma)
############################################
plt.clf()
fig,axs = plt.subplots(2,1,figsize=(10,8))
axs = np.ravel(axs)
for noise in [0, -2.5, 2.5]:
    np.random.seed(0)

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
    iterations=int(15./dt)
    x=copy.deepcopy(x_init)
    diff_history=[]
    u_history=[]
    state_history=[]
    min_distance_hisotry_SRQ=[]


    for i in range(iterations):
        # calculate the input for control
        state_history.append(x)
        diff = x.reshape((n_agents, 1, nx_system)) - x.reshape((1, n_agents, nx_system))
        # add noise on the observation
        obs_diff=diff[: ]+noise
        diff_history.append(diff)
        r2 = np.multiply(obs_diff[:, :, 0], obs_diff[:, :, 0]) + np.multiply(obs_diff[:, :, 1],obs_diff[:, :, 1])
        u=controller_centralized(obs_diff, r2/r_scale**2)
        u_gamma=controller_gamma(x,Destination,u_gamma_position,u_gamma_velocity,r_scale, base)
        u=(u+u_gamma)*u_scale
        
        u_history.append(u)
        np.fill_diagonal(r2,np.Inf)
        diff_min=np.sqrt(np.min(r2))
        min_distance_hisotry_SRQ.append(diff_min)

        # upate the state
        x=numerical_solution_state1(x, u, dt)
        

    diff_history=np.array(diff_history) # 2500, N, N, 4
    u_history=np.array(u_history)
    state_history=np.array(state_history)
    min_distance_hisotry_SRQ=np.array(min_distance_hisotry_SRQ)

    end_time=time()
    print(calc_SRQs(state_history,u_history,diff_history))



    # # visualize the trajectory convergence

    # first plot is the min distance between the swam
    axs[0].plot(min_distance_hisotry_SRQ,label=f'min distance {noise}')
    axs[0].set_title('SRQs')
    axs[0].set_ylabel('min distance')   
    axs[0].grid()

    # 2nd plot is the velocity diff norm
    velocity_diff_norm=np.linalg.norm(diff_history[:,:,:,2:],axis=3)
    velocity_diff_max_SRQ = np.max(velocity_diff_norm,axis=(1,2))
    axs[1].plot(velocity_diff_max_SRQ,label=f'velocity diff norm max {noise}')
    axs[1].set_ylabel('velocity diff norm max')
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    # save SRQs
    min_distance_hisotry_SRQ=np.array(min_distance_hisotry_SRQ)
    velocity_diff_max_SRQ=np.array(velocity_diff_max_SRQ)
    np.save(f'./data/min_distance_hisotry_SRQ_{noise}.npy',min_distance_hisotry_SRQ)
    np.save(f'./data/velocity_diff_max_SRQ_{noise}.npy',velocity_diff_max_SRQ)
plt.savefig('./plots/hw5_trajectory_convergence.png',dpi=300, bbox_inches='tight')


# save the histories
# np.save(f'./data/state_history_{n_agents}.npy',state_history)
# np.save(f'./data/u_history_{n_agents}.npy',u_history)
# np.save(f'./data/diff_history_{n_agents}.npy',diff_history)