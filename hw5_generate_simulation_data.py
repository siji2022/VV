import numpy as np
import matplotlib.pyplot as plt
# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *
from time import time
import lhsmdu

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
# add noise on observation and measure the SRQ: min_distance, velocity_diff_max
# noise is added at each time stemp as diff_observed following the same normal distribution noise ~ (0, 0.1)
# record the SRQ at iteration 500
############################################
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
            
plt.clf()
fig,axs = plt.subplots(2,1,figsize=(10,8))
axs = np.ravel(axs)
for noise in range(3):
    

    
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
        # generate gaussian noise the same size as diff
        obs_noise=lhsmdu.sample(n_agents, 1)-0.5
        obs_noise=np.array(obs_noise)
        # repeat nx_system times for last dimension in obs_noise
        obs_noise=np.repeat(obs_noise[:,np.newaxis],n_agents,axis=1)
        obs_noise=np.repeat(obs_noise,nx_system,axis=2)
        obs_diff=diff[: ] + obs_noise*2.5
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
    axs[0].plot(min_distance_hisotry_SRQ,label=f'min distance {noise}',alpha=0.3)
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
plt.savefig('./plots/hw5_simulation.png',dpi=300, bbox_inches='tight')


# save the histories
# np.save(f'./data/state_history_{n_agents}.npy',state_history)
# np.save(f'./data/u_history_{n_agents}.npy',u_history)
# np.save(f'./data/diff_history_{n_agents}.npy',diff_history)