# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *

# create a 10*3 array with space 0.1
ll=np.random.rand(1)
n_agents=4

def one_simulation(save_fig=False, simulation_sec=10, x_init=None, dt=0.01, exact_solution=True):


    # define destination
    Destination=(15,15,0,0)
    u_gamma_position=1.
    u_gamma_velocity=1.0
    r_scale=4.0
    base=1.0
    u_scale=10.0


    
    # start the simulation
    iterations=int(simulation_sec/dt)
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
        obs_diff=diff[: ]
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
        if exact_solution:
            x=numerical_solution_state1(x, u, dt)
        else:
            x=numerical_solution_state(x, u, dt)

    return x

def order_of_accuracy_test(r,dt,iterations=8, type='base'):
    # r: mesh refinement factor
    # dt: initial time step
    error_norm1_p=[]
    error_norm2_p=[]
    error_normInf_p=[]

    error_norm1_v=[]
    error_norm2_v=[]
    error_normInf_v=[]
    x_axis=[]


    
    for _ in range(iterations):
        x_axis.append(dt)
        x_exact=one_simulation(False, simulation_time, x_init, dt, True)
        x_numerical = one_simulation(False, simulation_time, x_init, dt, False)
        raw_error_p=(x_exact-x_numerical)[:,:2]
        raw_error_v=(x_exact-x_numerical)[:,2:]

        error_norm1_p.append(np.linalg.norm(raw_error_p,ord=1))
        error_norm2_p.append(np.linalg.norm(raw_error_p,ord=2))
        error_normInf_p.append(np.linalg.norm(raw_error_p,ord=np.inf))
        error_norm1_v.append(np.linalg.norm(raw_error_v,ord=1))
        error_norm2_v.append(np.linalg.norm(raw_error_v,ord=2))
        error_normInf_v.append(np.linalg.norm(raw_error_v,ord=np.inf))
        dt=dt/r
    error_norm1_p=np.array(error_norm1_p)
    error_norm2_p=np.array(error_norm2_p)
    error_normInf_p=np.array(error_normInf_p)
    error_norm1_v=np.array(error_norm1_v)
    error_norm2_v=np.array(error_norm2_v)
    error_normInf_v=np.array(error_normInf_v)
    x_axis=np.array(x_axis)

    # last_error_norm1=error_norm1[-1]
    # divide the first element by the 2nd element
    error_norm1_ratio_p=error_norm1_p[:-1]/error_norm1_p[1:]
    error_norm2_ratio_p=error_norm2_p[:-1]/error_norm2_p[1:]
    error_normInf_ratio_p=error_normInf_p[:-1]/error_normInf_p[1:]

    error_norm1_ratio_v=error_norm1_v[:-1]/error_norm1_v[1:]
    error_norm2_ratio_v=error_norm2_v[:-1]/error_norm2_v[1:]
    error_normInf_ratio_v=error_normInf_v[:-1]/error_normInf_v[1:]

    p_norm1_p=np.log(error_norm1_ratio_p)/np.log(r)
    p_norm2_p=np.log(error_norm2_ratio_p)/np.log(r)
    p_normInf_p=np.log(error_normInf_ratio_p)/np.log(r)
    p_norm1_v=np.log(error_norm1_ratio_v)/np.log(r)
    p_norm2_v=np.log(error_norm2_ratio_v)/np.log(r)
    p_normInf_v=np.log(error_normInf_ratio_v)/np.log(r)

    fig,axs = plt.subplots(1,2,figsize=(12,4))
    axs = np.ravel(axs)

    axs[0].plot(x_axis[:-1],p_norm1_p,'o-',label='1-norm')
    axs[0].plot(x_axis[:-1],p_norm2_p,'o-',label='2-norm')
    axs[0].plot(x_axis[:-1],p_normInf_p,'o-',label='Inf-norm')
    # loglog plot
    axs[0].set_xscale('log')
    # plt.yscale('log')
    axs[0].set_xlabel('h')
    axs[0].set_ylabel('Order of accuracy')
    axs[0].set_title(f'Order of accuracy: position')
    axs[0].set_ylim(0,10)
    axs[0].set_yticks(np.arange(0, 10, 1))
    axs[0].grid()

    axs[1].plot(x_axis[:-1],p_norm1_v,'o-',label='1-norm')
    axs[1].plot(x_axis[:-1],p_norm2_v,'o-',label='2-norm')
    axs[1].plot(x_axis[:-1],p_normInf_v,'o-',label='Inf-norm')
    # loglog plot
    axs[1].set_xscale('log')
    # plt.yscale('log')
    axs[1].set_xlabel('h')
    axs[1].set_ylabel('Order of accuracy')
    plt.title(f'Order of accuracy: velocity')
    axs[1].set_ylim(0,10)
    axs[1].set_yticks(np.arange(0, 10, 1))
    axs[1].grid()
    plt.legend()

    plt.savefig(f'./plots/test_observed_order_of_accuracy_{simulation_time}.png')
    plt.clf()

    # print('x_axis',x_axis)  
    # print('error_norm1',error_norm1)
    # plt.plot(x_axis,error_norm1,'o-',label='1-norm')
    # plt.plot(x_axis,error_norm2,'o-',label='2-norm')
    # plt.plot(x_axis,error_normInf,'o-',label='Inf-norm')
    # plt.title(f'Error norm')
    # plt.legend()
    # # y axis range -1 to 1
    # # plt.ylim(-10,10)
    # plt.xlabel('dt')
    # plt.ylabel('error')
    # # grid for every 1
    # plt.grid(which='minor')
    # plt.savefig('./plots/test_error.png')

# fix the random seed
np.random.seed(0)
n_agents=4
nx_system=4

#load x_init
x_init=np.load('./data/x_init.npy')
simulation_time=4.1
# order of accuracy test with 14 iterations
order_of_accuracy_test(2, 0.16, 4,'random')

