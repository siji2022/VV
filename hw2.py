# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *


def compare_solutions(x, u, title=''):
    trajectory_exact=np.array([x])
    trajectory_numerical=np.array([x])
    # start testing the exact solution and numerical solution
    ts=np.linspace(0,1,10)
    for t in ts:
        if t>0:
            x_update = exact_solution_state(x, u, t)
            trajectory_exact=np.append(trajectory_exact,[x_update],axis=0)

    for t in ts:
        # if t>0:
        #     dt=ts[1]-ts[0]
        #     x = numerical_solution_state(x, u, dt)
        #     trajectory_numerical=np.append(trajectory_numerical,[x],axis=0)
        x_update=numerical_solution_state(x, u, t)
        trajectory_numerical=np.append(trajectory_numerical,[x_update],axis=0)
    # make 2 side by side plots
    ax, fig = plt.subplots(1,2,figsize=(10,5))
    # the first plot is the exact solution
    plt.subplot(1,2,1)
    plt.plot(trajectory_exact[:,0,0],trajectory_exact[:,0,1],'.-',label='exact solution')
    plt.plot(trajectory_numerical[:,0,0],trajectory_numerical[:,0,1],'.',label='numerical solution')
    plt.plot(trajectory_numerical[0,0,0],trajectory_numerical[0,0,1],'o',color='r',markersize=10,label='start')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Location {title}')
    plt.legend()
    plt.grid()
    # the second plot is the numerical solution
    plt.subplot(1,2,2)
    plt.plot(trajectory_exact[:,0,2],trajectory_exact[:,0,3],'.-',label='exact solution')
    plt.plot(trajectory_numerical[:,0,2],trajectory_numerical[:,0,3],'.',label='numerical solution')
    plt.plot(trajectory_numerical[0,0,2],trajectory_numerical[0,0,3],'o',color='r',markersize=10,label='start velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Velocity {title}')
    plt.legend()
    plt.grid()
    plt.savefig(f'./test_exact_numerical_solutions_{title}.png')

    plt.clf()



def order_of_accuracy_test(r,dt,iterations=8, type='base'):
    # r: mesh refinement factor
    # dt: initial time step
    error_norm1=[]
    error_norm2=[]
    error_normInf=[]
    x_axis=[]

    x=init(n_agents,nx_system, type=type)
    # u=np.array([0.,-1]).reshape(1,2)
    # random generate u in the range of -100 to 100
    u=np.random.rand(n_agents,2)
    for _ in range(iterations):
        x_axis.append(dt)
        x_exact = exact_solution_state(x, u, dt)
        x_numerical = numerical_solution_state(x, u, dt)
        raw_error=(x_exact-x_numerical).reshape(-1,1)
        # print(raw_error)
        error_norm1.append(np.linalg.norm(raw_error,ord=1))
        error_norm2.append(np.linalg.norm(raw_error,ord=2))
        error_normInf.append(np.linalg.norm(raw_error,ord=np.inf))
        dt=dt/r
    error_norm1=np.array(error_norm1)
    error_norm2=np.array(error_norm2)
    error_normInf=np.array(error_normInf)
    x_axis=np.array(x_axis)

    last_error_norm1=error_norm1[-1]
    # divide the first element by the 2nd element
    error_norm1_ratio=error_norm1[:-1]/error_norm1[1:]
    error_norm2_ratio=error_norm2[:-1]/error_norm2[1:]
    error_normInf_ratio=error_normInf[:-1]/error_normInf[1:]
    p_norm1=np.log(error_norm1_ratio)/np.log(r)
    p_norm2=np.log(error_norm2_ratio)/np.log(r)
    p_normInf=np.log(error_normInf_ratio)/np.log(r)


    plt.plot(x_axis[:-1],p_norm1,'o-',label='1-norm')
    plt.plot(x_axis[:-1],p_norm2,'o-',label='2-norm')
    plt.plot(x_axis[:-1],p_normInf,'o-',label='Inf-norm')
    # loglog plot
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.title(f'Order of accuracy, last L1 error/dt: {last_error_norm1:.6f}/{dt*r:.6f}')
    plt.legend()
    plt.grid()
    plt.savefig('./test_observed_order_of_accuracy.png')
    plt.clf()

    print('x_axis',x_axis)  
    print('error_norm1',error_norm1)
    plt.plot(x_axis,error_norm1,'o-',label='1-norm')
    plt.plot(x_axis,error_norm2,'o-',label='2-norm')
    plt.plot(x_axis,error_normInf,'o-',label='Inf-norm')
    plt.title(f'Error norm')
    plt.legend()
    # y axis range -1 to 1
    # plt.ylim(-10,10)
    plt.xlabel('dt')
    plt.ylabel('error')
    plt.grid()
    plt.savefig('./test_error.png')

# fix the random seed
np.random.seed(0)
n_agents=1
nx_system=4

# compare the base initialization condition. the result should be a parabola
x=init(n_agents,nx_system, type='base')
u=np.array([0.,-1]).reshape(1,2)
compare_solutions(x, u)

x=init(n_agents,nx_system, type='random')
# random generate u in the range of -100 to 100
u=np.random.rand(n_agents,2)
compare_solutions(x, u, 'random_initialization')

# order of accuracy test with 14 iterations
order_of_accuracy_test(2, 1.0, 14,'random')

