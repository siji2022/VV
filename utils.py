import numpy as np
import copy
import matplotlib.pyplot as plt
import lhsmdu
import time

def init(n_agents,nx_system, type='random'):
    # initial condition with all zeros
    if type == 'zero':
        x = np.zeros((n_agents,nx_system))
    elif type == 'random':
        x = np.random.rand(n_agents,nx_system)-0.5
    elif type == 'base':
        x=np.array([0.,0,1,0]).reshape(1,nx_system)
    return x

def exact_solution_state(x, u, t):
    # update the state using exact solution
    # u is constant
    # deep copy the state
    # x=copy.deepcopy(x)
    # x[:,0:2] = x[:,0:2] + x[:,2:4]*t + (1/2)*u*t**2
    # x[:,2:4] = x[:,2:4] + u*t
    
    # return x
    return numerical_solution_state1(x, u, t)

def numerical_solution_state(x, u, dt):
    # update the state using numerical solution, submitted in HW2
    # u is constant
    # x is iterativly updated
    x_new=copy.deepcopy(x)
    x_new[:, :2] = x[:, :2]+  x[:, 2:]*dt
    x_new[:, 2:] = x[:, 2:] + u*dt
    
    return x_new

def numerical_solution_state1(x, u, dt):
    # update the state using numerical solution
    # u is constant
    # x is iterativly updated
    x=copy.deepcopy(x)
    x[:, :2] += x[:, 2:]*dt
    x[:, :2] += u*dt*dt*0.5
    x[:, 2:] += u*dt
    
    return x

def potential_grad(pos_diff, r2):
    """
    Computes the gradient of the potential function for flocking proposed in Turner 2003.
    Args:
        pos_diff (): difference in a component of position among all agents
        r2 (): distance squared between agents

    Returns: corresponding component of the gradient of the potential

    """
    r=np.sqrt(r2)
    # now shift r to manipulate the distance
    r=r
    # r2_2=np.nan_to_num(r2_2,nan=np.Inf)
    # grad = -1.0 * np.divide(pos_diff, r2*r2) + 1 * np.divide(pos_diff, r2)
    grad = -1.0 * np.divide(pos_diff, r**3) + 1 * np.divide(pos_diff, r)
    # grad = -1.0 * np.divide(pos_diff, r2) + 1 * np.divide(pos_diff, r)
    grad[r > 1.0] = 0 # this cut-off lead to osciliation
    return grad*2.0

def controller_centralized(diff, r2 ):
    """
    The controller for flocking from Turner 2003.
    Returns: the optimal action
    """
    n_agents = diff.shape[0]
    nx_system = diff.shape[2]
    # self dist is not considered
    np.fill_diagonal(r2, np.Inf)
    potentials = np.dstack((diff, potential_grad(diff[:, :, 0], r2), potential_grad(diff[:, :, 1], r2)))

    # potentials = np.nan_to_num(potentials, nan=0.0)  # fill nan with 0
    p_sum = np.sum(potentials, axis=1).reshape((n_agents, nx_system + 2))
    controls = np.hstack(((- p_sum[:, 4] - p_sum[:, 2]*0).reshape(
        (-1, 1)), (- p_sum[:, 3]*0 - p_sum[:, 5]).reshape(-1, 1)))
    
    return controls

def controller_gamma(x,  Destination, C1_gamma, C2_gamma, r_scale, base):
        Destination_x, Destination_y, Desired_v_x, Desired_v_y = Destination
        agent_p = x[:, :2]  # position of agent i
        agent_v = x[:, 2:]  # velocity of agent i
        u_gamma = -C1_gamma * sigma_1(agent_p - [Destination_x, Destination_y], r_scale, base) - C2_gamma * (
            agent_v - [Desired_v_x, Desired_v_y])
        return u_gamma

def sigma_1(z, r_scale, base=1.0):
    z=z/r_scale # scale z as the comm_range = 4.0
    sigma_1_val = z/np.sqrt(base + z**2)
    return sigma_1_val

def calc_SRQs(x_history,u_history,diff_history):
    # calculate SRQs
    # find when u is 0
    # u_history is the control history
    # x_history is the state history
    # diff_history is the difference history
    # find the convergence time
    try:
        u_norm=np.linalg.norm(u_history,axis=2)
        thr=1e-3
        u_norm[u_norm<thr]=0
        u_norm[u_norm>=thr]=1
        u_norm_diff=np.diff(u_norm,axis=0)
        # find the time when u is 0
        u_zero_time=np.where(u_norm_diff==-1)
        converged_steps=np.max(u_zero_time)
    except:
        converged_steps=-1

    # final step's min distance
    min_dist=np.linalg.norm(diff_history[-1,:,:,:2],axis=2)
    np.fill_diagonal(min_dist,np.inf)
    min_dist=np.min(min_dist)
    # max u
    max_u=np.max(np.abs(u_history))
    # variance after convergence
    diff_var_position=np.var(np.var(diff_history[converged_steps:,:,:,:2],axis=(0)))
    diff_var_velocity=np.var(diff_history[converged_steps:,:,:,2:])
    return converged_steps,min_dist, max_u, diff_var_position, diff_var_velocity


def one_simulation_hw5(noise=0, fix_seed=True, save_fig=False, simulation_sec=15, simulation_noise=None, x_init=None):
    ## make hw 5 easier
    # return the 2 SRQs
    if fix_seed:
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


    if x_init is None:
        print('initialize x_init')
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
        obs_diff=diff[: ]+noise
        if simulation_noise is not None:
            obs_diff+=simulation_noise
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
    velocity_diff_norm=np.linalg.norm(diff_history[:,:,:,2:],axis=3)
    velocity_diff_max_SRQ = np.max(velocity_diff_norm,axis=(1,2))
    # save SRQs
    # min_distance_hisotry_SRQ=np.array(min_distance_hisotry_SRQ)
    # velocity_diff_max_SRQ=np.array(velocity_diff_max_SRQ)
    if save_fig:
        plt.clf()
        fig,axs = plt.subplots(2,1,figsize=(10,8))
        axs = np.ravel(axs)
        # # visualize the trajectory convergence
        # first plot is the min distance between the swam
        axs[0].plot(min_distance_hisotry_SRQ,label=f'min distance {noise}')
        axs[0].set_title('SRQs')
        axs[0].set_ylabel('min distance')   
        axs[0].grid()

        # 2nd plot is the velocity diff norm
        
        axs[1].plot(velocity_diff_max_SRQ,label=f'velocity diff norm max {noise}')
        axs[1].set_ylabel('velocity diff norm max')
        axs[1].grid()
        axs[0].legend()
        axs[1].legend()
    
    return min_distance_hisotry_SRQ[-1], velocity_diff_max_SRQ[-1]
    
# calculate the area between two CDFs
def calc_AVM(SRQ1, SRQ2, ds=0.01):
    # calculate the area between two CDFs
    # SRQ1, SRQ2: two CDFs
    # ds: step size
    # return: area between two CDFs
    # sort SRQ1 and SRQ2
    SRQ1.sort()
    SRQ2.sort()
    # each CDF needs to have the same length
    assert len(SRQ1)==len(SRQ2)

    AVM=0
    for i in range(len(SRQ1)-1):
        AVM+=np.abs(SRQ1[i]-SRQ2[i])*ds
    return AVM






# if main is called, run the test
if __name__ == '__main__':
    # test sigma_1
    plt.clf()
    z_orig=np.arange(-5,5,0.005)
    z=z_orig
    # z=z_orig/4.0
    sigma_1_val=sigma_1(z,1.0)
    sigma_1_val1=sigma_1(z,0.1)
    sigma_1_val2=sigma_1(z,5.0)
    
    plt.plot(z_orig,sigma_1_val1,label='base 0.1')
    plt.plot(z_orig,sigma_1_val,label='base 1.0')
    plt.plot(z_orig,sigma_1_val2,label='base 5.0')
    plt.xlabel('z')
    plt.ylabel('sigma_1')
    plt.legend()
    plt.title('sigma_1')
    plt.grid()
    plt.savefig('./plots/sigma_1.png')
    