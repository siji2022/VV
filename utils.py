import numpy as np
import copy
import matplotlib.pyplot as plt

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
    