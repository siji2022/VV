import numpy as np
import copy

def init(n_agents,nx_system, type='random'):
    # initial condition with all zeros
    if type == 'zero':
        x = np.zeros((n_agents,nx_system))
    elif type == 'random':
        x = np.random.rand(n_agents,nx_system)
    elif type == 'base':
        x=np.array([0.,0,1,0]).reshape(1,nx_system)
    return x

def exact_solution_state(x, u, t):
    # update the state using exact solution
    # u is constant
    # deep copy the state
    x=copy.deepcopy(x)
    x[:,0:2] = x[:,0:2] + x[:,2:4]*t + (1/2)*u*t**2
    x[:,2:4] = x[:,2:4] + u*t
    
    return x

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
    r=r/4.0
    # r2_2=np.nan_to_num(r2_2,nan=np.Inf)
    # grad = -1.0 * np.divide(pos_diff, r2*r2) + 1 * np.divide(pos_diff, r2)
    grad = -1.0 * np.divide(pos_diff, r**3) + 1 * np.divide(pos_diff, r)
    # grad = -1.0 * np.divide(pos_diff, r2) + 1 * np.divide(pos_diff, r)
    # grad[r > 1.0] = 0
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
    controls = np.hstack(((- p_sum[:, 4] - p_sum[:, 2]).reshape(
        (-1, 1)), (- p_sum[:, 3] - p_sum[:, 5]).reshape(-1, 1)))
    
    return controls