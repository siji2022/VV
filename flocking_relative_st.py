# from collections import deque


# import io
# from PIL import Image

# import numpy as np
# import configparser
# from os import path
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import gca

# import torch
# from torch_sparse import dense_to_sparse

# class Flocking():
#     metadata = {'render.modes': [
#         'human', 'rgb_array'], 'video.frames_per_second': 50}

#     def __init__(self):
#         # number states per agent
#         self.nx_system = 4
#         # number of actions per agent
#         self.nu = 2
#         self.curr_step = 0
#         # default problem parameters
#         self.n_agents = 100  # int(config['network_size'])
#         self.comm_radius = 0.9  # float(config['comm_radius'])

#         self.dt = 0.01  # #float(config['system_dt'])

#         self.comm_radius2 = self.comm_radius * self.comm_radius

#         # intitialize state matrices
#         self.x = None
#         self.u = None
#         self.u_gamma = None  # gamma action (n_agents,2)
#         self.u_beta = None  # this is for controller to remove oscilliation; maybe a defeat in Saber, or my implementation is wrong
#         self.obs_state = None
#         self.mean_vel = None
#         self.init_vel = None

#         self.fig = None
#         self.line1 = None

#         # used to track the performance
#         self.history = []
#         self.u_history = []
#         self.min_history = []
#         self.curr_step=0


#         self.max_accel = 10
#         self.max_velocity = 10


#     def update_state(self, u):
#         # use this to update the state of the system
#         # state is stored in self.x
#         # u = np.clip(u, -self.max_accel, self.max_accel)
#         self.u = u

#         # x position
#         self.x[:, :2] += 0.5*self.x[:, 2:]**2*self.dt
#         self.x[:, :2] += self.u**3*self.dt*0.5*(1/3)

#         self.x[:, 2:] += 0.5*self.u*self.u*self.dt
#         # self.x[:, 2:] = np.clip(
#         #     self.x[:, 2:], -self.max_velocity, self.max_velocity)

#     def step(self, u):
#         assert u.shape == (self.n_agents, self.nu)
        


#         self.compute_helpers()

        
#         return (self.state_values, self.state_network), self.instant_cost(), False, {}

#     def compute_helpers(self):

#         self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape(
#             (1, self.n_agents, self.nx_system))
#         self.r2 = np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1],
#                                                                                     self.diff[:, :, 1])
#         np.fill_diagonal(self.r2, np.Inf)
#         self.min_history += [np.sqrt(np.min(self.r2))]

#         # check x_feature's range
#         self.x_features = np.dstack((self.diff[:, :, 2]/self.RANGE, np.divide(self.diff[:, :, 0], self.r2)*self.RANGE,
#                                      np.divide(
#                                          self.diff[:, :, 0], np.sqrt(self.r2)),
#                                      self.diff[:, :, 3]/self.RANGE, np.divide(
#                                          self.diff[:, :, 1], self.r2)*self.RANGE,
#                                      np.divide(self.diff[:, :, 1], np.sqrt(self.r2))))

#         # non neighbor nodes' features are set to 0
#         if not self.centralized:
#             np.fill_diagonal(self.r2, 0)
#             mask = (self.r2 >= self.comm_radius2)
#             self.x_features[mask] = 0
#             np.fill_diagonal(self.r2, np.Inf)
#             self.adj_mat = (self.r2 < self.comm_radius2).astype(float)
#             self.state_network = self.adj_mat
#         else:
#             self.adj_mat = (self.r2 < np.Inf).astype(float)
#             self.state_network = self.adj_mat
#         # # obs
#         # obs_diff = self.obs_state-self.x  # n_obs,n_agents,nx_system
#         # obs_r2 = np.multiply(obs_diff[:, :, 0], obs_diff[:, :, 0]) + \
#         #     np.multiply(obs_diff[:, :, 1], obs_diff[:, :, 1])  # 3,50
#         # obs_adj = (obs_r2 < self.DISTANCE_OBS *
#         #            self.DISTANCE_OBS).astype(float)  # 3,50
#         # np.fill_diagonal(obs_r2, np.Inf)

#         # # check obs_features range
#         # obs_features = np.dstack((obs_diff[:, :, 2]/self.RANGE_OBS, np.divide(obs_diff[:, :, 0], obs_r2)*self.RANGE_OBS,
#         #                           np.divide(obs_diff[:, :, 0],
#         #                                     np.sqrt(obs_r2)),
#         #                           obs_diff[:, :, 3]/self.RANGE_OBS, np.divide(
#         #                               obs_diff[:, :, 1], obs_r2)*self.RANGE_OBS,
#         #                           np.divide(obs_diff[:, :, 1], np.sqrt(obs_r2))))  # 3,50,6

#         # self.obs_state_values = np.einsum(
#         #     "ij,jik->ik", obs_adj.T, obs_features)

#         # convert into pytorch type
#         self.state_network = torch.tensor(self.state_network)
#         # self.state_network=add_remaining_self_loops(dense_to_sparse(self.state_network)[0])[0]
#         self.state_network = dense_to_sparse(self.state_network)[0]
#         self.x_features = torch.clip(torch.tensor(
#             self.x_features, dtype=torch.float), -self.max_state_value, self.max_state_value)
#         self.state_values = self.x_features
#         self.obs_state_values = None
#         self.obs_state_network = None
#         self.u_gamma = None
#         # # self.obs_state_values = torch.clip(torch.tensor(
#         # #     self.obs_state_values, dtype=torch.float), -self.max_state_value, self.max_state_value)
#         # # self.obs_state_network = torch.tensor(obs_adj, dtype=torch.long)
#         # self.u_gamma = torch.clip(torch.tensor(self.controller_gamma(
#         # ), dtype=torch.float), -self.max_state_value, self.max_state_value)

#     # def get_stats(self):

#     #     stats = calc_metrics(self)
#     #     return stats

#     # def instant_cost(self):  # sum of differences in velocities
#     #     return calc_reward(self)


#     def reset(self):
#         self.fig = None
#         self.history = []
#         self.u_history = []
#         self.min_history = []
#         # reset history queue
#         self.x_queue = deque()
#         self.state_values_queue = deque()
#         self.state_network_queue = deque()
#         self.obs_values_queue = deque()
#         self.obs_network_queue = deque()
#         self.gamma_queue = deque()

#         # initialization simple way, vel is 0
#         x = np.hstack([np.random.uniform(0, np.sqrt(
#             self.n_agents)*self.comm_radius, (self.n_agents, 2)), np.zeros((self.n_agents, 2))])

#         constrant_initialization(self)

    
#         # self.a_net = self.get_connectivity(self.x)
#         # self.obs_state = obs_projection_sphere(
#         #     self.x, self.RK, self.yk)  # number_obs,number,4
#         self.compute_helpers()

#         # populate previous states; initial to the len of k
#         while len(self.x_queue) < self.len:
#             self.x_queue.append(self.x)
#             self.state_values_queue.append(self.x_features)
#             self.state_network_queue.append(self.state_network)
#             self.obs_values_queue.append(self.obs_state_values)
#             self.obs_network_queue.append(self.obs_state_network)
#             self.gamma_queue.append(self.u_gamma)

#         return (self.x_features, self.state_network)

#     def controller(self):
#         if self.gt_centralized:
#             return self.controller_centralized()

#         adjacency_matrix = get_adjacency_matrix(
#             self.x, self.RANGE)  # N*N matrix with True/False.
#         # print(get_adj_min_dist(multi_agent_system.agents))
#         u = np.zeros((self.NUMBER_OF_AGENTS, 2))
#         neighbors_p = self.x[:, :2].reshape(self.NUMBER_OF_AGENTS, 1, 2)
#         neighbors_q = self.x[:, 2:].reshape(self.NUMBER_OF_AGENTS, 1, 2)
#         agent_p = self.x[:, :2].reshape(1, self.NUMBER_OF_AGENTS, 2)
#         agent_v = self.x[:, 2:].reshape(1, self.NUMBER_OF_AGENTS, 2)
#         n_ij = get_n_ij(agent_p, neighbors_p)
#         term1 = np.sum(phi_alpha(sigma_norm(neighbors_p - agent_p), self.RANGE, self.DISTANCE, 0.2)
#                        * n_ij*adjacency_matrix.reshape(self.NUMBER_OF_AGENTS, self.NUMBER_OF_AGENTS, 1), axis=0)

#         a_ij = get_a_ij(neighbors_p, agent_p, self.RANGE, H=0.2)
#         term2 = self.C2_alpha * np.sum(a_ij * (neighbors_q - agent_v)*adjacency_matrix.reshape(
#             self.NUMBER_OF_AGENTS, self.NUMBER_OF_AGENTS, 1), axis=0)
#         u_alpha = self.C1_alpha*term1 + self.C2_alpha*term2
#         u += u_alpha

#         # agent_p = self.x[:, :2]  # position of agent i
#         # agent_v = self.x[:, 2:]  # velocity of agent i
#         # u_gamma = -self.C1_gamma * sigma_1(agent_p - [self.Destination_x, self.Destination_y]) - self.C2_gamma * (
#         #     agent_v - [self.Desired_v_x, self.Desired_v_y])
#         # u += u_gamma
#         # u_beta = 0
#         # for j in range(self.NUMBER_OF_OBS):  # TODO OPTIMIZE
#         #     curr_obs = self.obs_state[j]
#         #     adjacency_matrix_obs = get_adjacency_matrix_obs(
#         #         self.x, curr_obs, self.RANGE_OBS)
#         #     # if np.linalg.norm(obs_state[j,i,:2]-multi_agent_system.agents[i,:2],axis=-1) <RANGE_OBS:
#         #     obs_p = curr_obs[:, :2]
#         #     obs_q = curr_obs[:, 2:]
#         #     n_ij = get_n_ij(agent_p, obs_p)
#         #     term1 = self.C1_beta * np.sum(phi_alpha(sigma_norm(obs_p - agent_p), self.RANGE_OBS, self.DISTANCE_OBS, 0.2)
#         #                                   * n_ij*adjacency_matrix_obs.reshape(self.NUMBER_OF_AGENTS, self.NUMBER_OF_AGENTS, 1), axis=0)

#         #     # term2
#         #     a_ij = get_a_ij(obs_p, agent_p, self.RANGE_OBS, 0.9)
#         #     term2 = self.C2_beta * np.sum(a_ij * (obs_q - agent_v)*adjacency_matrix_obs.reshape(
#         #         self.NUMBER_OF_AGENTS, self.NUMBER_OF_AGENTS, 1), axis=0)

#         #     if np.max(term1) > 5:
#         #         sdsd = 1
#         #     if np.max(term2) > 5:
#         #         sdsd = 1
#         #     # u_alpha
#         #     u_beta += (term1 + term2)
#         # if self.u_beta is not None:
#         #     u_beta = self.u_beta*0.2+u_beta*0.8

#         # u += u_beta
#         # self.u_beta = u_beta
#         # if np.sum(term1) != 0:
#         #     print(f'{np.sum(n_ij)}, {np.sum(term1)},term2: {np.sum(term2)}')

#         return np.clip(u, -self.max_accel, self.max_accel)

#     def controller_centralized(self):
#         """
#         The controller for flocking from Turner 2003.
#         Returns: the optimal action
#         """
#         # normalize based on comm_radius
#         diff = self.diff/self.comm_radius
#         r2 = self.r2/self.comm_radius2
#         r2 = r2
#         # TODO use the helper quantities here more?
#         potentials = np.dstack((diff, self.potential_grad(
#             diff[:, :, 0], r2), self.potential_grad(diff[:, :, 1], r2)))
#         potentials = np.nan_to_num(potentials, nan=0.0)  # fill nan with 0
#         # ground truth centralized or not
#         # if not centralized:
#         #     potentials = potentials * \
#         #         self.adj_mat.reshape(self.n_agents, self.n_agents, 1)

#         p_sum = np.sum(potentials, axis=1).reshape(
#             (self.n_agents, self.nx_system + 2))
#         controls = np.hstack(((- p_sum[:, 4] - p_sum[:, 2]).reshape(
#             (-1, 1)), (- p_sum[:, 3] - p_sum[:, 5]).reshape(-1, 1)))
#         # controls+=self.controller_gamma()
#         controls = np.clip(controls*self.comm_radius, -
#                            self.max_accel, self.max_accel)

#         return controls

#     def potential_grad(self, pos_diff, r2):
#         """
#         Computes the gradient of the potential function for flocking proposed in Turner 2003.
#         Args:
#             pos_diff (): difference in a component of position among all agents
#             r2 (): distance squared between agents

#         Returns: corresponding component of the gradient of the potential

#         """
#         r2_2 = np.multiply(r2, r2)
#         # r2_2=np.nan_to_num(r2_2,nan=np.Inf)
#         grad = -1.0 * np.divide(pos_diff, r2_2
#                                 ) + 1 * np.divide(pos_diff, r2)
#         grad[r2 > 1.0] = 0
#         return grad*2.0

#     def plot(self, j=0, fname='',dir='plots'):
#         plot_details(self,j,fname,dir)

#     def render(self, mode='human'):
#         """
#         Render the environment with agents as points in 2D space
#         """
#         if self.fig is None:
#             plt.clf()
#             plt.ion()
#             fig = plt.figure()
#             self.ax = fig.add_subplot(111)
#             line1, = self.ax.plot(self.x[:, 0], self.x[:, 1],
#                                   'bo', markersize=1)  # Returns a tuple of line objects, thus the comma
#             self.ax.plot([0], [0], 'kx')
#             plt.ylim(np.min(self.x[:, 1]) - 50, np.max(self.x[:, 1]) + 50)
#             plt.xlim(np.min(self.x[:, 0]) - 50, np.max(self.x[:, 0]) + 50)

#             # for i in range(self.NUMBER_OF_OBS):
#             #     phis = np.arange(0, np.pi*2, 0.01)
#             #     plt.plot(*xy(self.RK[i], phis, self.yk[i]), c='r', ls='-')

#             a = gca()
#             # a.set_xticklabels(a.get_xticks(), font)
#             # a.set_yticklabels(a.get_yticks(), font)
#             plt.title('GNN Controller {} agents'.format(self.n_agents))
#             self.fig = fig
#             self.line1 = line1
#         plt.ylim(np.min(self.x[:, 1]) - 50, np.max(self.x[:, 1]) + 50)
#         plt.xlim(np.min(self.x[:, 0]) - 50, np.max(self.x[:, 0]) + 50)
#         self.line1.set_xdata(self.x[:, 0])
#         self.line1.set_ydata(self.x[:, 1])
#         #         self.fig.canvas.draw()
#         #         self.fig.canvas.flush_events()
#         if mode == 'human':
#             self.fig.canvas.draw()
#             self.fig.canvas.flush_events()
#             return self.fig.canvas.draw()
#         else:
#             # test
#             buf = io.BytesIO()
#             self.fig.savefig(buf)
#             buf.seek(0)
#             im = np.asarray(Image.open(buf))
#             # buf.close()
#             return im
    

#     def plot_details(self, j=0, fname='', dir='plots', plot_leaders=False):
#         history_array = np.array(self.history)
#         u_history_array = np.array(self.u_history)
#         min_history_array = np.array(self.min_history)
#         state = self.x
#         plt.clf()
#         for i in range(self.n_agents):
#             plt.plot(history_array[:, i, 0], history_array[:, i, 1])
#             # plt.plot(self.Destination_x, self.Destination_y, 'rx', markersize=3)
#             if plot_leaders and i < self.n_leaders:
#                 plt.plot(state[i, 0], state[i, 1], 'ro', markersize=2)
#             else:
#                 plt.plot(state[i, 0], state[i, 1], 'bo', markersize=1)

#         # for i in range(self.NUMBER_OF_OBS):
#         #     phis = np.arange(0, np.pi*2, 0.01)
#         #     plt.plot(*xy(self.RK[i], phis, self.yk[i]), c='r', ls='-')
#         plt.savefig(f'./{dir}/{j}_test_{fname}_{self.n_agents}', dpi=150)
#         plt.close()
#         plt.clf()
#         for i in range(self.n_agents):
#             plt.plot(history_array[:, i, 2])
#         plt.savefig(f'./{dir}/{j}_test_vx_{fname}_{self.n_agents}', dpi=150)
#         plt.close()
#         plt.clf()
#         for i in range(self.n_agents):
#             plt.plot(history_array[:, i, 2])
#         plt.savefig(f'./{dir}/{j}_test_vy_{fname}_{self.n_agents}', dpi=150)
#         plt.close()
#         plt.clf()
#         for i in range(self.n_agents):
#             plt.plot(u_history_array[:, i, 0])
#         plt.savefig(f'./{dir}/{j}_{fname}_{self.n_agents}_test_action_x', dpi=150)
#         plt.close()
#         plt.clf()
#         for i in range(self.n_agents):
#             plt.plot(u_history_array[:, i, 1])
#         plt.savefig(f'./{dir}/{j}_{fname}_{self.n_agents}_test_action_y', dpi=150)
#         plt.close()
#         plt.clf()
#         plt.plot(min_history_array)
#         plt.savefig(f'./{dir}/{j}_{fname}_{self.n_agents}_test_min_dist', dpi=150)

#     def constrant_initialization(self):
#         x = np.zeros((self.n_agents, self.nx_system))
#         degree = 0
#         min_dist = 0
#         # min_dist_thresh = 0.1  # 0.25
#         min_dist_thresh = 0.5  # 0.25
#         v_bias = np.min([self.max_velocity, 10])
#         v_max = np.min([self.max_velocity, 10])
#         # generate an initial configuration with all agents connected,
#         # and minimum distance between agents > min_dist_thresh
#         while degree < 2 or min_dist < min_dist_thresh:

#             # randomly initialize the location and velocity of all agents
#             length = np.sqrt(np.random.uniform(
#                 0, 1*self.comm_radius*np.sqrt(self.n_agents), size=(self.n_agents,)))
#             angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
#             x[:, 0] = length * np.cos(angle)
#             x[:, 1] = length * np.sin(angle)

#             bias = np.random.uniform(
#                 low=-v_bias, high=v_bias, size=(2,))
#             x[:, 2] = np.random.uniform(
#                 low=-v_max, high=v_max, size=(self.n_agents,)) + bias[0]
#             x[:, 3] = np.random.uniform(
#                 low=-v_max, high=v_max, size=(self.n_agents,)) + bias[1]

#             # compute distances between agents
#             x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
#             a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
#                         np.transpose(x_loc, (2, 0, 1))), axis=2)
#             np.fill_diagonal(a_net, np.Inf)

#             # compute minimum distance between agents and degree of network to check if good initial configuration
#             min_dist = np.sqrt(np.min(np.min(a_net)))
#             a_net = a_net < self.comm_radius2
#             degree = np.min(np.sum(a_net.astype(int), axis=1))
#         # keep good initialization
#         # print('initialization successfully!')
#         self.mean_vel = np.mean(x[:, 2:4], axis=0)
#         self.init_vel = x[:, 2:4]
#         self.x = x