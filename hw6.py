import numpy as np
import matplotlib.pyplot as plt
# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
import copy
from utils import *

from scipy.stats import cumfreq

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


#load x_init
x_init=np.load('./data/x_init.npy')
print('no noise')

n_agents=4
nx_system=4

# get  simulations and compare with SRQ1 and SRQ2
TIMES_OUTTER=100
TIMES_INNER=100
simulation_sec=13.6
print('start simulation', TIMES_OUTTER, TIMES_INNER,simulation_sec)
obs_noise=lhsmdu.sample(1,TIMES_OUTTER)-0.5
obs_noise=(np.array(obs_noise)*5).flatten() # 


# plot the distribution of obs+noise
plt.hist(obs_noise,bins=10)
plt.title('Noise (LHS)')
plt.savefig('./plots/obs_noise.png')
plt.clf()
print('obs_noise',np.mean(obs_noise),np.std(obs_noise))

u_noise=np.random.normal(0,1,(n_agents,2)).flatten()
# plot the distribution of obs+noise
plt.hist(u_noise,bins=10)
plt.title('Noise (MC)')
plt.savefig('./plots/u_noise.png')
plt.clf()
u_noise=np.random.normal(0,1,(n_agents,2)).flatten()
# plot the distribution of obs+noise
plt.hist(u_noise,bins=10)
plt.title('Noise (MC)')
plt.savefig('./plots/u_noise1.png')
plt.clf()
results_min_distance=[]
results_velocity_diff_max=[]
for i in range(TIMES_OUTTER):
    min_distance_sim_SRQ=[]
    velocity_diff_max_sim_SRQ=[]
    for j in range(TIMES_INNER):
        # obs_noise=lhsmdu.sample(1,TIMES_OUTTER)-0.5
        # obs_noise=(np.array(obs_noise)*5).flatten() # 
        min_dist_srq, velocity_diff_max_srq = one_simulation_hw5(noise=obs_noise[i], fix_seed=False, simulation_sec=simulation_sec, save_fig=False,x_init=x_init,u_noise=1.0,dt_noise=0.004)
        min_distance_sim_SRQ.append(min_dist_srq)
        velocity_diff_max_sim_SRQ.append(velocity_diff_max_srq)
    min_distance_sim_SRQ=np.array(min_distance_sim_SRQ)
    velocity_diff_max_sim_SRQ=np.array(velocity_diff_max_sim_SRQ)
    results_min_distance.append(min_distance_sim_SRQ)
    results_velocity_diff_max.append(velocity_diff_max_sim_SRQ)
    plt.ecdf(min_distance_sim_SRQ, label='simulation')
# save the results
results_min_distance=np.array(results_min_distance)
results_velocity_diff_max=np.array(results_velocity_diff_max)
np.save(f'./data/results_min_distance_sim{TIMES_OUTTER}_{TIMES_INNER}_{simulation_sec}.npy',results_min_distance)
np.save(f'./data/results_velocity_diff_max_sim{TIMES_OUTTER}_{TIMES_INNER}_{simulation_sec}.npy',results_velocity_diff_max)

# print(min_distance_sim_SRQ)
# plot the CDF of min_distance_SRQ1 and min_distance_SRQ2


# plt.legend()
plt.ylim(-0.1,1.1)
plt.xlabel('min dist ')
plt.ylabel('CDF')
plt.title('Empirical CDF of min distance')
plt.grid()

plt.savefig(f'./plots/pbox_min_distance_sim{TIMES_OUTTER}_{TIMES_INNER}_{simulation_sec}.png')
plt.clf()


