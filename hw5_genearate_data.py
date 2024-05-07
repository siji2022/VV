import numpy as np
import matplotlib.pyplot as plt
# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
import copy
from utils import *

from scipy.stats import cumfreq

n_agents=15
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
# x_init=np.load('./data/x_init.npy')
x_init=None
print('no noise')
min_distance_SRQ_alpha, velocity_diff_max_SRQ_alpha = one_simulation_hw5(noise=0,fix_seed=True, save_fig=False,x_init=x_init,n_agents=n_agents)
print(min_distance_SRQ_alpha)

print('generate synthetic data')
# genearate_data.py
chi_1=np.array([0.55, 0.95, 1.0, 1.1, 1.5])
chi_2=np.array([0.1, 0.4, 0.6, 0.75, 0.8, 0.9, 0.91, 0.97, 1.3, 1.6])

min_distance_SRQ_alpha, velocity_diff_max_SRQ_alpha = one_simulation_hw5(noise=2.5,fix_seed=False, save_fig=False,x_init=x_init,n_agents=n_agents)
min_distance_SRQ_beta, velocity_diff_max_SRQ_beta = one_simulation_hw5(noise=-2.5,fix_seed=False, save_fig=False,x_init=x_init,n_agents=n_agents)

min_distance_SRQ1=min_distance_SRQ_alpha+chi_1*(min_distance_SRQ_beta-min_distance_SRQ_alpha)
min_distance_SRQ2=min_distance_SRQ_alpha+chi_2*(min_distance_SRQ_beta-min_distance_SRQ_alpha)

# n_agents=4
nx_system=4
min_distance_sim_SRQ=[]
velocity_diff_max_sim_SRQ=[]
# get  simulations and compare with SRQ1 and SRQ2
TIMES=100
print('start simulation', TIMES)
obs_noise=lhsmdu.sample(1,TIMES)-0.5
obs_noise=(np.array(obs_noise)*5).flatten() # 
# plot the distribution of obs+noise
plt.hist(obs_noise,bins=10)
plt.title('Noise (LHS)')
plt.savefig('./plots/obs_noise.png')
plt.clf()
print('obs_noise',np.mean(obs_noise),np.std(obs_noise))

for i in range(TIMES):
    
    min_dist_srq, velocity_diff_max_srq = one_simulation_hw5(noise=obs_noise[i], fix_seed=False, simulation_sec=13.6, save_fig=False,x_init=x_init,n_agents=n_agents)
    min_distance_sim_SRQ.append(min_dist_srq)
    velocity_diff_max_sim_SRQ.append(velocity_diff_max_srq)
min_distance_sim_SRQ=np.array(min_distance_sim_SRQ)
velocity_diff_max_sim_SRQ=np.array(velocity_diff_max_sim_SRQ)
# print(min_distance_sim_SRQ)
# plot the CDF of min_distance_SRQ1 and min_distance_SRQ2
fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs = np.ravel(axs)
axs[0].ecdf(min_distance_SRQ1, label='min_distance_SRQ1')
axs[0].ecdf(min_distance_sim_SRQ, label='simulation')
axs[1].ecdf(min_distance_SRQ2, label='min_distance_SRQ2')
axs[1].ecdf(min_distance_sim_SRQ, label='simulation')
axs[0].legend()
axs[1].legend()
axs[0].set_xlabel('min dist ')
axs[0].set_ylabel('CDF')
axs[0].set_title('Empirical CDF of min distance 1 ')
axs[0].grid()
axs[1].set_xlabel('min dist')
axs[1].grid()
axs[1].set_title('Empirical CDF of min distance 2')
plt.savefig(f'./plots/CDF_min_distance_sim{TIMES}.png')
plt.clf()

# I want to sample 100 times /ds=0.01
repeat_times=int(100/len(min_distance_SRQ1))
min_distance_SRQ1=np.repeat(min_distance_SRQ1,repeat_times)
repeat_times=int(100/len(min_distance_SRQ2))
min_distance_SRQ2=np.repeat(min_distance_SRQ2,repeat_times)
repeat_times=int(100/len(min_distance_sim_SRQ))
min_distance_sim_SRQ=np.repeat(min_distance_sim_SRQ,repeat_times)
avm1=calc_AVM(min_distance_SRQ1, min_distance_sim_SRQ)
avm2=calc_AVM(min_distance_SRQ2, min_distance_sim_SRQ)
print('avm1',avm1)
print('avm2',avm2)
