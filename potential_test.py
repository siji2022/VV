# from flocking_relative_st import Flocking
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import *


# fix the random seed
np.random.seed(0)
n_agents=2
nx_system=4
dt=0.05

# get 100 points on x
x=np.linspace(0.6,5,1000)
ax=x
x=x # shift on x to right, 
u=1/x**2 + np.log(x)
grad=-2/x**3 + 2/(x)
plt.plot(ax,u,label='potential function')
plt.plot(ax,grad,label='gradient')
plt.grid()
plt.title('Potential function')
plt.xlabel('r')
plt.ylabel('u')
# plt.ylim(-30,10)
plt.legend()
plt.savefig('./plots/potential_func.png')

