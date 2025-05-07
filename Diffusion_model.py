"""A 1D diffusion model."""

import numpy as np
import matplotlib.pyplot as plt


# start by setting two fixed model parameters, the diffusivity and the size of model domain

# In[ ]:


D=100
Lx=300


# Next , set up the model grid using a Numpy array

# In[ ]:


dx=0.5
x=np.arange(start=0, stop=Lx , step=dx)
nx= len(x)


# set the initial condition for the model. 

# In[ ]:


C=np.zeros_like(x)
C_left=500
C_right=0
C[x<=Lx/2]=C_left
C[x>Lx/2]=C_right


# Plot the initial profile

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("initial profile")


# Set time steps

# In[ ]:


nt=5000
dt=0.5*dx**2/D


# loop over the time step of model, sove the equation using FTCS

# In[ ]:


for t in range(0, nt):
    C[1:-1] += D*dt/dx**2* (C[:-2] -2*C[1:-1] + C[2:])


# plot the result

# In[ ]:


plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("final profile")


# In[ ]:




