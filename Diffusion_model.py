#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model

# here we develope a 1-d  model of diffusion.
# It assume a constant diffusivity. it uses a regular grid. it has a step function for an initial condition. it has fixed boundary conditions.

# Here is the diffusion equation:

# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$

# here is the discretized version of the diffusion equation we will solve with our model:

# 
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$

# This the FTCS scheme as described by Slingerland and Kump (2011):

# We will use libraries, Numpy (for arrays) matplotlib for plottng. that are not part of python distribution

# In[ ]:


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




